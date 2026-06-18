// Copyright 2017 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/dawn/native/opengl/PipelineGL.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#include "src/dawn/common/Range.h"
#include "src/dawn/native/BindGroupLayoutInternal.h"
#include "src/dawn/native/Blob.h"
#include "src/dawn/native/CacheKey.h"
#include "src/dawn/native/Device.h"
#include "src/dawn/native/Pipeline.h"
#include "src/dawn/native/Serializable.h"
#include "src/dawn/native/opengl/BufferGL.h"
#include "src/dawn/native/opengl/DeviceGL.h"
#include "src/dawn/native/opengl/Forward.h"
#include "src/dawn/native/opengl/OpenGLFunctions.h"
#include "src/dawn/native/opengl/PipelineLayoutGL.h"
#include "src/dawn/native/opengl/SamplerGL.h"
#include "src/dawn/native/opengl/ShaderModuleGL.h"
#include "src/dawn/native/opengl/TextureGL.h"
#include "src/dawn/native/opengl/UtilsGL.h"

namespace dawn::native::opengl {

namespace {

#define GL_PROGRAM_BINARY_MEMBERS(X) \
    X(GLenum, binaryFormat)          \
    X(std::vector<uint8_t>, binary)
DAWN_SERIALIZABLE(struct, GLProgramBinary, GL_PROGRAM_BINARY_MEMBERS) {
    static ResultOrError<GLProgramBinary> FromValidatedBlob(Blob blob) {
        GLProgramBinary result;
        DAWN_TRY_ASSIGN(result, GLProgramBinary::FromBlob(std::move(blob)));
        DAWN_INVALID_IF(result.binaryFormat == 0, "Cached OpenGL program binary has no format");
        DAWN_INVALID_IF(result.binary.empty(), "Cached OpenGL program binary is empty");
        return result;
    }
};
#undef GL_PROGRAM_BINARY_MEMBERS

struct TranslatedPipelineShader {
    SingleShaderStage stage;
    std::string glsl;
};

MaybeError WaitForProgramCompletion(const OpenGLFunctions& gl, GLuint program) {
    if (!gl.SupportsParallelShaderCompile()) {
        return {};
    }

    GLint completionStatus = GL_FALSE;
    do {
        DAWN_GL_TRY(gl, GetProgramiv(program, GL_COMPLETION_STATUS_KHR, &completionStatus));
        if (completionStatus == GL_FALSE) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } while (completionStatus == GL_FALSE);

    return {};
}

ResultOrError<bool> SupportsProgramBinaries(const OpenGLFunctions& gl) {
    if (gl.ProgramBinary == nullptr || gl.GetProgramBinary == nullptr ||
        gl.ProgramParameteri == nullptr) {
        return false;
    }

    GLint binaryFormatCount = 0;
    DAWN_GL_TRY(gl, GetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &binaryFormatCount));
    return binaryFormatCount > 0;
}

CacheKey CreateProgramBinaryCacheKey(DeviceBase* device,
                                     const wgpu::ShaderStage activeStages,
                                     const std::vector<TranslatedPipelineShader>& shaders) {
    CacheKey key = device->GetCacheKey();
    StreamIn(&key, std::string("OpenGL.ProgramBinary.v1"), activeStages);
    for (const TranslatedPipelineShader& shader : shaders) {
        StreamIn(&key, shader.stage, shader.glsl);
    }
    return key;
}

ResultOrError<bool> LoadProgramBinary(const OpenGLFunctions& gl,
                                      DeviceBase* device,
                                      const CacheKey& cacheKey,
                                      GLuint program) {
    Blob blob = device->LoadCachedBlob(cacheKey);
    if (blob.Empty()) {
        return false;
    }

    auto binaryResult = GLProgramBinary::FromValidatedBlob(std::move(blob));
    if (binaryResult.IsError()) {
        binaryResult.AcquireError();
        return false;
    }

    GLProgramBinary binary = binaryResult.AcquireSuccess();
    gl.ProgramBinary(program, binary.binaryFormat, binary.binary.data(),
                     dchecked_cast<GLsizei>(binary.binary.size()));
    if (gl.GetError() != GL_NO_ERROR) {
        return false;
    }

    GLint linkStatus = GL_FALSE;
    DAWN_GL_TRY(gl, GetProgramiv(program, GL_LINK_STATUS, &linkStatus));
    return linkStatus != GL_FALSE;
}

MaybeError StoreProgramBinary(const OpenGLFunctions& gl,
                              DeviceBase* device,
                              const CacheKey& cacheKey,
                              GLuint program) {
    GLint binaryLength = 0;
    DAWN_GL_TRY(gl, GetProgramiv(program, GL_PROGRAM_BINARY_LENGTH, &binaryLength));
    if (binaryLength <= 0) {
        return {};
    }

    GLProgramBinary binary;
    binary.binary.resize(binaryLength);
    GLsizei bytesWritten = 0;
    gl.GetProgramBinary(program, binaryLength, &bytesWritten, &binary.binaryFormat,
                        binary.binary.data());
    if (gl.GetError() != GL_NO_ERROR || bytesWritten <= 0) {
        return {};
    }

    binary.binary.resize(bytesWritten);
    device->StoreCachedBlob(cacheKey, binary.ToBlob());
    return {};
}

}  // anonymous namespace

PipelineGL::PipelineGL() : mProgram(0) {}

PipelineGL::~PipelineGL() = default;

MaybeError PipelineGL::InitializeBase(const OpenGLFunctions& gl,
                                      const PipelineLayout* layout,
                                      const PerStage<ProgrammableStage>& stages,
                                      ImmediateMask& pipelineImmediateMask,
                                      VertexAttributeMask bgraSwizzleAttributes,
                                      Extent3D* workgroupSize) {
    // Compute the set of active stages.
    wgpu::ShaderStage activeStages = wgpu::ShaderStage::None;
    for (SingleShaderStage stage : IterateStages(kAllStages)) {
        if (stages[stage].module != nullptr) {
            activeStages |= StageBit(stage);
        }
    }

    // Create an OpenGL shader for each stage and gather the list of combined samplers.
    std::set<CombinedSampler> combinedSamplers;
    mNeedsSSBOLengthUniformBuffer = false;
    std::vector<TranslatedPipelineShader> translatedShaders;
    EmulatedTextureBuiltinRegistrar emulatedTextureBuiltins(layout);
    for (SingleShaderStage stage : IterateStages(activeStages)) {
        ShaderModule* module = ToBackend(stages[stage].module.Get());
        bool needsSSBOLengthUniformBuffer = false;
        std::vector<CombinedSampler> stageCombinedSamplers;
        TranslatedShader translatedShader;
        DAWN_TRY_ASSIGN(
            translatedShader,
            module->TranslateToGLSL(gl, stages[stage], stage, pipelineImmediateMask,
                                    bgraSwizzleAttributes, &stageCombinedSamplers, layout,
                                    &emulatedTextureBuiltins, &needsSSBOLengthUniformBuffer));
        if (stage == SingleShaderStage::Compute) {
            *workgroupSize = translatedShader.workgroupSize;
        }

        mNeedsSSBOLengthUniformBuffer |= needsSSBOLengthUniformBuffer;
        combinedSamplers.insert(stageCombinedSamplers.begin(), stageCombinedSamplers.end());

        translatedShaders.push_back(
            {.stage = stage, .glsl = std::move(translatedShader.glsl)});
    }

    mEmulatedTextureBuiltinInfo = emulatedTextureBuiltins.AcquireInfo();

    DeviceBase* device = layout->GetDevice();
    CacheKey programBinaryCacheKey =
        CreateProgramBinaryCacheKey(device, activeStages, translatedShaders);

    bool canUseProgramBinaries = false;
    DAWN_TRY_ASSIGN(canUseProgramBinaries, SupportsProgramBinaries(gl));

    mProgram = DAWN_GL_TRY(gl, CreateProgram());
    bool loadedProgramBinary = false;
    if (canUseProgramBinaries) {
        DAWN_TRY_ASSIGN(loadedProgramBinary,
                        LoadProgramBinary(gl, device, programBinaryCacheKey, mProgram));
    }

    std::vector<GLuint> glShaders;
    if (!loadedProgramBinary) {
        DAWN_GL_TRY(gl, DeleteProgram(mProgram));
        mProgram = DAWN_GL_TRY(gl, CreateProgram());

        if (canUseProgramBinaries) {
            gl.ProgramParameteri(mProgram, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
            if (gl.GetError() != GL_NO_ERROR) {
                canUseProgramBinaries = false;
            }
        }

        for (const TranslatedPipelineShader& translatedShader : translatedShaders) {
            ShaderModule* module = ToBackend(stages[translatedShader.stage].module.Get());
            GLuint shader;
            DAWN_TRY_ASSIGN(shader,
                            module->CompileShader(gl, translatedShader.stage,
                                                  translatedShader.glsl));
            DAWN_GL_TRY(gl, AttachShader(mProgram, shader));
            glShaders.push_back(shader);
        }

        // Link all the shaders together.
        DAWN_GL_TRY(gl, LinkProgram(mProgram));
        DAWN_TRY(WaitForProgramCompletion(gl, mProgram));
    }

    GLint linkStatus = GL_FALSE;
    DAWN_GL_TRY(gl, GetProgramiv(mProgram, GL_LINK_STATUS, &linkStatus));
    if (linkStatus == GL_FALSE) {
        GLint infoLogLength = 0;
        DAWN_GL_TRY(gl, GetProgramiv(mProgram, GL_INFO_LOG_LENGTH, &infoLogLength));

        if (infoLogLength > 1) {
            std::vector<char> buffer(infoLogLength);
            DAWN_GL_TRY(gl, GetProgramInfoLog(mProgram, infoLogLength, nullptr, &buffer[0]));
            return DAWN_VALIDATION_ERROR("Program link failed:\n%s", buffer.data());
        }
    }

    if (!loadedProgramBinary && canUseProgramBinaries) {
        DAWN_TRY(StoreProgramBinary(gl, device, programBinaryCacheKey, mProgram));
    }

    // Compute links between stages for combined samplers, then bind them to texture units
    DAWN_GL_TRY(gl, UseProgram(mProgram));
    const auto& indices = layout->GetBindingIndexInfo();

    mUnitsForSamplers.resize(layout->GetNumSamplers());
    mUnitsForTextures.resize(layout->GetNumSampledTextures());

    // Assign combined texture/samplers to GL texture units.
    TextureUnit textureUnit{0u};
    for (const auto& combined : combinedSamplers) {
        // All the texture/samplers of a binding_array are set in a single glUniform1iv, gather them
        // all in this vector.
        absl::InlinedVector<GLint, 1> uniformsToSet;

        BindingIndex textureArrayStart = combined.textureLocation.index;
        for (auto textureArrayElement : Range(combined.textureLocation.shaderArraySize)) {
            FlatBindingIndex textureGLIndex =
                indices[combined.textureLocation.group][textureArrayStart + textureArrayElement];
            mUnitsForTextures[textureGLIndex].push_back(textureUnit);

            // Record that the placeholder sampler must be set for this texture unit if no sampler
            // is used in the shader.
            if (!combined.samplerLocation) {
                mPlaceholderSamplerUnits.push_back(textureUnit);
            } else {
                // Record that the sampler used in the shader must be set for this texture unit.
                BindingIndex samplerBindingIndex = combined.samplerLocation->index;
                FlatBindingIndex samplerGLIndex =
                    indices[combined.samplerLocation->group][samplerBindingIndex];
                mUnitsForSamplers[samplerGLIndex].push_back(textureUnit);
            }

            uniformsToSet.push_back(dchecked_cast<GLint>(textureUnit));
            textureUnit++;
        }

        std::string name = combined.GetName();
        GLint location = DAWN_GL_TRY(gl, GetUniformLocation(mProgram, name.c_str()));
        // Non-arrayed GLSL variables cannot be set with glUniform1iv
        if (uniformsToSet.size() == 1) {
            DAWN_GL_TRY(gl, Uniform1i(location, uniformsToSet[0]));
        } else {
            DAWN_GL_TRY(gl, Uniform1iv(location, uniformsToSet.size(), uniformsToSet.data()));
        }
    }

    if (!mPlaceholderSamplerUnits.empty()) {
        Ref<SamplerBase> sampler;
        DAWN_TRY_ASSIGN(sampler, layout->GetDevice()->CreateSampler());
        mPlaceholderSampler = ToBackend(std::move(sampler));
    }

    // If the pipeline declares immediates but the GL driver determines that they are unused and
    // optimizes out the uniform variable, reset the mask. This prevents a GL_INVALID_VALUE error
    // when trying to update it via glUniform*().
    if (pipelineImmediateMask.any()) {
        auto location = DAWN_GL_TRY(gl, GetUniformLocation(mProgram, "tint_immediates"));
        if (location == -1) {
            pipelineImmediateMask.reset();
        }
    }

    if (!loadedProgramBinary) {
        for (GLuint glShader : glShaders) {
            DAWN_GL_TRY(gl, DetachShader(mProgram, glShader));
            DAWN_GL_TRY(gl, DeleteShader(glShader));
        }
    }

    return {};
}

const std::vector<TextureUnit>& PipelineGL::GetTextureUnitsForSampler(
    FlatBindingIndex index) const {
    DAWN_ASSERT(index < mUnitsForSamplers.size());
    return mUnitsForSamplers[index];
}

const std::vector<TextureUnit>& PipelineGL::GetTextureUnitsForTextureView(
    FlatBindingIndex index) const {
    DAWN_ASSERT(index < mUnitsForTextures.size());
    return mUnitsForTextures[index];
}

MaybeError PipelineGL::ApplyNow(const OpenGLFunctions& gl, const PipelineLayout* layout) {
    DAWN_GL_TRY(gl, UseProgram(mProgram));
    for (TextureUnit unit : mPlaceholderSamplerUnits) {
        DAWN_ASSERT(mPlaceholderSampler.Get() != nullptr);
        DAWN_GL_TRY(gl, BindSampler(GLuint(unit), mPlaceholderSampler->GetHandle()));
    }

    return {};
}

const EmulatedTextureBuiltinInfo& PipelineGL::GetEmulatedTextureBuiltinInfo() const {
    return mEmulatedTextureBuiltinInfo;
}

bool PipelineGL::NeedsTextureBuiltinUniformBuffer() const {
    return !mEmulatedTextureBuiltinInfo.empty();
}

bool PipelineGL::NeedsSSBOLengthUniformBuffer() const {
    return mNeedsSSBOLengthUniformBuffer;
}

// EmulatedTextureBuiltinRegistrar

EmulatedTextureBuiltinRegistrar::EmulatedTextureBuiltinRegistrar(const PipelineLayout* layout)
    : mLayout(layout) {}

uint32_t EmulatedTextureBuiltinRegistrar::Register(BindGroupIndex group,
                                                   BindingIndex binding,
                                                   TextureQuery query) {
    FlatBindingIndex firstTextureIndex = mLayout->GetBindingIndexInfo()[group][binding];

    if (!mEmulatedTextureBuiltinInfo.contains(firstTextureIndex)) {
        // Register the metadata to add for each element of the binding_array (if there is one).
        BindingIndex arraySize =
            mLayout->GetBindGroupLayout(group)->GetBindingInfo(binding).arraySize;
        for (BindingIndex arrayElement : Range(arraySize)) {
            FlatBindingIndex textureIndex =
                mLayout->GetBindingIndexInfo()[group][binding + arrayElement];

            mEmulatedTextureBuiltinInfo.emplace(
                textureIndex,
                EmulatedTextureBuiltin{.index = mCurrentIndex, .query = query, .group = group});
            mCurrentIndex++;
        }
    }

    return mEmulatedTextureBuiltinInfo[firstTextureIndex].index;
}

EmulatedTextureBuiltinInfo EmulatedTextureBuiltinRegistrar::AcquireInfo() {
    return mEmulatedTextureBuiltinInfo;
}

}  // namespace dawn::native::opengl
