// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "DoubleSphereCameraShader.h"

#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Assert.h>
#include <Corrade/Utility/FormatStl.h>
#include <Corrade/Utility/Resource.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/Version.h>

#include <sstream>

// This is to import the "resources" at runtime. When the resource is
// compiled into static library, it must be explicitly initialized via this
// macro, and should be called *outside* of any namespace.
static void importShaderResources() {
  CORRADE_RESOURCE_INITIALIZE(ShaderResources)
}

namespace Mn = Magnum;
namespace Cr = Corrade;

namespace esp {
namespace gfx {
DoubleSphereCameraShader::DoubleSphereCameraShader(
    CubeMapShaderBase::Flags flags)
    : CubeMapShaderBase(flags) {
  if (!Cr::Utility::Resource::hasGroup("default-shaders")) {
    importShaderResources();
  }

#ifdef MAGNUM_TARGET_WEBGL
  Mn::GL::Version glVersion = Mn::GL::Version::GLES300;
#else
  Mn::GL::Version glVersion = Mn::GL::Version::GL410;
#endif

  // this is not the file name, but the group name in the config file
  // see Shaders.conf in the shaders folder
  const Cr::Utility::Resource rs{"default-shaders"};

  Mn::GL::Shader vert{glVersion, Mn::GL::Shader::Type::Vertex};
  Mn::GL::Shader frag{glVersion, Mn::GL::Shader::Type::Fragment};

  // Add macros
  vert.addSource(rs.getString("bigTriangle.vert"));

  std::stringstream outputAttributeLocationsStream;

  if (flags_ & CubeMapShaderBase::Flag::ColorTexture) {
    outputAttributeLocationsStream << Cr::Utility::formatString(
        "#define OUTPUT_ATTRIBUTE_LOCATION_COLOR {}\n", ColorOutput);
  }
  if (flags_ & CubeMapShaderBase::Flag::ObjectIdTexture) {
    outputAttributeLocationsStream << Cr::Utility::formatString(
        "#define OUTPUT_ATTRIBUTE_LOCATION_OBJECT_ID {}\n", ObjectIdOutput);
  }

  frag.addSource(outputAttributeLocationsStream.str())
      .addSource(flags_ & CubeMapShaderBase::Flag::ColorTexture
                     ? "#define COLOR_TEXTURE\n"
                     : "")
      .addSource(flags_ & CubeMapShaderBase::Flag::DepthTexture
                     ? "#define DEPTH_TEXTURE\n"
                     : "")
      .addSource(flags_ & CubeMapShaderBase::Flag::ObjectIdTexture
                     ? "#define OBJECT_ID_TEXTURE\n"
                     : "")
      .addSource(rs.getString("doubleSphereCamera.frag"));

  CORRADE_INTERNAL_ASSERT_OUTPUT(Mn::GL::Shader::compile({vert, frag}));

  attachShaders({vert, frag});

  CORRADE_INTERNAL_ASSERT_OUTPUT(link());

  // set texture binding points in the shader
  if (flags_ & CubeMapShaderBase::Flag::ColorTexture) {
    CORRADE_INTERNAL_ASSERT(uniformLocation("ColorTexture") >= 0);
    setUniform(uniformLocation("ColorTexture"),
               CubeMapShaderBaseTexUnitSpace::TextureUnit::Color);
  }
  if (flags_ & CubeMapShaderBase::Flag::DepthTexture) {
    CORRADE_INTERNAL_ASSERT(uniformLocation("DepthTexture") >= 0);
    setUniform(uniformLocation("DepthTexture"),
               CubeMapShaderBaseTexUnitSpace::TextureUnit::Depth);
  }
  if (flags_ & CubeMapShaderBase::Flag::ObjectIdTexture) {
    CORRADE_INTERNAL_ASSERT(uniformLocation("ObjectIdTexture") >= 0);
    setUniform(uniformLocation("ObjectIdTexture"),
               CubeMapShaderBaseTexUnitSpace::TextureUnit::ObjectId);
  }

  // cache the uniform locations
  // it hurts the performance to call glGetUniformLocation() every frame due
  // to string operations. therefore, cache the locations in the constructor

  focalLengthUniform_ = uniformLocation("FocalLength");
  CORRADE_INTERNAL_ASSERT(focalLengthUniform_ >= 0);
  principalPointOffsetUniform_ = uniformLocation("PrincipalPointOffset");
  CORRADE_INTERNAL_ASSERT(principalPointOffsetUniform_ >= 0);
  alphaUniform_ = uniformLocation("Alpha");
  CORRADE_INTERNAL_ASSERT(alphaUniform_ >= 0);
  xiUniform_ = uniformLocation("Xi");
  CORRADE_INTERNAL_ASSERT(xiUniform_ >= 0);
}

DoubleSphereCameraShader& DoubleSphereCameraShader::setFocalLength(
    Magnum::Vector2 focalLength) {
  setUniform(focalLengthUniform_, focalLength);
  return *this;
}

DoubleSphereCameraShader& DoubleSphereCameraShader::setPrincipalPointOffset(
    Magnum::Vector2 offset) {
  setUniform(principalPointOffsetUniform_, offset);
  return *this;
}

DoubleSphereCameraShader& DoubleSphereCameraShader::setAlpha(float alpha) {
  setUniform(alphaUniform_, alpha);
  return *this;
}

DoubleSphereCameraShader& DoubleSphereCameraShader::setXi(float xi) {
  setUniform(xiUniform_, xi);
  return *this;
}

}  // namespace gfx
}  // namespace esp
