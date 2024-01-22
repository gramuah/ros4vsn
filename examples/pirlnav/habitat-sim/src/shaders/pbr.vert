// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

precision highp float;

// ------------ input ------------------------
// position, normal, tangent in model local space
layout(location = ATTRIBUTE_LOCATION_POSITION) in highp vec4 vertexPosition;
layout(location = ATTRIBUTE_LOCATION_NORMAL) in highp vec3 vertexNormal;
#if defined(TEXTURED)
layout(location = ATTRIBUTE_LOCATION_TEXCOORD) in highp vec2 vertexTexCoord;
#endif
#if defined(NORMAL_TEXTURE) && defined(PRECOMPUTED_TANGENT)
layout(location = ATTRIBUTE_LOCATION_TANGENT4) in highp vec4 vertexTangent;
#endif

// -------------- output ---------------------
// position, normal, tangent in *world* space, NOT camera space!
out highp vec3 position;
out highp vec3 normal;
#if defined(TEXTURED)
out highp vec2 texCoord;
#endif
#if defined(NORMAL_TEXTURE) && defined(PRECOMPUTED_TANGENT)
out highp vec3 tangent;
out highp vec3 biTangent;
#endif

// ------------ uniform ----------------------
uniform highp mat4 ViewMatrix;
uniform highp mat3 NormalMatrix;  // inverse transpose of 3x3 model matrix, NOT modelview matrix
uniform highp mat4 ModelMatrix;
uniform highp mat4 ProjectionMatrix;

#ifdef TEXTURE_TRANSFORMATION
uniform highp mat3 TextureMatrix
#ifndef GL_ES
    = mat3(1.0)
#endif
    ;
#endif

// ------------ shader -----------------------
void main() {
  vec4 vertexWorldPosition = ModelMatrix * vertexPosition;
  position = vertexWorldPosition.xyz;
  normal = normalize(NormalMatrix * vertexNormal);
#if defined(TEXTURED)
  texCoord =
#if defined(TEXTURE_TRANSFORMATION)
      (TextureMatrix * vec3(vertexTexCoord, 1.0)).xy;
#else
      vertexTexCoord;
#endif  // TEXTURE_TRANSFORMATION
#endif  // TEXTURED

#if defined(NORMAL_TEXTURE) && defined(PRECOMPUTED_TANGENT)
  tangent = normalize(NormalMatrix * vec3(vertexTangent));
  // Gram–Schmidt
  tangent = normalize(tangent - dot(tangent, normal) * normal);
  biTangent = normalize(cross(normal, tangent) * vertexTangent.w);
  // later in .frag, TBN will transform the normal perturbation
  // (read from normal map) from tangent space to world space,
  // NOT camera space
#endif

  gl_Position = ProjectionMatrix * ViewMatrix * vertexWorldPosition;
}
