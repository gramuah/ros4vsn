// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "RenderAssetInstanceCreationInfo.h"

namespace esp {
namespace assets {

RenderAssetInstanceCreationInfo::RenderAssetInstanceCreationInfo(
    const std::string& _filepath,
    const Corrade::Containers::Optional<Magnum::Vector3>& _scale,
    const Flags& _flags,
    const std::string& _lightSetupKey)
    : filepath(_filepath),
      scale(_scale),
      flags(_flags),
      lightSetupKey(_lightSetupKey) {}

}  // namespace assets
}  // namespace esp
