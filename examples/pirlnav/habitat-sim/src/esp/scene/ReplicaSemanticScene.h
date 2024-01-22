// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ESP_SCENE_REPLICASEMANTICSCENE_H_
#define ESP_SCENE_REPLICASEMANTICSCENE_H_

#include "SemanticScene.h"

namespace esp {
namespace scene {

struct ReplicaObjectCategory : public SemanticCategory {
  ReplicaObjectCategory(const int id, const std::string& name)
      : id_(id), name_(name) {}

  int index(const std::string& /*mapping*/) const override { return id_; }

  std::string name(const std::string& mapping) const override {
    if (mapping == "category" || mapping == "") {
      return name_;
    } else {
      ESP_ERROR() << "Unknown mapping type:" << mapping;
      return "UNKNOWN";
    }
  }

 protected:
  int id_;
  std::string name_;
  friend SemanticScene;

  ESP_SMART_POINTERS(ReplicaObjectCategory)
};

}  // namespace scene
}  // namespace esp

#endif  // ESP_SCENE_REPLICASEMANTICSCENE_H_
