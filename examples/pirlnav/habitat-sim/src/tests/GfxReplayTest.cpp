// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "configure.h"

#include <Corrade/Containers/Optional.h>
#include <Corrade/TestSuite/Compare/Numeric.h>
#include <Corrade/TestSuite/Tester.h>
#include <Corrade/Utility/Path.h>
#include <Magnum/EigenIntegration/Integration.h>
#include <Magnum/Math/Range.h>

#include "esp/assets/RenderAssetInstanceCreationInfo.h"
#include "esp/assets/ResourceManager.h"
#include "esp/gfx/Renderer.h"
#include "esp/gfx/WindowlessContext.h"
#include "esp/gfx/replay/Player.h"
#include "esp/gfx/replay/Recorder.h"
#include "esp/gfx/replay/ReplayManager.h"
#include "esp/scene/SceneManager.h"
#include "esp/sim/Simulator.h"

#include <fstream>
#include <string>

namespace Cr = Corrade;
namespace Mn = Magnum;

using esp::assets::ResourceManager;
using esp::metadata::MetadataMediator;
using esp::scene::SceneManager;
using esp::sim::Simulator;
using esp::sim::SimulatorConfiguration;

namespace {

struct GfxReplayTest : Cr::TestSuite::Tester {
  explicit GfxReplayTest();

  void testRecorder();

  void testPlayer();

  void testPlayerReadMissingFile();
  void testPlayerReadInvalidFile();
  void testSimulatorIntegration();

  esp::logging::LoggingContext loggingContext;

};  // struct GfxReplayTest

// Helper function to get numberOfChildrenOfRoot
int getNumberOfChildrenOfRoot(esp::scene::SceneNode& rootNode) {
  int numberOfChildrenOfRoot = 1;
  const auto* lastRootChild = rootNode.children().first();
  if (!lastRootChild) {
    return 0;
  } else {
    CORRADE_VERIFY(lastRootChild);
    while (lastRootChild->nextSibling()) {
      lastRootChild = lastRootChild->nextSibling();
      ++numberOfChildrenOfRoot;
    }
  }
  return numberOfChildrenOfRoot;
}

GfxReplayTest::GfxReplayTest() {
  addTests({&GfxReplayTest::testRecorder, &GfxReplayTest::testPlayer,
            &GfxReplayTest::testPlayerReadMissingFile,
            &GfxReplayTest::testPlayerReadInvalidFile,
            &GfxReplayTest::testSimulatorIntegration});
}  // ctor

// Manipulate the scene and save some keyframes using replay::Recorder
void GfxReplayTest::testRecorder() {
  esp::gfx::WindowlessContext::uptr context_ =
      esp::gfx::WindowlessContext::create_unique(0);

  std::shared_ptr<esp::gfx::Renderer> renderer_ = esp::gfx::Renderer::create();

  auto cfg = esp::sim::SimulatorConfiguration{};
  auto MM = MetadataMediator::create(cfg);
  // must declare these in this order due to avoid deallocation errors
  ResourceManager resourceManager(MM);
  SceneManager sceneManager_;
  std::string boxFile =
      Cr::Utility::Path::join(TEST_ASSETS, "objects/transform_box.glb");

  int sceneID = sceneManager_.initSceneGraph();
  auto& sceneGraph = sceneManager_.getSceneGraph(sceneID);
  const esp::assets::AssetInfo info = esp::assets::AssetInfo::fromPath(boxFile);

  const std::string lightSetupKey = "";
  esp::assets::RenderAssetInstanceCreationInfo::Flags flags;
  flags |= esp::assets::RenderAssetInstanceCreationInfo::Flag::IsRGBD;
  flags |= esp::assets::RenderAssetInstanceCreationInfo::Flag::IsSemantic;
  esp::assets::RenderAssetInstanceCreationInfo creation(
      boxFile, Corrade::Containers::NullOpt, flags, lightSetupKey);

  std::vector<int> tempIDs{sceneID, esp::ID_UNDEFINED};
  auto* node = resourceManager.loadAndCreateRenderAssetInstance(
      info, creation, &sceneManager_, tempIDs);
  CORRADE_VERIFY(node);

  // cosntruct an AssetInfo with override color material
  CORRADE_VERIFY(!info.overridePhongMaterial);
  esp::assets::AssetInfo info2(info);
  // change shadertype to make sure change is registered and retained through
  // save/read of replay data
  info2.shaderTypeToUse =
      esp::metadata::attributes::ObjectInstanceShaderType::Flat;
  info2.overridePhongMaterial = esp::assets::PhongMaterialColor();
  info2.overridePhongMaterial->ambientColor = Mn::Color4(0.1, 0.2, 0.3, 0.4);
  info2.overridePhongMaterial->diffuseColor = Mn::Color4(0.2, 0.3, 0.4, 0.5);
  info2.overridePhongMaterial->specularColor = Mn::Color4(0.3, 0.4, 0.5, 0.6);

  esp::gfx::replay::Recorder recorder;
  recorder.onLoadRenderAsset(info);
  recorder.onCreateRenderAssetInstance(node, creation);
  recorder.saveKeyframe();
  node->setTranslation(Mn::Vector3(1.f, 2.f, 3.f));
  node->setSemanticId(7);

  // add the new override AssetInfo after 1st keyframe
  auto* node2 = resourceManager.loadAndCreateRenderAssetInstance(
      info2, creation, &sceneManager_, tempIDs);
  CORRADE_VERIFY(node2);
  recorder.onLoadRenderAsset(info2);
  recorder.onCreateRenderAssetInstance(node2, creation);

  recorder.saveKeyframe();
  delete node;
  recorder.addUserTransformToKeyframe("my_user_transform",
                                      Mn::Vector3(4.f, 5.f, 6.f),
                                      Mn::Quaternion(Mn::Math::IdentityInit));
  recorder.saveKeyframe();

  // verify 3 saved keyframes
  const auto& keyframes = recorder.debugGetSavedKeyframes();
  CORRADE_COMPARE(keyframes.size(), 3);

  // verify frame #0 loads a render asset, creates an instance, and stores a
  // state update for the instance
  CORRADE_COMPARE(keyframes[0].loads.size(), 1);
  CORRADE_VERIFY(keyframes[0].loads[0] == info);
  CORRADE_COMPARE(keyframes[0].creations.size(), 1);
  CORRADE_VERIFY(keyframes[0].creations[0].second.filepath.find(
                     "objects/transform_box.glb") != std::string::npos);
  CORRADE_COMPARE(keyframes[0].stateUpdates.size(), 1);
  esp::gfx::replay::RenderAssetInstanceKey instanceKey =
      keyframes[0].creations[0].first;
  CORRADE_COMPARE(keyframes[0].stateUpdates[0].first, instanceKey);

  // verify frame #1 has an updated state for node and state for new node2
  CORRADE_COMPARE(keyframes[1].stateUpdates.size(), 2);
  // verify frame #1 has our translation and semantic Id
  CORRADE_COMPARE(keyframes[1].stateUpdates[0].second.absTransform.translation,
                  Mn::Vector3(1.f, 2.f, 3.f));
  CORRADE_COMPARE(keyframes[1].stateUpdates[0].second.semanticId, 7);

  // verify override material AssetInfo is loaded correctly
  CORRADE_COMPARE(keyframes[1].loads.size(), 1);
  CORRADE_VERIFY(keyframes[1].loads[0] == info2);
  CORRADE_VERIFY(keyframes[1].loads[0].overridePhongMaterial);
  CORRADE_COMPARE(keyframes[1].loads[0].overridePhongMaterial->ambientColor,
                  info2.overridePhongMaterial->ambientColor);
  CORRADE_COMPARE(keyframes[1].loads[0].overridePhongMaterial->diffuseColor,
                  info2.overridePhongMaterial->diffuseColor);
  CORRADE_COMPARE(keyframes[1].loads[0].overridePhongMaterial->specularColor,
                  info2.overridePhongMaterial->specularColor);

  // verify frame #2 has our deletion and our user transform
  CORRADE_COMPARE(keyframes[2].deletions.size(), 1);
  CORRADE_COMPARE(keyframes[2].deletions[0], instanceKey);
  CORRADE_COMPARE(keyframes[2].userTransforms.size(), 1);
  CORRADE_VERIFY(keyframes[2].userTransforms.count("my_user_transform"));
  CORRADE_COMPARE(
      keyframes[2].userTransforms.at("my_user_transform").translation,
      Mn::Vector3(4.f, 5.f, 6.f));
}

// construct some render keyframes and play them using replay::Player
void GfxReplayTest::testPlayer() {
  esp::logging::LoggingContext loggingContext;
  esp::gfx::WindowlessContext::uptr context_ =
      esp::gfx::WindowlessContext::create_unique(0);

  std::shared_ptr<esp::gfx::Renderer> renderer_ = esp::gfx::Renderer::create();

  auto cfg = esp::sim::SimulatorConfiguration{};
  auto MM = MetadataMediator::create(cfg);
  // must declare these in this order due to avoid deallocation errors
  ResourceManager resourceManager(MM);
  SceneManager sceneManager_;
  std::string boxFile =
      Cr::Utility::Path::join(TEST_ASSETS, "objects/transform_box.glb");

  int sceneID = sceneManager_.initSceneGraph();
  auto& sceneGraph = sceneManager_.getSceneGraph(sceneID);

  // retrieve last child of scene root node
  auto& rootNode = sceneGraph.getRootNode();
  int numberOfChildren = getNumberOfChildrenOfRoot(rootNode);

  // Construct Player. Hook up ResourceManager::loadAndCreateRenderAssetInstance
  // to Player via callback.
  auto callback =
      [&](const esp::assets::AssetInfo& assetInfo,
          const esp::assets::RenderAssetInstanceCreationInfo& creation) {
        std::vector<int> tempIDs{sceneID, esp::ID_UNDEFINED};
        return resourceManager.loadAndCreateRenderAssetInstance(
            assetInfo, creation, &sceneManager_, tempIDs);
      };
  esp::gfx::replay::Player player(callback);

  std::vector<esp::gfx::replay::Keyframe> keyframes;

  esp::assets::AssetInfo info = esp::assets::AssetInfo::fromPath(boxFile);
  esp::gfx::replay::RenderAssetInstanceKey instanceKey = 7;

  const std::string lightSetupKey = "";
  esp::assets::RenderAssetInstanceCreationInfo::Flags flags;
  flags |= esp::assets::RenderAssetInstanceCreationInfo::Flag::IsRGBD;
  flags |= esp::assets::RenderAssetInstanceCreationInfo::Flag::IsSemantic;
  esp::assets::RenderAssetInstanceCreationInfo creation(
      boxFile, Corrade::Containers::NullOpt, flags, lightSetupKey);

  /*
  // Keyframe struct shown here for reference
  struct Keyframe {
    std::vector<esp::assets::AssetInfo> loads;
    std::vector<std::pair<RenderAssetInstanceKey,
                          esp::assets::RenderAssetInstanceCreationInfo>>
        creations;
    std::vector<RenderAssetInstanceKey> deletions;
    std::vector<std::pair<RenderAssetInstanceKey, RenderAssetInstanceState>>
        stateUpdates;
    std::unordered_map<std::string, Transform> userTransforms;
  };
  */

  // keyframe #0: load a render asset and create a render asset instance
  keyframes.emplace_back(esp::gfx::replay::Keyframe{
      {info}, {{instanceKey, creation}}, {}, {}, {}});

  constexpr int semanticId = 4;
  esp::gfx::replay::RenderAssetInstanceState stateUpdate{
      {Mn::Vector3(1.f, 2.f, 3.f), Mn::Quaternion(Mn::Math::IdentityInit)},
      semanticId};

  // keyframe #1: a state update
  keyframes.emplace_back(
      esp::gfx::replay::Keyframe{{}, {}, {}, {{instanceKey, stateUpdate}}, {}});

  // keyframe #2: delete instance
  keyframes.emplace_back(
      esp::gfx::replay::Keyframe{{}, {}, {instanceKey}, {}, {}});

  // keyframe #3: include a user transform
  keyframes.emplace_back(
      esp::gfx::replay::Keyframe{{},
                                 {},
                                 {},
                                 {},
                                 {{"my_user_transform",
                                   {Mn::Vector3(4.f, 5.f, 6.f),
                                    Mn::Quaternion(Mn::Math::IdentityInit)}}}});

  player.debugSetKeyframes(std::move(keyframes));

  CORRADE_COMPARE(player.getNumKeyframes(), 4);
  CORRADE_COMPARE(player.getKeyframeIndex(), -1);

  // test setting keyframes in various order
  const auto keyframeIndicesToTest = {-1, 0, 1,  2, 3,  -1, 3, 2,
                                      1,  0, -1, 1, -1, 2,  0};

  for (const auto keyframeIndex : keyframeIndicesToTest) {
    player.setKeyframeIndex(keyframeIndex);
    int updatedNumberOfChildren = getNumberOfChildrenOfRoot(rootNode);

    if (keyframeIndex == -1) {
      // assert that no new nodes were created
      CORRADE_COMPARE(updatedNumberOfChildren, numberOfChildren);
    } else if (keyframeIndex == 0) {
      // assert that a new node was created under root
      CORRADE_COMPARE_AS(updatedNumberOfChildren, numberOfChildren,
                         Cr::TestSuite::Compare::Greater);
    } else if (keyframeIndex == 1) {
      // assert that our stateUpdate was applied and a new node was created
      // under root
      CORRADE_COMPARE_AS(updatedNumberOfChildren, numberOfChildren,
                         Cr::TestSuite::Compare::Greater);
      if (numberOfChildren ==
          0) {  // if rootNode had no children to begin with, then stateUpdate
                // was applied to a new child node
        const auto* rootChild = rootNode.children().first();
        CORRADE_VERIFY(rootChild);
        const esp::scene::SceneNode* instanceNode =
            static_cast<const esp::scene::SceneNode*>(rootChild);
        CORRADE_COMPARE(instanceNode->translation(),
                        Mn::Vector3(1.f, 2.f, 3.f));
        CORRADE_COMPARE(instanceNode->getSemanticId(), semanticId);
      } else {  // if rootNode had children originally, then stateUpdate was
                // applied to a sibling of the lastRootChild
        // get the lastRootChild before the stateUpdate
        const auto* rootChild = rootNode.children().first();
        for (int i = 1; i < numberOfChildren; ++i) {
          CORRADE_VERIFY(rootChild);
          rootChild = rootChild->nextSibling();
        }
        // Check that stateUpdate was applied to the sibling of lastRootChild
        CORRADE_VERIFY(rootChild->nextSibling());
        const esp::scene::SceneNode* instanceNode =
            static_cast<const esp::scene::SceneNode*>(rootChild->nextSibling());
        CORRADE_COMPARE(instanceNode->translation(),
                        Mn::Vector3(1.f, 2.f, 3.f));
        CORRADE_COMPARE(instanceNode->getSemanticId(), semanticId);
      }
    } else if (keyframeIndex == 2) {
      // assert that no new nodes were created
      CORRADE_COMPARE(updatedNumberOfChildren, numberOfChildren);
      // assert that there's no user transform
      Mn::Vector3 userTranslation;
      Mn::Quaternion userRotation;
      CORRADE_VERIFY(!player.getUserTransform("my_user_transform",
                                              &userTranslation, &userRotation));
    } else if (keyframeIndex == 3) {
      // assert that no new nodes were created
      CORRADE_COMPARE(updatedNumberOfChildren, numberOfChildren);
      // assert on expected user transform
      Mn::Vector3 userTranslation;
      Mn::Quaternion userRotation;
      CORRADE_VERIFY(player.getUserTransform("my_user_transform",
                                             &userTranslation, &userRotation));
      CORRADE_COMPARE(userTranslation, Mn::Vector3(4.f, 5.f, 6.f));
    }
  }
}

void GfxReplayTest::testPlayerReadMissingFile() {
  auto dummyCallback =
      [&](const esp::assets::AssetInfo& assetInfo,
          const esp::assets::RenderAssetInstanceCreationInfo& creation) {
        return nullptr;
      };
  esp::gfx::replay::Player player(dummyCallback);

  player.readKeyframesFromFile("file_that_does_not_exist.json");
  CORRADE_COMPARE(player.getNumKeyframes(), 0);
}

void GfxReplayTest::testPlayerReadInvalidFile() {
  esp::logging::LoggingContext loggingContext;
  auto testFilepath =
      Corrade::Utility::Path::join(DATA_DIR, "./gfx_replay_test.json");

  std::ofstream out(testFilepath);
  out << "{invalid json";
  out.close();

  auto dummyCallback =
      [&](const esp::assets::AssetInfo& assetInfo,
          const esp::assets::RenderAssetInstanceCreationInfo& creation) {
        return nullptr;
      };
  esp::gfx::replay::Player player(dummyCallback);

  player.readKeyframesFromFile(testFilepath);
  CORRADE_COMPARE(player.getNumKeyframes(), 0);

  // remove bogus file created for this test
  bool success = Corrade::Utility::Path::remove(testFilepath);
  if (!success) {
    ESP_WARNING() << "Unable to remove temporary test JSON file"
                  << testFilepath;
  }
}

// test recording and playback through the simulator interface
void GfxReplayTest::testSimulatorIntegration() {
  std::string boxFile =
      Cr::Utility::Path::join(TEST_ASSETS, "objects/transform_box.glb");
  auto testFilepath =
      Corrade::Utility::Path::join(DATA_DIR, "./gfx_replay_test.json");

  SimulatorConfiguration simConfig{};
  simConfig.activeSceneName = boxFile;
  simConfig.enableGfxReplaySave = true;
  simConfig.createRenderer = false;

  auto sim = Simulator::create_unique(simConfig);
  auto objAttrMgr = sim->getObjectAttributesManager();
  objAttrMgr->loadAllJSONConfigsFromPath(
      Cr::Utility::Path::join(TEST_ASSETS, "objects/nested_box"), true);

  auto handles = objAttrMgr->getObjectHandlesBySubstring("nested_box");
  CORRADE_VERIFY(!handles.empty());
  auto rigidObj =
      sim->getRigidObjectManager()->addBulletObjectByHandle(handles[0]);
  auto rigidObjTranslation = Mn::Vector3(1.f, 2.f, 3.f);
  auto rigidObjRotation = Mn::Quaternion::rotation(
      Mn::Deg(45.f), Mn::Vector3(1.f, 1.f, 0.f).normalized());
  rigidObj->setTranslation(rigidObjTranslation);
  rigidObj->setRotation(rigidObjRotation);

  auto& sceneGraph = sim->getActiveSceneGraph();
  auto& rootNode = sceneGraph.getRootNode();
  auto prevNumberOfChildrenOfRoot = getNumberOfChildrenOfRoot(rootNode);

  const auto recorder = sim->getGfxReplayManager()->getRecorder();
  CORRADE_VERIFY(recorder);
  recorder->saveKeyframe();
  recorder->writeSavedKeyframesToFile(testFilepath);

  auto player = sim->getGfxReplayManager()->readKeyframesFromFile(testFilepath);
  CORRADE_VERIFY(player);
  CORRADE_COMPARE(player->getNumKeyframes(), 1);
  player->setKeyframeIndex(0);
  // second copies of transform_box and nested_box were loaded
  CORRADE_COMPARE(getNumberOfChildrenOfRoot(rootNode),
                  prevNumberOfChildrenOfRoot + 2);

  const auto& keyframes = player->debugGetKeyframes();
  // we expect state updates for the state and the object instance
  CORRADE_COMPARE(keyframes[0].stateUpdates.size(), 2);
  // check the pose for nested_box
  const auto& stateUpdate = keyframes[0].stateUpdates[1];
  const auto transform = stateUpdate.second.absTransform;
  CORRADE_COMPARE_AS((transform.translation - rigidObjTranslation).length(),
                     1.0e-5, Cr::TestSuite::Compare::LessOrEqual);
  CORRADE_COMPARE_AS(
      (transform.rotation.vector() - rigidObjRotation.vector()).length(),
      1.0e-5, Cr::TestSuite::Compare::LessOrEqual);

  player = nullptr;
  // second copies of transform_box and nested_box are removed when Player
  // is deleted
  CORRADE_COMPARE(getNumberOfChildrenOfRoot(rootNode),
                  prevNumberOfChildrenOfRoot);

  // remove file created for this test
  bool success = Corrade::Utility::Path::remove(testFilepath);
  if (!success) {
    ESP_WARNING() << "Unable to remove temporary test JSON file"
                  << testFilepath;
  }
}

}  // namespace

CORRADE_TEST_MAIN(GfxReplayTest)
