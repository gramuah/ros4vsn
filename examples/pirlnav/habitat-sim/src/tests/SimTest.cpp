// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <Corrade/Containers/StridedArrayView.h>
#include <Corrade/TestSuite/Compare/Numeric.h>
#include <Corrade/TestSuite/Tester.h>
#include <Corrade/Utility/Path.h>
#include <Magnum/DebugTools/CompareImage.h>
#include <Magnum/EigenIntegration/Integration.h>
#include <Magnum/ImageView.h>
#include <Magnum/Magnum.h>
#include <Magnum/PixelFormat.h>
#include <string>

#include "esp/assets/ResourceManager.h"
#include "esp/physics/RigidObject.h"
#include "esp/sensor/CameraSensor.h"
#include "esp/sim/Simulator.h"

#include "configure.h"

namespace Cr = Corrade;
namespace Mn = Magnum;

using esp::agent::Agent;
using esp::agent::AgentConfiguration;
using esp::agent::AgentState;
using esp::assets::ResourceManager;
using esp::gfx::LightInfo;
using esp::gfx::LightPositionModel;
using esp::gfx::LightSetup;
using esp::metadata::MetadataMediator;
using esp::metadata::attributes::AbstractPrimitiveAttributes;
using esp::metadata::attributes::ObjectAttributes;
using esp::nav::PathFinder;
using esp::sensor::CameraSensor;
using esp::sensor::CameraSensorSpec;
using esp::sensor::Observation;
using esp::sensor::ObservationSpace;
using esp::sensor::ObservationSpaceType;
using esp::sensor::SensorType;
using esp::sim::Simulator;
using esp::sim::SimulatorConfiguration;

namespace {
using namespace Magnum::Math::Literals;

const std::string vangogh =
    Cr::Utility::Path::join(SCENE_DATASETS,
                            "habitat-test-scenes/van-gogh-room.glb");
const std::string skokloster =
    Cr::Utility::Path::join(SCENE_DATASETS,
                            "habitat-test-scenes/skokloster-castle.glb");
const std::string planeStage =
    Cr::Utility::Path::join(TEST_ASSETS, "scenes/plane.glb");
const std::string physicsConfigFile =
    Cr::Utility::Path::join(TEST_ASSETS, "testing.physics_config.json");
const std::string screenshotDir =
    Cr::Utility::Path::join(TEST_ASSETS, "screenshots/");

struct SimTest : Cr::TestSuite::Tester {
  explicit SimTest();

  //! build a simulator via a SimulatorConfiguration alone
  static Simulator::uptr getSimulator(
      SimTest& self,
      const std::string& scene,
      const std::string& sceneLightingKey = esp::NO_LIGHT_KEY) {
    SimulatorConfiguration simConfig{};
    simConfig.activeSceneName = scene;
    simConfig.enablePhysics = true;
    simConfig.physicsConfigFile = physicsConfigFile;
    simConfig.overrideSceneLightDefaults = true;
    simConfig.sceneLightSetupKey = sceneLightingKey;

    auto sim = Simulator::create_unique(simConfig);
    auto objAttrMgr = sim->getObjectAttributesManager();
    objAttrMgr->loadAllJSONConfigsFromPath(
        Cr::Utility::Path::join(TEST_ASSETS, "objects/nested_box"), true);

    sim->setLightSetup(self.lightSetup1, "custom_lighting_1");
    sim->setLightSetup(self.lightSetup2, "custom_lighting_2");
    return sim;
  }

  //! build a simulator via an instanced Metadata Mediator
  static Simulator::uptr getSimulatorMM(
      SimTest& self,
      const std::string& scene,
      const std::string& sceneLightingKey = esp::NO_LIGHT_KEY) {
    SimulatorConfiguration simConfig{};
    simConfig.activeSceneName = scene;
    simConfig.enablePhysics = true;
    simConfig.physicsConfigFile = physicsConfigFile;
    simConfig.overrideSceneLightDefaults = true;
    simConfig.sceneLightSetupKey = sceneLightingKey;

    MetadataMediator::ptr MM = MetadataMediator::create(simConfig);
    auto sim = Simulator::create_unique(simConfig, MM);
    auto objAttrMgr = sim->getObjectAttributesManager();
    objAttrMgr->loadAllJSONConfigsFromPath(
        Cr::Utility::Path::join(TEST_ASSETS, "objects/nested_box"), true);

    sim->setLightSetup(self.lightSetup1, "custom_lighting_1");
    sim->setLightSetup(self.lightSetup2, "custom_lighting_2");
    return sim;
  }
  void checkPinholeCameraRGBAObservation(
      Simulator& sim,
      const std::string& groundTruthImageFile,
      Magnum::Float maxThreshold,
      Magnum::Float meanThreshold);

  void basic();
  void reconfigure();
  void reset();
  void getSceneRGBAObservation();
  void getSceneWithLightingRGBAObservation();
  void getDefaultLightingRGBAObservation();
  void getCustomLightingRGBAObservation();
  void updateLightSetupRGBAObservation();
  void updateObjectLightSetupRGBAObservation();
  void multipleLightingSetupsRGBAObservation();
  void recomputeNavmeshWithStaticObjects();
  void loadingObjectTemplates();
  void buildingPrimAssetObjectTemplates();
  void addObjectByHandle();
  void addSensorToObject();
  void createMagnumRenderingOff();

  esp::logging::LoggingContext loggingContext_;
  // TODO: remove outlier pixels from image and lower maxThreshold
  const Magnum::Float maxThreshold = 255.f;

  LightSetup lightSetup1{{Magnum::Vector4{1.0f, 1.5f, 0.5f, 0.0f},
                          {5.0, 5.0, 0.0},
                          LightPositionModel::Camera}};
  LightSetup lightSetup2{{Magnum::Vector4{0.0f, 0.5f, 1.0f, 0.0f},
                          {0.0, 5.0, 5.0},
                          LightPositionModel::Camera}};
};  // struct SimTest

struct {
  // display name for sim being tested
  const char* name;
  // function pointer to constructor to simulator
  Simulator::uptr (*creator)(SimTest& self,
                             const std::string& scene,
                             const std::string& sceneLightingKey);

} SimulatorBuilder[]{{"built with SimConfig", &SimTest::getSimulator},
                     {"built with MetadataMediator", &SimTest::getSimulatorMM}};
SimTest::SimTest() {
  // clang-format off
  //test instances test both mechanisms for constructing simulator
  addInstancedTests({
            &SimTest::basic,
            &SimTest::reconfigure,
            &SimTest::reset,
            &SimTest::getSceneRGBAObservation,
            &SimTest::getSceneWithLightingRGBAObservation,
            &SimTest::getDefaultLightingRGBAObservation,
            &SimTest::getCustomLightingRGBAObservation,
            &SimTest::updateLightSetupRGBAObservation,
            &SimTest::updateObjectLightSetupRGBAObservation,
            &SimTest::multipleLightingSetupsRGBAObservation,
            &SimTest::recomputeNavmeshWithStaticObjects,
            &SimTest::loadingObjectTemplates,
            &SimTest::buildingPrimAssetObjectTemplates,
            &SimTest::addObjectByHandle,
            &SimTest::addSensorToObject,
            &SimTest::createMagnumRenderingOff}, Cr::Containers::arraySize(SimulatorBuilder) );
  // clang-format on
}
void SimTest::basic() {
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, vangogh, esp::NO_LIGHT_KEY);
  PathFinder::ptr pathfinder = simulator->getPathFinder();
  CORRADE_VERIFY(pathfinder);
}

void SimTest::reconfigure() {
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, vangogh, esp::NO_LIGHT_KEY);
  PathFinder::ptr pathfinder = simulator->getPathFinder();
  SimulatorConfiguration cfg =
      simulator->getMetadataMediator()->getSimulatorConfiguration();
  simulator->reconfigure(cfg);
  CORRADE_COMPARE(pathfinder, simulator->getPathFinder());
  SimulatorConfiguration cfg2;
  cfg2.activeSceneName = skokloster;
  simulator->reconfigure(cfg2);
  CORRADE_VERIFY(pathfinder != simulator->getPathFinder());
}

void SimTest::reset() {
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, vangogh, esp::NO_LIGHT_KEY);

  PathFinder::ptr pathfinder = simulator->getPathFinder();
  auto pinholeCameraSpec = CameraSensorSpec::create();
  pinholeCameraSpec->sensorSubType = esp::sensor::SensorSubType::Pinhole;
  pinholeCameraSpec->sensorType = SensorType::Color;
  pinholeCameraSpec->position = {0.0f, 1.5f, 5.0f};
  pinholeCameraSpec->resolution = {100, 100};
  AgentConfiguration agentConfig{};
  agentConfig.sensorSpecifications = {pinholeCameraSpec};
  auto agent = simulator->addAgent(agentConfig);

  auto stateOrig = AgentState::create();
  agent->getState(stateOrig);

  simulator->reset();

  auto stateFinal = AgentState::create();
  agent->getState(stateFinal);
  CORRADE_COMPARE(stateOrig->position, stateFinal->position);
  CORRADE_COMPARE(stateOrig->rotation, stateFinal->rotation);
  CORRADE_COMPARE(pathfinder, simulator->getPathFinder());
}

void SimTest::checkPinholeCameraRGBAObservation(
    Simulator& simulator,
    const std::string& groundTruthImageFile,
    Magnum::Float maxThreshold,
    Magnum::Float meanThreshold) {
  // do not rely on default SensorSpec default constructor to remain constant
  auto pinholeCameraSpec = CameraSensorSpec::create();
  pinholeCameraSpec->sensorSubType = esp::sensor::SensorSubType::Pinhole;
  pinholeCameraSpec->sensorType = SensorType::Color;
  pinholeCameraSpec->position = {1.0f, 1.5f, 1.0f};
  pinholeCameraSpec->resolution = {128, 128};

  AgentConfiguration agentConfig{};
  agentConfig.sensorSpecifications = {pinholeCameraSpec};
  Agent::ptr agent = simulator.addAgent(agentConfig);
  agent->setInitialState(AgentState{});

  Observation observation;
  ObservationSpace obsSpace;
  CORRADE_VERIFY(
      simulator.getAgentObservation(0, pinholeCameraSpec->uuid, observation));
  CORRADE_VERIFY(
      simulator.getAgentObservationSpace(0, pinholeCameraSpec->uuid, obsSpace));

  std::vector<size_t> expectedShape{
      {static_cast<size_t>(pinholeCameraSpec->resolution[0]),
       static_cast<size_t>(pinholeCameraSpec->resolution[1]), 4}};

  CORRADE_VERIFY(obsSpace.spaceType == ObservationSpaceType::Tensor);
  CORRADE_VERIFY(obsSpace.dataType == esp::core::DataType::DT_UINT8);
  CORRADE_COMPARE(obsSpace.shape, expectedShape);
  CORRADE_COMPARE(observation.buffer->shape, expectedShape);

  // Compare with previously rendered ground truth
  CORRADE_COMPARE_WITH(
      (Mn::ImageView2D{
          Mn::PixelFormat::RGBA8Unorm,
          {pinholeCameraSpec->resolution[0], pinholeCameraSpec->resolution[1]},
          observation.buffer->data}),
      Cr::Utility::Path::join(screenshotDir, groundTruthImageFile),
      (Mn::DebugTools::CompareImageToFile{maxThreshold, meanThreshold}));
}

void SimTest::getSceneRGBAObservation() {
  ESP_DEBUG() << "Starting Test : getSceneRGBAObservation";
  setTestCaseName(CORRADE_FUNCTION);
  ESP_DEBUG() << "About to build simulator";
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, vangogh, esp::NO_LIGHT_KEY);
  ESP_DEBUG() << "Built simulator";
  checkPinholeCameraRGBAObservation(*simulator, "SimTestExpectedScene.png",
                                    maxThreshold, 0.75f);
}

void SimTest::getSceneWithLightingRGBAObservation() {
  ESP_DEBUG() << "Starting Test : getSceneWithLightingRGBAObservation";
  setTestCaseName(CORRADE_FUNCTION);
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, vangogh, "custom_lighting_1");
  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedSceneWithLighting.png", maxThreshold, 0.75f);
}

void SimTest::getDefaultLightingRGBAObservation() {
  ESP_DEBUG() << "Starting Test : getDefaultLightingRGBAObservation";
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, vangogh, esp::NO_LIGHT_KEY);
  // manager of object attributes
  auto objectAttribsMgr = simulator->getObjectAttributesManager();
  auto rigidObjMgr = simulator->getRigidObjectManager();
  auto objs = objectAttribsMgr->getObjectHandlesBySubstring("nested_box");
  auto obj = rigidObjMgr->addObjectByHandle(objs[0]);
  CORRADE_VERIFY(obj->isAlive());
  CORRADE_VERIFY(obj->getID() != esp::ID_UNDEFINED);
  obj->setTranslation({1.0f, 0.5f, -0.5f});
  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedDefaultLighting.png", maxThreshold, 0.71f);
}

void SimTest::getCustomLightingRGBAObservation() {
  ESP_DEBUG() << "Starting Test : getCustomLightingRGBAObservation";
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, vangogh, esp::NO_LIGHT_KEY);
  // manager of object attributes
  auto objectAttribsMgr = simulator->getObjectAttributesManager();
  auto rigidObjMgr = simulator->getRigidObjectManager();
  auto objs = objectAttribsMgr->getObjectHandlesBySubstring("nested_box");
  auto obj =
      rigidObjMgr->addObjectByHandle(objs[0], nullptr, "custom_lighting_1");
  CORRADE_VERIFY(obj->isAlive());
  CORRADE_VERIFY(obj->getID() != esp::ID_UNDEFINED);
  obj->setTranslation({1.0f, 0.5f, -0.5f});

  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedCustomLighting.png", maxThreshold, 0.71f);
}

void SimTest::updateLightSetupRGBAObservation() {
  ESP_DEBUG() << "Starting Test : updateLightSetupRGBAObservation";
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, vangogh, esp::NO_LIGHT_KEY);
  // manager of object attributes
  auto objectAttribsMgr = simulator->getObjectAttributesManager();
  auto rigidObjMgr = simulator->getRigidObjectManager();
  // update default lighting
  auto objs = objectAttribsMgr->getObjectHandlesBySubstring("nested_box");
  auto obj = rigidObjMgr->addObjectByHandle(objs[0]);
  CORRADE_VERIFY(obj->isAlive());
  CORRADE_VERIFY(obj->getID() != esp::ID_UNDEFINED);
  obj->setTranslation({1.0f, 0.5f, -0.5f});

  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedDefaultLighting.png", maxThreshold, 0.71f);

  simulator->setLightSetup(lightSetup1);
  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedCustomLighting.png", maxThreshold, 0.71f);
  rigidObjMgr->removePhysObjectByHandle(obj->getHandle());

  // update custom lighting
  obj = rigidObjMgr->addObjectByHandle(objs[0], nullptr, "custom_lighting_1");
  CORRADE_VERIFY(obj->isAlive());
  CORRADE_VERIFY(obj->getID() != esp::ID_UNDEFINED);
  obj->setTranslation({1.0f, 0.5f, -0.5f});

  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedCustomLighting.png", maxThreshold, 0.71f);

  simulator->setLightSetup(lightSetup2, "custom_lighting_1");
  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedCustomLighting2.png", maxThreshold, 0.71f);
}

void SimTest::updateObjectLightSetupRGBAObservation() {
  ESP_DEBUG() << "Starting Test : updateObjectLightSetupRGBAObservation";
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, vangogh, esp::NO_LIGHT_KEY);
  // manager of object attributes
  auto objectAttribsMgr = simulator->getObjectAttributesManager();
  auto rigidObjMgr = simulator->getRigidObjectManager();
  auto objs = objectAttribsMgr->getObjectHandlesBySubstring("nested_box");
  auto obj = rigidObjMgr->addObjectByHandle(objs[0]);
  CORRADE_VERIFY(obj->isAlive());
  CORRADE_VERIFY(obj->getID() != esp::ID_UNDEFINED);
  obj->setTranslation({1.0f, 0.5f, -0.5f});
  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedDefaultLighting.png", maxThreshold, 0.71f);

  // change from default lighting to custom
  obj->setLightSetup("custom_lighting_1");
  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedCustomLighting.png", maxThreshold, 0.71f);

  // change from one custom lighting to another
  obj->setLightSetup("custom_lighting_2");
  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedCustomLighting2.png", maxThreshold, 0.71f);
}

void SimTest::multipleLightingSetupsRGBAObservation() {
  ESP_DEBUG() << "Starting Test : multipleLightingSetupsRGBAObservation";
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, planeStage, esp::NO_LIGHT_KEY);
  // manager of object attributes
  auto objectAttribsMgr = simulator->getObjectAttributesManager();
  auto rigidObjMgr = simulator->getRigidObjectManager();
  // make sure updates apply to all objects using the light setup
  auto objs = objectAttribsMgr->getObjectHandlesBySubstring("nested_box");
  auto obj =
      rigidObjMgr->addObjectByHandle(objs[0], nullptr, "custom_lighting_1");
  CORRADE_VERIFY(obj->isAlive());
  CORRADE_VERIFY(obj->getID() != esp::ID_UNDEFINED);
  obj->setTranslation({0.0f, 0.5f, -0.5f});

  auto otherObj =
      rigidObjMgr->addObjectByHandle(objs[0], nullptr, "custom_lighting_1");
  CORRADE_VERIFY(otherObj->isAlive());
  CORRADE_VERIFY(otherObj->getID() != esp::ID_UNDEFINED);
  otherObj->setTranslation({2.0f, 0.5f, -0.5f});

  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedSameLighting.png", maxThreshold, 0.01f);

  simulator->setLightSetup(lightSetup2, "custom_lighting_1");
  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedSameLighting2.png", maxThreshold, 0.01f);
  simulator->setLightSetup(lightSetup1, "custom_lighting_1");

  // make sure we can move a single object to another group
  obj->setLightSetup("custom_lighting_2");
  checkPinholeCameraRGBAObservation(
      *simulator, "SimTestExpectedDifferentLighting.png", maxThreshold, 0.01f);
}

void SimTest::recomputeNavmeshWithStaticObjects() {
  ESP_DEBUG() << "Starting Test : recomputeNavmeshWithStaticObjects";
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, skokloster, esp::NO_LIGHT_KEY);
  // manager of object attributes
  auto objectAttribsMgr = simulator->getObjectAttributesManager();
  auto rigidObjMgr = simulator->getRigidObjectManager();

  // compute the initial navmesh
  esp::nav::NavMeshSettings navMeshSettings;
  navMeshSettings.setDefaults();
  simulator->recomputeNavMesh(*simulator->getPathFinder().get(),
                              navMeshSettings);

  esp::vec3f randomNavPoint =
      simulator->getPathFinder()->getRandomNavigablePoint();
  while (simulator->getPathFinder()->distanceToClosestObstacle(randomNavPoint) <
             1.0 ||
         randomNavPoint[1] > 1.0) {
    randomNavPoint = simulator->getPathFinder()->getRandomNavigablePoint();
  }

  // add static object at a known navigable point
  auto objs = objectAttribsMgr->getObjectHandlesBySubstring("nested_box");
  auto obj = rigidObjMgr->addObjectByHandle(objs[0]);
  obj->setTranslation(Magnum::Vector3{randomNavPoint});
  obj->setMotionType(esp::physics::MotionType::STATIC);
  CORRADE_VERIFY(
      simulator->getPathFinder()->isNavigable({randomNavPoint}, 0.1));

  // recompute with object
  simulator->recomputeNavMesh(*simulator->getPathFinder().get(),
                              navMeshSettings, true);
  CORRADE_VERIFY(!simulator->getPathFinder()->isNavigable(randomNavPoint, 0.1));

  // recompute without again
  simulator->recomputeNavMesh(*simulator->getPathFinder().get(),
                              navMeshSettings, false);
  CORRADE_VERIFY(simulator->getPathFinder()->isNavigable(randomNavPoint, 0.1));

  rigidObjMgr->removePhysObjectByHandle(obj->getHandle());

  // test scaling
  ObjectAttributes::ptr objectTemplate = objectAttribsMgr->getObjectCopyByID(0);
  objectTemplate->setScale({0.5, 0.5, 0.5});
  int tmplateID = objectAttribsMgr->registerObject(objectTemplate);

  obj = rigidObjMgr->addObjectByHandle(objs[0]);
  obj->setTranslation(Magnum::Vector3{randomNavPoint});
  obj->setTranslation(obj->getTranslation() + Magnum::Vector3{0, 0.5, 0});
  obj->setMotionType(esp::physics::MotionType::STATIC);
  esp::vec3f offset(0.75, 0, 0);
  CORRADE_VERIFY(simulator->getPathFinder()->isNavigable(randomNavPoint, 0.1));
  CORRADE_VERIFY(
      simulator->getPathFinder()->isNavigable(randomNavPoint + offset, 0.2));
  // recompute with object
  simulator->recomputeNavMesh(*simulator->getPathFinder().get(),
                              navMeshSettings, true);
  CORRADE_VERIFY(!simulator->getPathFinder()->isNavigable(randomNavPoint, 0.1));
  CORRADE_VERIFY(
      simulator->getPathFinder()->isNavigable(randomNavPoint + offset, 0.2));
}

void SimTest::loadingObjectTemplates() {
  ESP_DEBUG() << "Starting Test : loadingObjectTemplates";
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, planeStage, esp::NO_LIGHT_KEY);
  // manager of object attributes
  auto objectAttribsMgr = simulator->getObjectAttributesManager();

  // test directory of templates
  std::vector<int> templateIndices =
      objectAttribsMgr->loadAllJSONConfigsFromPath(
          Cr::Utility::Path::join(TEST_ASSETS, "objects"));
  CORRADE_VERIFY(!templateIndices.empty());
  for (auto index : templateIndices) {
    CORRADE_VERIFY(index != esp::ID_UNDEFINED);
  }

  // reload again and ensure that old loaded indices are returned
  std::vector<int> templateIndices2 =
      objectAttribsMgr->loadAllJSONConfigsFromPath(
          Cr::Utility::Path::join(TEST_ASSETS, "objects"));
  CORRADE_COMPARE(templateIndices2, templateIndices);

  // test the loaded assets and accessing them by name
  // verify that getting the template handles with empty string returns all
  int numLoadedTemplates = templateIndices2.size();
  std::vector<std::string> templateHandles =
      objectAttribsMgr->getFileTemplateHandlesBySubstring();
  CORRADE_COMPARE(numLoadedTemplates, templateHandles.size());

  // verify that querying with sub string returns template handle corresponding
  // to that substring
  // get full handle of an existing template
  std::string fullTmpHndl = templateHandles[templateHandles.size() - 1];
  // build substring
  std::div_t len = std::div(fullTmpHndl.length(), 2);
  // get 2nd half of handle
  std::string tmpHndl = fullTmpHndl.substr(len.quot);
  // get all handles that match 2nd half of known handle
  std::vector<std::string> matchTmpltHandles =
      objectAttribsMgr->getObjectHandlesBySubstring(tmpHndl);
  CORRADE_COMPARE(matchTmpltHandles[0], fullTmpHndl);

  // test fresh template as smart pointer
  ObjectAttributes::ptr newTemplate =
      objectAttribsMgr->createObject("new template", false);
  std::string boxPath =
      Cr::Utility::Path::join(TEST_ASSETS, "objects/transform_box.glb");
  newTemplate->setRenderAssetHandle(boxPath);
  int templateIndex = objectAttribsMgr->registerObject(newTemplate, boxPath);

  CORRADE_VERIFY(templateIndex != esp::ID_UNDEFINED);
  // change render asset for object template named boxPath
  std::string chairPath =
      Cr::Utility::Path::join(TEST_ASSETS, "objects/chair.glb");
  newTemplate->setRenderAssetHandle(chairPath);
  int templateIndex2 = objectAttribsMgr->registerObject(newTemplate, boxPath);

  CORRADE_VERIFY(templateIndex2 != esp::ID_UNDEFINED);
  CORRADE_COMPARE(templateIndex2, templateIndex);
  ObjectAttributes::ptr newTemplate2 =
      objectAttribsMgr->getObjectCopyByHandle(boxPath);
  CORRADE_COMPARE(newTemplate2->getRenderAssetHandle(), chairPath);
}

void SimTest::buildingPrimAssetObjectTemplates() {
  ESP_DEBUG() << "Starting Test : buildingPrimAssetObjectTemplates";
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, planeStage, esp::NO_LIGHT_KEY);

  // test that the correct number of default primitive assets are available as
  // render/collision targets
  // manager of asset attributes
  auto assetAttribsMgr = simulator->getAssetAttributesManager();
  // manager of object attributes
  auto objectAttribsMgr = simulator->getObjectAttributesManager();
  auto rigidObjMgr = simulator->getRigidObjectManager();

  // get all handles of templates for primitive-based render objects
  std::vector<std::string> primObjAssetHandles =
      objectAttribsMgr->getSynthTemplateHandlesBySubstring("");

  // there should be 1 prim template per default primitive asset template
  int numPrimsExpected =
      static_cast<int>(esp::metadata::PrimObjTypes::END_PRIM_OBJ_TYPES);
  // verify the number of primitive templates
  CORRADE_COMPARE(numPrimsExpected, primObjAssetHandles.size());

  AbstractPrimitiveAttributes::ptr primAttr;
  {
    // test that there are existing templates for each key, and that they have
    // valid values to be used to construct magnum primitives
    for (int i = 0; i < numPrimsExpected; ++i) {
      std::string handle = primObjAssetHandles[i];
      CORRADE_VERIFY(!handle.empty());
      primAttr = assetAttribsMgr->getObjectCopyByHandle(handle);
      CORRADE_VERIFY(primAttr);
      CORRADE_VERIFY(primAttr->isValidTemplate());
      // verify that the attributes contains the handle, and the handle contains
      // the expected class name
      std::string className =
          esp::metadata::managers::AssetAttributesManager::PrimitiveNames3DMap
              .at(static_cast<esp::metadata::PrimObjTypes>(
                  primAttr->getPrimObjType()));

      CORRADE_COMPARE(primAttr->getHandle(), handle);
      CORRADE_VERIFY(handle.find(className) != std::string::npos);
    }
  }
  // empty vector of handles
  primObjAssetHandles.clear();
  {
    // test that existing template handles can be accessed via name string.
    // This access is case insensitive
    primObjAssetHandles =
        objectAttribsMgr->getSynthTemplateHandlesBySubstring("CONESOLID");
    // should only be one handle in this vector
    CORRADE_COMPARE(primObjAssetHandles.size(), 1);
    // handle should not be empty and be long enough to hold class name prefix
    CORRADE_COMPARE_AS(primObjAssetHandles[0].length(), 9,
                       Cr::TestSuite::Compare::Greater);
    // coneSolid should appear in handle
    std::string checkStr("coneSolid");
    CORRADE_VERIFY(primObjAssetHandles[0].find(checkStr) != std::string::npos);
    // empty vector of handles
    primObjAssetHandles.clear();

    // test that existing template handles can be accessed through exclusion -
    // all but certain string.  This access is case insensitive
    primObjAssetHandles = objectAttribsMgr->getSynthTemplateHandlesBySubstring(
        "CONESOLID", false);
    // should be all handles but coneSolid handle here
    CORRADE_COMPARE((numPrimsExpected - 1), primObjAssetHandles.size());
    for (auto primObjAssetHandle : primObjAssetHandles) {
      CORRADE_COMPARE(primObjAssetHandle.find(checkStr), std::string::npos);
    }
  }
  // empty vector of handles
  primObjAssetHandles.clear();

  // test that primitive asset attributes are able to be modified and saved and
  // the changes persist, while the old templates are not removed
  {
    // get existing default cylinder handle
    primObjAssetHandles = assetAttribsMgr->getTemplateHandlesByPrimType(
        esp::metadata::PrimObjTypes::CYLINDER_SOLID);
    // should only be one handle in this vector
    CORRADE_COMPARE(primObjAssetHandles.size(), 1);
    // primitive render object uses primitive render asset as handle
    std::string origCylinderHandle = primObjAssetHandles[0];
    primAttr = assetAttribsMgr->getObjectCopyByHandle(origCylinderHandle);
    // verify that the origin handle matches what is expected
    CORRADE_COMPARE(primAttr->getHandle(), origCylinderHandle);
    // get original number of rings for this cylinder
    int origNumRings = primAttr->getNumRings();
    // modify attributes - this will change handle
    primAttr->setNumRings(2 * origNumRings);
    // verify that internal name of attributes has changed due to essential
    // quantity being modified
    std::string newHandle = primAttr->getHandle();

    CORRADE_COMPARE_AS(newHandle, origCylinderHandle,
                       Cr::TestSuite::Compare::NotEqual);
    // set bogus file directory, to validate that copy is reggistered
    primAttr->setFileDirectory("test0");
    // register new attributes
    int idx = assetAttribsMgr->registerObject(primAttr);

    CORRADE_VERIFY(idx != esp::ID_UNDEFINED);
    // set new test label, to validate against retrieved copy
    primAttr->setFileDirectory("test1");
    // retrieve registered attributes copy
    AbstractPrimitiveAttributes::ptr primAttr2 =
        assetAttribsMgr->getObjectCopyByHandle(newHandle);
    // verify pre-reg and post-reg are named the same
    CORRADE_COMPARE(primAttr->getHandle(), primAttr2->getHandle());
    // verify retrieved attributes is copy, not original

    CORRADE_COMPARE_AS(primAttr->getFileDirectory(),
                       primAttr2->getFileDirectory(),
                       Cr::TestSuite::Compare::NotEqual);
    // remove modified attributes
    AbstractPrimitiveAttributes::ptr primAttr3 =
        assetAttribsMgr->removeObjectByHandle(newHandle);
    CORRADE_VERIFY(primAttr3);
  }
  // empty vector of handles
  primObjAssetHandles.clear();
  {
    // test creation of new object, using edited attributes
    // get existing default cylinder handle
    primObjAssetHandles = assetAttribsMgr->getTemplateHandlesByPrimType(
        esp::metadata::PrimObjTypes::CYLINDER_SOLID);
    // primitive render object uses primitive render asset as handle
    std::string origCylinderHandle = primObjAssetHandles[0];
    primAttr = assetAttribsMgr->getObjectCopyByHandle(origCylinderHandle);
    // modify attributes - this will change handle
    primAttr->setNumRings(2 * primAttr->getNumRings());
    // verify that internal name of attributes has changed due to essential
    // quantity being modified
    std::string newHandle = primAttr->getHandle();
    // register new attributes
    int idx = assetAttribsMgr->registerObject(primAttr);

    // create object template with modified primitive asset attributes, by
    // passing handle.  defaults to register object template
    auto newCylObjAttr = objectAttribsMgr->createObject(newHandle);
    CORRADE_VERIFY(newCylObjAttr);
    // create object with new attributes
    auto obj = rigidObjMgr->addObjectByHandle(newHandle);
    CORRADE_VERIFY(obj->isAlive());
    CORRADE_VERIFY(obj->getID() != esp::ID_UNDEFINED);
  }
  // empty vector of handles
  primObjAssetHandles.clear();

}  // SimTest::buildingPrimAssetObjectTemplates

void SimTest::addObjectByHandle() {
  ESP_DEBUG() << "Starting Test : addObject";
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, planeStage, esp::NO_LIGHT_KEY);
  auto rigidObjMgr = simulator->getRigidObjectManager();

  auto obj = rigidObjMgr->addObjectByHandle("invalid_handle");
  CORRADE_COMPARE(obj, nullptr);

  // pass valid object_config.json filepath as handle to addObjectByHandle
  const auto validHandle = Cr::Utility::Path::join(
      TEST_ASSETS, "objects/nested_box.object_config.json");
  obj = rigidObjMgr->addObjectByHandle(validHandle);
  CORRADE_VERIFY(obj->isAlive());
  CORRADE_VERIFY(obj->getID() != esp::ID_UNDEFINED);
}

void SimTest::addSensorToObject() {
  ESP_DEBUG() << "Starting Test : addSensorToObject";
  auto&& data = SimulatorBuilder[testCaseInstanceId()];
  setTestCaseDescription(data.name);
  auto simulator = data.creator(*this, vangogh, esp::NO_LIGHT_KEY);
  auto rigidObjMgr = simulator->getRigidObjectManager();
  // manager of object attributes
  auto objectAttribsMgr = simulator->getObjectAttributesManager();
  auto objs = objectAttribsMgr->getObjectHandlesBySubstring("icosphereSolid");
  auto obj = rigidObjMgr->addObjectByHandle(objs[0]);
  CORRADE_VERIFY(obj->isAlive());
  CORRADE_VERIFY(obj->getID() != esp::ID_UNDEFINED);
  esp::scene::SceneNode& objectNode = *obj->getSceneNode();

  // Add sensor to sphere object
  auto objectSensorSpec = esp::sensor::CameraSensorSpec::create();
  objectSensorSpec->uuid = std::to_string(obj->getID());
  objectSensorSpec->position = {0, 0, 0};
  objectSensorSpec->orientation = {0, 0, 0};
  objectSensorSpec->resolution = {128, 128};
  simulator->addSensorToObject(obj->getID(), objectSensorSpec);
  std::string expectedUUID = std::to_string(obj->getID());
  CameraSensor& cameraSensor = dynamic_cast<CameraSensor&>(
      objectNode.getNodeSensorSuite().get(expectedUUID));
  cameraSensor.setTransformationFromSpec();

  obj->setTranslation(
      {1.0f, 1.5f, 1.0f});  // Move camera to same place as agent

  auto objs2 = objectAttribsMgr->getObjectHandlesBySubstring("nested_box");
  auto obj2 = rigidObjMgr->addObjectByHandle(objs[0]);
  CORRADE_VERIFY(obj2->isAlive());
  CORRADE_VERIFY(obj2->getID() != esp::ID_UNDEFINED);
  obj2->setTranslation({1.0f, 0.5f, -0.5f});
  esp::scene::SceneNode& objectNode2 = *obj2->getSceneNode();

  Observation observation;
  ObservationSpace obsSpace;
  simulator->getRenderer()->bindRenderTarget(cameraSensor);
  CORRADE_VERIFY(cameraSensor.getObservation(*simulator, observation));
  CORRADE_VERIFY(cameraSensor.getObservationSpace(obsSpace));

  esp::vec2i defaultResolution = {128, 128};
  std::vector<size_t> expectedShape{{static_cast<size_t>(defaultResolution[0]),
                                     static_cast<size_t>(defaultResolution[1]),
                                     4}};

  CORRADE_VERIFY(obsSpace.spaceType == ObservationSpaceType::Tensor);
  CORRADE_VERIFY(obsSpace.dataType == esp::core::DataType::DT_UINT8);
  CORRADE_COMPARE(obsSpace.shape, expectedShape);
  CORRADE_COMPARE(observation.buffer->shape, expectedShape);

  // Compare with previously rendered ground truth
  // Object camera at same location as agent camera should render similar image
  CORRADE_COMPARE_WITH(
      (Mn::ImageView2D{Mn::PixelFormat::RGBA8Unorm,
                       {defaultResolution[0], defaultResolution[1]},
                       observation.buffer->data}),
      Cr::Utility::Path::join(screenshotDir, "SimTestExpectedScene.png"),
      (Mn::DebugTools::CompareImageToFile{maxThreshold, 0.75f}));
}

void SimTest::createMagnumRenderingOff() {
  ESP_DEBUG() << "Starting Test : createMagnumRenderingOff";

  // create a simulator
  SimulatorConfiguration simConfig{};
  simConfig.activeSceneName = vangogh;
  simConfig.enablePhysics = true;
  simConfig.physicsConfigFile = physicsConfigFile;
  simConfig.overrideSceneLightDefaults = true;
  simConfig.createRenderer = false;
  simConfig.sceneLightSetupKey = "custom_lighting_1";
  auto simulator = Simulator::create_unique(simConfig);

  // configure objectAttributesManager
  auto objectAttribsMgr = simulator->getObjectAttributesManager();
  auto rigidObjMgr = simulator->getRigidObjectManager();
  objectAttribsMgr->loadAllJSONConfigsFromPath(
      Cr::Utility::Path::join(TEST_ASSETS, "objects/nested_box"), true);

  // check that we can load a glb file
  auto objs = objectAttribsMgr->getObjectHandlesBySubstring("nested_box");
  auto obj = rigidObjMgr->addObjectByHandle(objs[0]);
  CORRADE_VERIFY(obj->isAlive());
  CORRADE_VERIFY(obj->getID() != esp::ID_UNDEFINED);
  obj->setTranslation({-10.0f, -10.0f, -10.0f});

  // check that adding a primitive object works
  obj = rigidObjMgr->addObjectByHandle("cubeSolid");
  obj->setTranslation({10.0f, 10.0f, 10.0f});
  CORRADE_VERIFY(obj->isAlive());
  CORRADE_VERIFY(obj->getID() != esp::ID_UNDEFINED);
  esp::scene::SceneNode* objectNode = obj->getSceneNode();

  auto distanceBetween = [](Mn::Vector3 a, Mn::Vector3 b) {
    Mn::Vector3 d = b - a;
    return Mn::Math::pow(dot(d, d), 0.5f);
  };

  auto testRaycast = [&]() {
    // cast a ray at the object to check that the object is actually there
    auto raycastresults = simulator->castRay(
        esp::geo::Ray({10.0, 9.0, 10.0}, {0.0, 1.0, 0.0}), 100.0, 0);
    CORRADE_COMPARE(raycastresults.hits[0].objectId, obj->getID());
    auto point = raycastresults.hits[0].point;
    CORRADE_COMPARE_AS(distanceBetween(point, {10.0, 9.9, 10.0}), 0.001,
                       Cr::TestSuite::Compare::Less);
    raycastresults = simulator->castRay(
        esp::geo::Ray({10.0, 11.0, 10.0}, {0.0, -1.0, 0.0}), 100.0, 0);
    CORRADE_COMPARE(raycastresults.hits[0].objectId, obj->getID());
    point = raycastresults.hits[0].point;
    CORRADE_COMPARE_AS(distanceBetween(point, {10.0, 10.1, 10.0}), 0.001,
                       Cr::TestSuite::Compare::Less);
  };

  auto testBoundingBox = [&]() {
    // check that we can still compute bounding box of the object
    Magnum::Range3D meshbb = objectNode->getCumulativeBB();
    float eps = 0.001;
    CORRADE_COMPARE_WITH(meshbb.left(), -0.1,
                         Cr::TestSuite::Compare::around(eps));
    CORRADE_COMPARE_WITH(meshbb.right(), 0.1,
                         Cr::TestSuite::Compare::around(eps));
    CORRADE_COMPARE_WITH(meshbb.bottom(), -0.1,
                         Cr::TestSuite::Compare::around(eps));
    CORRADE_COMPARE_WITH(meshbb.top(), 0.1,
                         Cr::TestSuite::Compare::around(eps));
    CORRADE_COMPARE_WITH(meshbb.back(), -0.1,
                         Cr::TestSuite::Compare::around(eps));
    CORRADE_COMPARE_WITH(meshbb.front(), 0.1,
                         Cr::TestSuite::Compare::around(eps));
  };
  // test raycast and bounding box for cubeSolid
  testRaycast();
  testBoundingBox();

  // test raycast and bounding box for cubeWireframe
  rigidObjMgr->removePhysObjectByHandle(obj->getHandle());
  obj = rigidObjMgr->addObjectByHandle("cubeWireframe");
  CORRADE_VERIFY(obj->isAlive());
  CORRADE_VERIFY(obj->getID() != esp::ID_UNDEFINED);
  obj->setTranslation({10.0f, 10.0f, 10.0f});
  objectNode = obj->getSceneNode();
  testRaycast();
  testBoundingBox();

  // do some sensor stuff to check that nothing errors
  auto objectSensorSpec = esp::sensor::CameraSensorSpec::create();
  objectSensorSpec->uuid = std::to_string(obj->getID());
  objectSensorSpec->position = {0, 0, 0};
  objectSensorSpec->orientation = {0, 0, 0};
  objectSensorSpec->resolution = {128, 128};
  simulator->addSensorToObject(obj->getID(), objectSensorSpec);
  std::string expectedUUID = std::to_string(obj->getID());
  CameraSensor& cameraSensor = dynamic_cast<CameraSensor&>(
      objectNode->getNodeSensorSuite().get(expectedUUID));
  cameraSensor.setTransformationFromSpec();
  Observation observation;

  // check that there is no renderer
  CORRADE_VERIFY(!simulator->getRenderer());
  CORRADE_VERIFY(!cameraSensor.getObservation(*simulator, observation));
}

}  // namespace

CORRADE_TEST_MAIN(SimTest)
