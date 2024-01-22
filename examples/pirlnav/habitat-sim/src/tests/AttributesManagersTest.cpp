// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <Corrade/TestSuite/Compare/Numeric.h>
#include <Corrade/TestSuite/Tester.h>
#include <string>

#include "esp/metadata/MetadataMediator.h"
#include "esp/metadata/managers/AssetAttributesManager.h"
#include "esp/metadata/managers/AttributesManagerBase.h"
#include "esp/metadata/managers/ObjectAttributesManager.h"
#include "esp/metadata/managers/PhysicsAttributesManager.h"
#include "esp/metadata/managers/StageAttributesManager.h"

#include "configure.h"

namespace Cr = Corrade;

namespace AttrMgrs = esp::metadata::managers;
namespace Attrs = esp::metadata::attributes;

using esp::metadata::MetadataMediator;
using esp::metadata::PrimObjTypes;

using esp::physics::MotionType;

using AttrMgrs::AttributesManager;
using Attrs::AbstractPrimitiveAttributes;
using Attrs::CapsulePrimitiveAttributes;
using Attrs::ConePrimitiveAttributes;
using Attrs::CubePrimitiveAttributes;
using Attrs::CylinderPrimitiveAttributes;
using Attrs::IcospherePrimitiveAttributes;
using Attrs::ObjectAttributes;
using Attrs::PhysicsManagerAttributes;
using Attrs::SceneInstanceAttributes;
using Attrs::StageAttributes;
using Attrs::UVSpherePrimitiveAttributes;

namespace {
const std::string physicsConfigFile =
    Cr::Utility::Path::join(DATA_DIR,
                            "test_assets/testing.physics_config.json");

/**
 * @brief Test attributesManagers' functionality via loading, creating, copying
 * and deleting Attributes.
 */
struct AttributesManagersTest : Cr::TestSuite::Tester {
  explicit AttributesManagersTest();

  /**
   * @brief Test creation, copying and removal of templates for Object,
   * Physics and Stage Attributes Managers
   * @tparam Class of attributes manager
   * @param mgr the Attributes Manager being tested,
   * @param handle the handle of the desired attributes template to work with
   */
  template <typename T>
  void testCreateAndRemove(std::shared_ptr<T> mgr, const std::string& handle);

  /**
   * @brief Test creation, copying and removal of templates for lights
   * attributes managers.
   * @param mgr the Attributes Manager being tested,
   * @param handle the handle of the desired attributes template to work with
   */
  void testCreateAndRemoveLights(
      AttrMgrs::LightLayoutAttributesManager::ptr mgr,
      const std::string& handle);

  /**
   * @brief Test creation many templates and removing all but defaults.
   * @tparam Class of attributes manager
   * @param mgr the Attributes Manager being tested,
   * @param renderHandle a legal render handle to set for the new template so
   * that registration won't fail.
   */
  template <typename T>
  void testRemoveAllButDefault(std::shared_ptr<T> mgr,
                               const std::string& handle,
                               bool setRenderHandle);

  /**
   * @brief Test creation, copying and removal of new default/empty templates
   * for Object, Physics and Stage Attributes Managers
   * @tparam Class of attributes manager
   * @param mgr the Attributes Manager being tested,
   * @param renderHandle a legal render handle to set for the new template so
   * that registration won't fail.
   */

  // specialization so we can ignore this for physics attributes
  void processTemplateRenderAsset(
      std::shared_ptr<AttrMgrs::PhysicsAttributesManager> mgr,
      std::shared_ptr<Attrs::PhysicsManagerAttributes> newAttrTemplate0,
      const std::string& handle) {}

  template <typename T, typename U>
  void processTemplateRenderAsset(std::shared_ptr<T> mgr,
                                  std::shared_ptr<U> newAttrTemplate0,
                                  const std::string& handle) {
    auto attrTemplate1 = mgr->createObject(handle, false);
    // set legitimate render handle in template
    newAttrTemplate0->setRenderAssetHandle(
        attrTemplate1->template get<std::string>("render_asset"));
  }

  template <typename T>
  void testCreateAndRemoveDefault(std::shared_ptr<T> mgr,
                                  const std::string& handle,
                                  bool setRenderHandle);
  /**
   * @brief This method will test the user-defined configuration values to see
   * that they match with expected passed values.  The config is expected to
   * hold one of each type that it supports.
   * @param userConfig The configuration object whose contents are to be
   * tested
   * @param str_val Expected string value
   * @param bool_val Expected boolean value
   * @param double_val Exptected double value
   * @param vec_val Expected Magnum::Vector3 value
   * @param quat_val Expected Quaternion value - note that the JSON is read
   * with scalar at idx 0, whereas the quaternion constructor takes the vector
   * component in the first position and the scalar in the second.
   */
  void testUserDefinedConfigVals(
      std::shared_ptr<esp::core::config::Configuration> userConfig,
      const std::string& str_val,
      bool bool_val,
      int int_val,
      double double_val,
      Magnum::Vector3 vec_val,
      Magnum::Quaternion quat_val);

  /**
   * @brief Test creation, copying and removal of templates for primitive
   * assets.
   * @tparam Class of attributes being managed
   * @param defaultAttribs the default template of the passed type T
   * @param ctorModField the name of the modified field of type @ref U that
   * impacts the constructor.
   * @param legalVal a legal value of ctorModField; This should be different
   * than template default for @ref ctorModField.
   * @param illegalVal a legal value of ctorModField.  If null ptr then no
   * illegal values possible.
   */
  template <typename T>
  void testAssetAttributesModRegRemove(std::shared_ptr<T> defaultAttribs,
                                       int legalVal,
                                       int const* illegalVal);

  void testAssetAttributesTemplateCreateFromHandle(
      const std::string& newTemplateName);

  /**
   * @brief This test will test creating, modifying, registering and deleting
   * Attributes via the AttributesManager for PhysicsManagerAttributes. These
   * tests should be consistent with most types of future attributes managers
   * specializing the AttributesManager class template that follow the same
   * expected behavior paths as extent attributes/attributesManagers.  Note :
   * PrimitiveAssetAttributes exhibit slightly different behavior and need their
   * own tests.
   */
  void testPhysicsAttributesManagersCreate();

  /**
   * @brief This test will test creating, modifying, registering and deleting
   * Attributes via the AttributesManager for StageAttributes.  These
   * tests should be consistent with most types of future attributes managers
   * specializing the AttributesManager class template that follow the same
   * expected behavior paths as extent attributes/attributesManagers.  Note :
   * PrimitiveAssetAttributes exhibit slightly different behavior and need their
   * own tests.
   */
  void testStageAttributesManagersCreate();

  /**
   * @brief This test will test creating, modifying, registering and deleting
   * Attributes via the AttributesManager for ObjectAttributes.  These
   * tests should be consistent with most types of future attributes managers
   * specializing the AttributesManager class template that follow the same
   * expected behavior paths as extent attributes/attributesManagers.  Note :
   * PrimitiveAssetAttributes exhibit slightly different behavior and need their
   * own tests.
   */
  void testObjectAttributesManagersCreate();

  /**
   * @brief This test will test creating, modifying, registering and deleting
   * Attributes via the AttributesManager for LightLayoutsAttributes
   */
  void testLightLayoutAttributesManager();

  /**
   * @brief test primitive asset attributes functionality in attirbutes
   * managers. This includes testing handle auto-gen when relevant fields in
   * asset attributes are changed.
   */
  void testPrimitiveAssetAttributes();

  // test member vars

  esp::logging::LoggingContext loggingContext_;
  AttrMgrs::AssetAttributesManager::ptr assetAttributesManager_ = nullptr;
  AttrMgrs::LightLayoutAttributesManager::ptr lightLayoutAttributesManager_ =
      nullptr;
  AttrMgrs::ObjectAttributesManager::ptr objectAttributesManager_ = nullptr;
  AttrMgrs::PhysicsAttributesManager::ptr physicsAttributesManager_ = nullptr;
  AttrMgrs::SceneInstanceAttributesManager::ptr
      sceneInstanceAttributesManager_ = nullptr;
  AttrMgrs::StageAttributesManager::ptr stageAttributesManager_ = nullptr;

};  // struct AttributesManagersTest

AttributesManagersTest::AttributesManagersTest() {
  // set up a default simulation config to initialize MM
  auto cfg = esp::sim::SimulatorConfiguration{};
  auto MM = MetadataMediator::create(cfg);
  // get attributes managers for default dataset
  assetAttributesManager_ = MM->getAssetAttributesManager();
  lightLayoutAttributesManager_ = MM->getLightLayoutAttributesManager();
  objectAttributesManager_ = MM->getObjectAttributesManager();
  physicsAttributesManager_ = MM->getPhysicsAttributesManager();
  sceneInstanceAttributesManager_ = MM->getSceneInstanceAttributesManager();
  stageAttributesManager_ = MM->getStageAttributesManager();

  addTests({
      &AttributesManagersTest::testPhysicsAttributesManagersCreate,
      &AttributesManagersTest::testStageAttributesManagersCreate,
      &AttributesManagersTest::testObjectAttributesManagersCreate,
      &AttributesManagersTest::testLightLayoutAttributesManager,
      &AttributesManagersTest::testPrimitiveAssetAttributes,
  });
}

/**
 * @brief Test creation, copying and removal of templates for Object, Physics
 * and Stage Attributes Managers
 * @tparam Class of attributes manager
 * @param mgr the Attributes Manager being tested,
 * @param handle the handle of the desired attributes template to work with
 */
template <typename T>
void AttributesManagersTest::testCreateAndRemove(std::shared_ptr<T> mgr,
                                                 const std::string& handle) {
  // get starting number of templates
  int orignNumTemplates = mgr->getNumObjects();
  // verify template is not present - should not be
  bool isPresentAlready = mgr->getObjectLibHasHandle(handle);
  CORRADE_VERIFY(!isPresentAlready);

  // create template from source handle, register it and retrieve it
  // Note: registration of template means this is a copy of registered
  // template
  auto attrTemplate1 = mgr->createObject(handle, true);
  // verify it exists
  CORRADE_VERIFY(attrTemplate1);
  // verify ID exists
  bool idIsPresent = mgr->getObjectLibHasID(attrTemplate1->getID());
  CORRADE_VERIFY(idIsPresent);
  // retrieve a copy of the named attributes template
  auto attrTemplate2 = mgr->getObjectOrCopyByHandle(handle);
  // verify copy has same quantities and values as original
  CORRADE_COMPARE(attrTemplate1->getHandle(), attrTemplate2->getHandle());

  // test changing a user-defined field in each template, verify the templates
  // are not now the same
  attrTemplate1->setFileDirectory("temp_dir_1");
  attrTemplate2->setFileDirectory("temp_dir_2");
  CORRADE_COMPARE_AS(attrTemplate1->getFileDirectory(),
                     attrTemplate2->getFileDirectory(),
                     Cr::TestSuite::Compare::NotEqual);
  // get original template ID
  int oldID = attrTemplate1->getID();

  // register modified template and verify that this is the template now
  // stored
  int newID = mgr->registerObject(attrTemplate2, handle);
  // verify IDs are the same
  CORRADE_COMPARE(oldID, newID);

  // get another copy
  auto attrTemplate3 = mgr->getObjectOrCopyByHandle(handle);
  // verify added field is present and the same
  CORRADE_COMPARE(attrTemplate3->getFileDirectory(),
                  attrTemplate2->getFileDirectory());
  // change field in new copy
  attrTemplate3->setFileDirectory("temp_dir_3");
  // verify that now they are different
  CORRADE_COMPARE_AS(attrTemplate3->getFileDirectory(),
                     attrTemplate2->getFileDirectory(),
                     Cr::TestSuite::Compare::NotEqual);

  // test removal
  int removeID = attrTemplate2->getID();
  // remove template by ID, acquire copy of removed template
  auto oldTemplate = mgr->removeObjectByID(removeID);
  // verify it exists
  CORRADE_VERIFY(oldTemplate);
  // verify there are same number of templates as when we started
  CORRADE_COMPARE(orignNumTemplates, mgr->getNumObjects());
  // re-add template copy via registration
  int newAddID = mgr->registerObject(attrTemplate2, handle);
  // verify IDs are the same
  CORRADE_COMPARE(removeID, newAddID);

  // lock template referenced by handle
  bool success = mgr->setLock(handle, true);
  // attempt to remove attributes via handle
  auto oldTemplate2 = mgr->removeObjectByHandle(handle);
  // verify no template was deleted
  CORRADE_VERIFY(!oldTemplate2);
  // unlock template
  success = mgr->setLock(handle, false);

  // remove  attributes via handle
  auto oldTemplate3 = mgr->removeObjectByHandle(handle);
  // verify deleted template exists
  CORRADE_VERIFY(oldTemplate3);
  // verify ID does not exist in library now
  idIsPresent = mgr->getObjectLibHasID(oldTemplate3->getID());
  CORRADE_VERIFY(!idIsPresent);
  // verify there are same number of templates as when we started
  CORRADE_COMPARE(orignNumTemplates, mgr->getNumObjects());

}  // AttributesManagersTest::testCreateAndRemove

void AttributesManagersTest::testCreateAndRemoveLights(
    AttrMgrs::LightLayoutAttributesManager::ptr mgr,
    const std::string& handle) {
  // get starting number of templates
  int origNumTemplates = mgr->getNumObjects();

  // Source config for lights holds multiple light configurations.
  // Create a single template for each defined light in configuration and
  // register it.
  mgr->createObject(handle, true);
  // get number of templates loaded
  int numLoadedLights = mgr->getNumObjects();

  // verify lights were added
  CORRADE_COMPARE_AS(numLoadedLights, origNumTemplates,
                     Cr::TestSuite::Compare::NotEqual);

  // get handles of all lights added
  auto lightHandles = mgr->getObjectHandlesBySubstring();
  CORRADE_COMPARE(lightHandles.size(), numLoadedLights);

  // remove all added handles
  for (auto handle : lightHandles) {
    mgr->removeObjectByHandle(handle);
  }
  // verify there are same number of templates as when we started
  CORRADE_COMPARE(mgr->getNumObjects(), origNumTemplates);

}  // AttributesManagersTest::testCreateAndRemove

/**
 * @brief Test creation many templates and removing all but defaults.
 * @tparam Class of attributes manager
 * @param mgr the Attributes Manager being tested,
 * @param renderHandle a legal render handle to set for the new template so
 * that registration won't fail.
 */
template <typename T>
void AttributesManagersTest::testRemoveAllButDefault(std::shared_ptr<T> mgr,
                                                     const std::string& handle,
                                                     bool setRenderHandle) {
  // get starting number of templates
  int orignNumTemplates = mgr->getNumObjects();
  // lock all current handles
  std::vector<std::string> origHandles =
      mgr->setLockBySubstring(true, "", true);
  // make sure we have locked all original handles
  CORRADE_COMPARE(orignNumTemplates, origHandles.size());

  // create multiple new templates, and then test deleting all those created
  // using single command.
  int numToAdd = 10;
  for (int i = 0; i < numToAdd; ++i) {
    // assign template a handle
    std::string newHandleIter("newTemplateHandle_" + std::to_string(i));
    CORRADE_ITERATION(newHandleIter);
    // create a template with a legal handle
    auto attrTemplate1 = mgr->createObject(handle, false);
    // register template with new handle
    int tmpltID = mgr->registerObject(attrTemplate1, newHandleIter);
    // verify template added
    CORRADE_VERIFY(tmpltID != esp::ID_UNDEFINED);
    auto attrTemplate2 = mgr->getObjectOrCopyByHandle(newHandleIter);
    // verify added template  exists
    CORRADE_VERIFY(attrTemplate2);
  }

  // now delete all templates that
  auto removedNamedTemplates =
      mgr->removeObjectsBySubstring("newTemplateHandle_", true);
  // verify that the number removed == the number added
  CORRADE_COMPARE(removedNamedTemplates.size(), numToAdd);

  // re-add templates
  for (auto& tmplt : removedNamedTemplates) {
    // register template with new handle
    int tmpltID = mgr->registerObject(tmplt);
    // verify template added
    CORRADE_VERIFY(tmpltID != esp::ID_UNDEFINED);
    auto attrTemplate2 = mgr->getObjectOrCopyByHandle(tmplt->getHandle());
    // verify added template  exists
    CORRADE_VERIFY(attrTemplate2);
  }

  // now delete all templates that have just been added
  auto removedTemplates = mgr->removeAllObjects();
  // verify that the number removed == the number added
  CORRADE_COMPARE(removedTemplates.size(), numToAdd);
  // verify there are same number of templates as when we started
  CORRADE_COMPARE(orignNumTemplates, mgr->getNumObjects());

  // unlock all original handles
  std::vector<std::string> newOrigHandles =
      mgr->setLockByHandles(origHandles, false);
  // verify orig handles are those that have been unlocked
  CORRADE_COMPARE(newOrigHandles, origHandles);
  // make sure we have unlocked all original handles
  CORRADE_COMPARE(orignNumTemplates, newOrigHandles.size());

}  // AttributesManagersTest::testRemoveAllButDefault

template <typename T>
void AttributesManagersTest::testCreateAndRemoveDefault(
    std::shared_ptr<T> mgr,
    const std::string& handle,
    bool setRenderHandle) {
  // get starting number of templates
  int orignNumTemplates = mgr->getNumObjects();
  // assign template a handle
  std::string newHandle = "newTemplateHandle";

  // create new template but do not register it
  auto newAttrTemplate0 = mgr->createDefaultObject(newHandle, false);
  // verify real template was returned
  CORRADE_VERIFY(newAttrTemplate0);

  // create template from source handle, register it and retrieve it
  // Note: registration of template means this is a copy of registered
  // template
  processTemplateRenderAsset(mgr, newAttrTemplate0, handle);
  // register modified template and verify that this is the template now
  // stored
  int newID = mgr->registerObject(newAttrTemplate0, newHandle);

  // get a copy of added template
  auto attrTemplate3 = mgr->getObjectOrCopyByHandle(newHandle);

  // remove new template by name
  auto newAttrTemplate1 = mgr->removeObjectByHandle(newHandle);

  // verify it exists
  CORRADE_VERIFY(newAttrTemplate1);
  // verify there are same number of templates as when we started
  CORRADE_COMPARE(orignNumTemplates, mgr->getNumObjects());

}  // AttributesManagersTest::testCreateAndRemoveDefault

template <typename T>
void AttributesManagersTest::testAssetAttributesModRegRemove(
    std::shared_ptr<T> defaultAttribs,
    int legalVal,
    int const* illegalVal) {
  // get starting number of templates
  int orignNumTemplates = assetAttributesManager_->getNumObjects();

  // get name of default template
  std::string oldHandle = defaultAttribs->getHandle();

  // verify default template is valid
  bool isTemplateValid = defaultAttribs->isValidTemplate();
  CORRADE_VERIFY(isTemplateValid);

  // if illegal values are possible
  if (illegalVal != nullptr) {
    // modify template value used by primitive constructor (will change
    // name) illegal modification
    defaultAttribs->setNumSegments(*illegalVal);
    // verify template is not valid
    bool isTemplateValid = defaultAttribs->isValidTemplate();
    CORRADE_VERIFY(!isTemplateValid);
  }
  // legal modification, different than default
  defaultAttribs->setNumSegments(legalVal);
  // verify template is valid
  isTemplateValid = defaultAttribs->isValidTemplate();
  CORRADE_VERIFY(isTemplateValid);
  // rebuild handle to reflect new parameters
  defaultAttribs->buildHandle();

  // get synthesized handle
  std::string newHandle = defaultAttribs->getHandle();
  // register modified template
  assetAttributesManager_->registerObject(defaultAttribs);

  // verify new handle is in template library
  CORRADE_INTERNAL_ASSERT(
      assetAttributesManager_->getObjectLibHasHandle(newHandle));
  // verify old template is still present as well
  CORRADE_INTERNAL_ASSERT(
      assetAttributesManager_->getObjectLibHasHandle(oldHandle));

  // get new template
  std::shared_ptr<T> newAttribs =
      assetAttributesManager_->getObjectOrCopyByHandle<T>(newHandle);
  // verify template has modified values
  int newValue = newAttribs->getNumSegments();
  CORRADE_COMPARE(legalVal, newValue);
  // remove modified template via handle
  auto oldTemplate2 = assetAttributesManager_->removeObjectByHandle(newHandle);
  // verify deleted template  exists
  CORRADE_VERIFY(oldTemplate2);

  // verify there are same number of templates as when we started
  CORRADE_COMPARE(orignNumTemplates, assetAttributesManager_->getNumObjects());

}  // AttributesManagersTest::testAssetAttributesModRegRemove

void AttributesManagersTest::testAssetAttributesTemplateCreateFromHandle(
    const std::string& newTemplateName) {
  // get starting number of templates
  int orignNumTemplates = assetAttributesManager_->getNumObjects();
  // first verify that no template with given name exists
  bool templateExists =
      assetAttributesManager_->getObjectLibHasHandle(newTemplateName);
  CORRADE_VERIFY(!templateExists);
  // create new template based on handle and verify that it is created
  auto newTemplate =
      assetAttributesManager_->createTemplateFromHandle(newTemplateName, true);
  CORRADE_VERIFY(newTemplate);

  // now verify that template is in library
  templateExists =
      assetAttributesManager_->getObjectLibHasHandle(newTemplateName);
  CORRADE_VERIFY(templateExists);

  // remove new template via handle
  auto oldTemplate =
      assetAttributesManager_->removeObjectByHandle(newTemplateName);
  // verify deleted template  exists
  CORRADE_VERIFY(oldTemplate);

  // verify there are same number of templates as when we started
  CORRADE_COMPARE(orignNumTemplates, assetAttributesManager_->getNumObjects());

}  // AttributesManagersTest::testAssetAttributesTemplateCreateFromHandle

void AttributesManagersTest::testPhysicsAttributesManagersCreate() {
  CORRADE_INFO(
      "Start Test : Create, Edit, Remove Attributes for "
      "PhysicsAttributesManager @"
      << physicsConfigFile);

  // physics attributes manager attributes verifcation
  testCreateAndRemove<AttrMgrs::PhysicsAttributesManager>(
      physicsAttributesManager_, physicsConfigFile);
  testCreateAndRemoveDefault<AttrMgrs::PhysicsAttributesManager>(
      physicsAttributesManager_, physicsConfigFile, false);
}  // AttributesManagersTest::PhysicsAttributesManagersCreate

void AttributesManagersTest::testStageAttributesManagersCreate() {
  std::string stageConfigFile =
      Cr::Utility::Path::join(DATA_DIR, "test_assets/scenes/simple_room.glb");

  CORRADE_INFO(
      "Start Test : Create, Edit, Remove Attributes for "
      "StageAttributesManager @"
      << stageConfigFile);

  // scene attributes manager attributes verifcation
  testCreateAndRemove<AttrMgrs::StageAttributesManager>(stageAttributesManager_,
                                                        stageConfigFile);
  testCreateAndRemoveDefault<AttrMgrs::StageAttributesManager>(
      stageAttributesManager_, stageConfigFile, true);

}  // AttributesManagersTest::StageAttributesManagersCreate

void AttributesManagersTest::testObjectAttributesManagersCreate() {
  std::string objectConfigFile = Cr::Utility::Path::join(
      DATA_DIR, "test_assets/objects/chair.object_config.json");

  CORRADE_INFO(
      "Start Test : Create, Edit, Remove Attributes for "
      "ObjectAttributesManager @"
      << objectConfigFile);

  int origNumFileBased = objectAttributesManager_->getNumFileTemplateObjects();
  int origNumPrimBased = objectAttributesManager_->getNumSynthTemplateObjects();

  // object attributes manager attributes verifcation
  testCreateAndRemove<AttrMgrs::ObjectAttributesManager>(
      objectAttributesManager_, objectConfigFile);
  // verify that no new file-based and no new synth based template objects
  // remain
  int newNumFileBased1 = objectAttributesManager_->getNumFileTemplateObjects();
  int newNumPrimBased1 = objectAttributesManager_->getNumSynthTemplateObjects();
  CORRADE_COMPARE(origNumFileBased, newNumFileBased1);
  CORRADE_COMPARE(origNumPrimBased, newNumPrimBased1);
  testCreateAndRemoveDefault<AttrMgrs::ObjectAttributesManager>(
      objectAttributesManager_, objectConfigFile, true);
  // verify that no new file-based and no new synth based template objects
  // remain
  int newNumFileBased2 = objectAttributesManager_->getNumFileTemplateObjects();
  int newNumPrimBased2 = objectAttributesManager_->getNumSynthTemplateObjects();
  CORRADE_COMPARE(origNumFileBased, newNumFileBased2);
  CORRADE_COMPARE(origNumPrimBased, newNumPrimBased2);

  // test adding many and removing all but defaults
  testRemoveAllButDefault<AttrMgrs::ObjectAttributesManager>(
      objectAttributesManager_, objectConfigFile, true);
  // verify that no new file-based and no new synth based template objects
  // remain
  int newNumFileBased3 = objectAttributesManager_->getNumFileTemplateObjects();
  int newNumPrimBased3 = objectAttributesManager_->getNumSynthTemplateObjects();
  CORRADE_COMPARE(origNumFileBased, newNumFileBased3);
  CORRADE_COMPARE(origNumPrimBased, newNumPrimBased3);
}  // AttributesManagersTest::ObjectAttributesManagersCreate test

void AttributesManagersTest::testLightLayoutAttributesManager() {
  std::string lightConfigFile = Cr::Utility::Path::join(
      DATA_DIR, "test_assets/lights/test_lights.lighting_config.json");

  CORRADE_INFO(
      "Start Test : Create, Edit, Remove Attributes for "
      "LightLayoutAttributesManager @"
      << lightConfigFile);
  // light attributes manager attributes verifcation
  testCreateAndRemoveLights(lightLayoutAttributesManager_, lightConfigFile);

}  // AttributesManagersTest::LightLayoutAttributesManagerTest

void AttributesManagersTest::testPrimitiveAssetAttributes() {
  /**
   * Primitive asset attributes require slightly different testing since a
   * default set of attributes (matching the default Magnum::Primitive
   * parameters) are created on program load and are always present.  User
   * modification of asset attributes always starts by modifying an existing
   * default template - users will never create an attributes template from
   * scratch.
   */
  int legalModValWF = 64;
  int illegalModValWF = 25;
  int legalModValSolid = 5;
  int illegalModValSolid = 0;

  const std::string capsule3DSolidHandle =
      "capsule3DSolid_hemiRings_5_cylRings_2_segments_16_halfLen_1.75_"
      "useTexCoords_true_useTangents_true";
  const std::string capsule3DWireframeHandle =
      "capsule3DWireframe_hemiRings_8_cylRings_2_segments_20_halfLen_1.5";

  const std::string coneSolidHandle =
      "coneSolid_segments_12_halfLen_1.35_rings_1_useTexCoords_true_"
      "useTangents_true_capEnd_true";
  const std::string coneWireframeHandle =
      "coneWireframe_segments_32_halfLen_1.44";

  const std::string cylinderSolidHandle =
      "cylinderSolid_rings_1_segments_28_halfLen_1.11_useTexCoords_true_"
      "useTangents_true_capEnds_true";
  const std::string cylinderWireframeHandle =
      "cylinderWireframe_rings_1_segments_32_halfLen_1.23";

  const std::string uvSphereSolidHandle =
      "uvSphereSolid_rings_16_segments_8_useTexCoords_true_useTangents_true";
  const std::string uvSphereWireframeHandle =
      "uvSphereWireframe_rings_20_segments_24";

  //////////////////////////
  // get default template for solid capsule
  {
    CORRADE_INFO("Starting CapsulePrimitiveAttributes");
    CapsulePrimitiveAttributes::ptr dfltCapsAttribs =
        assetAttributesManager_->getDefaultCapsuleTemplate(false);
    // verify it exists
    CORRADE_VERIFY(dfltCapsAttribs);

    // for solid primitives, and value > 2 for segments is legal
    testAssetAttributesModRegRemove<CapsulePrimitiveAttributes>(
        dfltCapsAttribs, legalModValSolid, &illegalModValSolid);

    // test that a new template can be created from the specified handles
    testAssetAttributesTemplateCreateFromHandle(capsule3DSolidHandle);

    // test wireframe version
    dfltCapsAttribs = assetAttributesManager_->getDefaultCapsuleTemplate(true);
    // verify it exists
    CORRADE_VERIFY(dfltCapsAttribs);
    // segments must be mult of 4 for wireframe primtives
    testAssetAttributesModRegRemove<CapsulePrimitiveAttributes>(
        dfltCapsAttribs, legalModValWF, &illegalModValWF);
    // test that a new template can be created from the specified handles
    testAssetAttributesTemplateCreateFromHandle(capsule3DWireframeHandle);
  }
  //////////////////////////
  // get default template for solid cone
  {
    CORRADE_INFO("Starting ConePrimitiveAttributes");

    ConePrimitiveAttributes::ptr dfltConeAttribs =
        assetAttributesManager_->getDefaultConeTemplate(false);
    // verify it exists
    CORRADE_VERIFY(dfltConeAttribs);

    // for solid primitives, and value > 2 for segments is legal
    testAssetAttributesModRegRemove<ConePrimitiveAttributes>(
        dfltConeAttribs, legalModValSolid, &illegalModValSolid);

    // test that a new template can be created from the specified handles
    testAssetAttributesTemplateCreateFromHandle(coneSolidHandle);

    // test wireframe version
    dfltConeAttribs = assetAttributesManager_->getDefaultConeTemplate(true);
    // verify it exists
    CORRADE_VERIFY(dfltConeAttribs);
    // segments must be mult of 4 for wireframe primtives
    testAssetAttributesModRegRemove<ConePrimitiveAttributes>(
        dfltConeAttribs, legalModValWF, &illegalModValWF);

    // test that a new template can be created from the specified handles
    testAssetAttributesTemplateCreateFromHandle(coneWireframeHandle);
  }
  //////////////////////////
  // get default template for solid cylinder
  {
    CORRADE_INFO("Starting CylinderPrimitiveAttributes");

    CylinderPrimitiveAttributes::ptr dfltCylAttribs =
        assetAttributesManager_->getDefaultCylinderTemplate(false);
    // verify it exists
    CORRADE_VERIFY(dfltCylAttribs);

    // for solid primitives, and value > 2 for segments is legal
    testAssetAttributesModRegRemove<CylinderPrimitiveAttributes>(
        dfltCylAttribs, 5, &illegalModValSolid);

    // test that a new template can be created from the specified handles
    testAssetAttributesTemplateCreateFromHandle(cylinderSolidHandle);

    // test wireframe version
    dfltCylAttribs = assetAttributesManager_->getDefaultCylinderTemplate(true);
    // verify it exists
    CORRADE_VERIFY(dfltCylAttribs);
    // segments must be mult of 4 for wireframe primtives
    testAssetAttributesModRegRemove<CylinderPrimitiveAttributes>(
        dfltCylAttribs, legalModValWF, &illegalModValWF);
    // test that a new template can be created from the specified handles
    testAssetAttributesTemplateCreateFromHandle(cylinderWireframeHandle);
  }
  //////////////////////////
  // get default template for solid UV Sphere
  {
    CORRADE_INFO("Starting UVSpherePrimitiveAttributes");

    UVSpherePrimitiveAttributes::ptr dfltUVSphereAttribs =
        assetAttributesManager_->getDefaultUVSphereTemplate(false);
    // verify it exists
    CORRADE_VERIFY(dfltUVSphereAttribs);

    // for solid primitives, and value > 2 for segments is legal
    testAssetAttributesModRegRemove<UVSpherePrimitiveAttributes>(
        dfltUVSphereAttribs, 5, &illegalModValSolid);

    // test that a new template can be created from the specified handles
    testAssetAttributesTemplateCreateFromHandle(uvSphereSolidHandle);

    // test wireframe version
    dfltUVSphereAttribs =
        assetAttributesManager_->getDefaultUVSphereTemplate(true);
    // verify it exists
    CORRADE_VERIFY(dfltUVSphereAttribs);
    // segments must be mult of 4 for wireframe primtives
    testAssetAttributesModRegRemove<UVSpherePrimitiveAttributes>(
        dfltUVSphereAttribs, legalModValWF, &illegalModValWF);

    // test that a new template can be created from the specified handles
    testAssetAttributesTemplateCreateFromHandle(uvSphereWireframeHandle);
  }
}  // AttributesManagersTest::AsssetAttributesManagerGetAndModify test

}  // namespace

CORRADE_TEST_MAIN(AttributesManagersTest)
