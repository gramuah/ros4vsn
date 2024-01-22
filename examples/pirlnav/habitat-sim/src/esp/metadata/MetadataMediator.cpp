// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "MetadataMediator.h"

namespace esp {
namespace metadata {

MetadataMediator::MetadataMediator(const sim::SimulatorConfiguration& cfg) {
  buildAttributesManagers();
  // sets simConfig_, activeSceneDataset_ and currPhysicsManagerAttributes_
  // based on config
  setSimulatorConfiguration(cfg);
}  // MetadataMediator ctor (SimulatorConfiguration)

void MetadataMediator::buildAttributesManagers() {
  physicsAttributesManager_ = managers::PhysicsAttributesManager::create();

  sceneDatasetAttributesManager_ =
      managers::SceneDatasetAttributesManager::create(
          physicsAttributesManager_);

  // should always have default dataset, but this is managed by MM instead of
  // made undeletable in SceneDatasetManager, so that it can be easily "reset"
  // by deleting and remaking.
  createSceneDataset("default");
  // should always have default physicsManagerAttributesPath
  createPhysicsManagerAttributes(ESP_DEFAULT_PHYSICS_CONFIG_REL_PATH);
  // after this setSimulatorConfiguration will be called
}  // MetadataMediator::buildAttributesManagers

bool MetadataMediator::setSimulatorConfiguration(
    const sim::SimulatorConfiguration& cfg) {
  simConfig_ = cfg;

  // set current active dataset name - if unchanged, does nothing
  bool success = setActiveSceneDatasetName(simConfig_.sceneDatasetConfigFile);
  if (!success) {
    // something failed about setting up active scene dataset
    ESP_ERROR()
        << "Some error prevented current scene dataset name to be changed to"
        << simConfig_.sceneDatasetConfigFile;
    return false;
  }

  // set active physics attributes handle - if unchanged, does nothing
  success = setCurrPhysicsAttributesHandle(simConfig_.physicsConfigFile);
  if (!success) {
    // something failed about setting up physics attributes
    ESP_ERROR() << "Some error prevented current physics attributes to"
                << simConfig_.physicsConfigFile;
    return false;
  }

  // get a ref to current dataset attributes
  attributes::SceneDatasetAttributes::ptr datasetAttr = getActiveDSAttribs();
  // pass relevant config values to the current dataset
  if (datasetAttr != nullptr) {
    datasetAttr->setCurrCfgVals(simConfig_.sceneLightSetupKey,
                                simConfig_.frustumCulling);
  } else {
    ESP_ERROR() << "No active dataset exists or has been specified. Aborting";
    return false;
  }
  ESP_DEBUG() << "Set new simulator config for scene/stage :"
              << simConfig_.activeSceneName
              << "and dataset :" << simConfig_.sceneDatasetConfigFile << "which"
              << (activeSceneDataset_ == simConfig_.sceneDatasetConfigFile
                      ? "is currently active dataset."
                      : "is NOT active dataset (THIS IS PROBABLY AN ERROR.)");
  return true;
}  // MetadataMediator::setSimulatorConfiguration

bool MetadataMediator::createPhysicsManagerAttributes(
    const std::string& _physicsManagerAttributesPath) {
  bool exists = physicsAttributesManager_->getObjectLibHasHandle(
      _physicsManagerAttributesPath);
  if (!exists) {
    auto physAttrs = physicsAttributesManager_->createObject(
        _physicsManagerAttributesPath, true);

    if (physAttrs == nullptr) {
      // something failed during creation process.
      ESP_WARNING()
          << "Unknown physics manager configuration file :"
          << _physicsManagerAttributesPath
          << "does not exist and is not able to be created.  Aborting.";
      return false;
    }
  }  // if dne then create
  return true;
}  // MetadataMediator::createPhysicsManagerAttributes

bool MetadataMediator::createSceneDataset(const std::string& sceneDatasetName,
                                          bool overwrite) {
  // see if exists
  bool exists = sceneDatasetExists(sceneDatasetName);
  if (exists) {
    // check if not overwrite and exists already
    if (!overwrite) {
      ESP_WARNING() << "Scene Dataset" << sceneDatasetName
                    << "already exists.  To reload and overwrite existing "
                       "data, set overwrite to true. Aborting.";
      return false;
    }
    // overwrite specified, make sure not locked
    sceneDatasetAttributesManager_->setLock(sceneDatasetName, false);
  }
  // by here dataset either does not exist or exists but is unlocked.
  auto datasetAttribs =
      sceneDatasetAttributesManager_->createObject(sceneDatasetName, true);
  if (datasetAttribs == nullptr) {
    // not created, do not set name
    ESP_WARNING() << "Unknown dataset" << sceneDatasetName
                  << "does not exist and is not able to be created.  Aborting.";
    return false;
  }
  // if not null then successfully created
  ESP_DEBUG() << "Dataset" << sceneDatasetName << "successfully created.";
  // lock dataset to prevent accidental deletion
  sceneDatasetAttributesManager_->setLock(sceneDatasetName, true);
  return true;
}  // MetadataMediator::createSceneDataset

bool MetadataMediator::removeSceneDataset(const std::string& sceneDatasetName) {
  // First check if SceneDatasetAttributes exists
  if (!sceneDatasetExists(sceneDatasetName)) {
    // DNE, do nothing
    ESP_WARNING() << "SceneDatasetAttributes" << sceneDatasetName
                  << "does not exist. Aborting.";
    return false;
  }

  // Next check if is current activeSceneDataset_, and if so skip with warning
  if (sceneDatasetName == activeSceneDataset_) {
    ESP_WARNING() << "Cannot remove "
                     "active SceneDatasetAttributes"
                  << sceneDatasetName
                  << ".  Switch to another dataset before removing.";
    return false;
  }

  // Now force unlock and remove requested SceneDatasetAttributes- there should
  // be no SceneDatasetAttributes set to undeletable
  sceneDatasetAttributesManager_->setLock(sceneDatasetName, false);
  auto delDataset =
      sceneDatasetAttributesManager_->removeObjectByHandle(sceneDatasetName);
  // if failed here, probably means SceneDatasetAttributes was set to
  // undeletable, return message and false
  if (delDataset == nullptr) {
    ESP_WARNING() << "SceneDatasetAttributes" << sceneDatasetName
                  << "unable to be deleted. Aborting.";
    return false;
  }
  // Should always have a default dataset. Use this process to remove extraneous
  // configs in default Scene Dataset
  if (sceneDatasetName == "default") {
    // removing default dataset should still create another, empty, default
    // dataset.
    createSceneDataset("default");
  }
  ESP_DEBUG() << "SceneDatasetAttributes" << sceneDatasetName
              << "successfully removed.";
  return true;

}  // MetadataMediator::removeSceneDataset

bool MetadataMediator::setCurrPhysicsAttributesHandle(
    const std::string& _physicsManagerAttributesPath) {
  // first check if physics manager attributes exists, if so then set as current
  if (physicsAttributesManager_->getObjectLibHasHandle(
          _physicsManagerAttributesPath)) {
    if (currPhysicsManagerAttributes_ != _physicsManagerAttributesPath) {
      ESP_DEBUG() << "Old physics manager attributes"
                  << currPhysicsManagerAttributes_ << "changed to"
                  << _physicsManagerAttributesPath << "successfully.";
      currPhysicsManagerAttributes_ = _physicsManagerAttributesPath;
    }
    sceneDatasetAttributesManager_->setCurrPhysicsManagerAttributesHandle(
        currPhysicsManagerAttributes_);
    return true;
  }
  // if this handle does not exist, create the attributes for it.
  bool success = createPhysicsManagerAttributes(_physicsManagerAttributesPath);
  // if successfully created, set default name to physics manager attributes in
  // SceneDatasetAttributesManager
  if (success) {
    currPhysicsManagerAttributes_ = _physicsManagerAttributesPath;
    sceneDatasetAttributesManager_->setCurrPhysicsManagerAttributesHandle(
        currPhysicsManagerAttributes_);
    /// setCurrPhysicsManagerAttributesHandle
  }
  return success;

}  // MetadataMediator::setCurrPhysicsAttributesHandle

bool MetadataMediator::setActiveSceneDatasetName(
    const std::string& sceneDatasetName) {
  // first check if dataset exists/is loaded, if so then set as default
  if (sceneDatasetExists(sceneDatasetName)) {
    if (activeSceneDataset_ != sceneDatasetName) {
      ESP_DEBUG() << "Previous active dataset" << activeSceneDataset_
                  << "changed to" << sceneDatasetName << "successfully.";
      activeSceneDataset_ = sceneDatasetName;
    }
    return true;
  }
  // if does not exist, attempt to create it
  ESP_DEBUG() << "Attempting to create new dataset" << sceneDatasetName;
  bool success = createSceneDataset(sceneDatasetName);
  // if successfully created, set default name to access dataset attributes in
  // SceneDatasetAttributesManager
  if (success) {
    activeSceneDataset_ = sceneDatasetName;
  }
  ESP_DEBUG() << "Attempt to create new dataset" << sceneDatasetName << ""
              << (success ? " succeeded." : " failed.")
              << "Currently active dataset :" << activeSceneDataset_;
  return success;
}  // MetadataMediator::setActiveSceneDatasetName

attributes::SceneInstanceAttributes::ptr
MetadataMediator::getSceneInstanceAttributesByName(
    const std::string& sceneName) {
  // get current dataset attributes
  attributes::SceneDatasetAttributes::ptr datasetAttr = getActiveDSAttribs();
  // this should never happen
  if (datasetAttr == nullptr) {
    ESP_ERROR() << "No dataset specified/exists.";
    return nullptr;
  }
  // directory to look for attributes for this dataset
  const std::string dsDir = datasetAttr->getFileDirectory();
  // get appropriate attr managers for the current dataset
  managers::SceneInstanceAttributesManager::ptr dsSceneAttrMgr =
      datasetAttr->getSceneInstanceAttributesManager();
  managers::StageAttributesManager::ptr dsStageAttrMgr =
      datasetAttr->getStageAttributesManager();

  attributes::SceneInstanceAttributes::ptr sceneInstanceAttributes = nullptr;
  // get list of scene attributes handles that contain sceneName as a substring
  auto sceneList = dsSceneAttrMgr->getObjectHandlesBySubstring(sceneName);
  // sceneName can legally match any one of the following conditions :
  if (!sceneList.empty()) {
    // 1.  Existing, registered SceneInstanceAttributes in current active
    // dataset.
    //    In this case the SceneInstanceAttributes is returned.
    ESP_DEBUG() << "Query dataset :" << activeSceneDataset_
                << "for SceneInstanceAttributes named :" << sceneName
                << "yields" << sceneList.size() << "candidates.  Using"
                << sceneList[0] << Mn::Debug::nospace << ".";
    sceneInstanceAttributes =
        dsSceneAttrMgr->getObjectCopyByHandle(sceneList[0]);
  } else {
    const std::string sceneFilenameCandidate =
        dsSceneAttrMgr->getFormattedJSONFileName(sceneName);

    if (Cr::Utility::Path::exists(sceneFilenameCandidate)) {
      // 2.  Existing, valid SceneInstanceAttributes file on disk, but not in
      // dataset.
      //    If this is the case, then the SceneInstanceAttributes should be
      //    loaded, registered, added to the dataset and returned.
      ESP_DEBUG() << "Dataset :" << activeSceneDataset_
                  << "does not reference a SceneInstanceAttributes named"
                  << sceneName << "but a SceneInstanceAttributes config named"
                  << sceneFilenameCandidate << "was found on disk, so loading.";
      sceneInstanceAttributes = dsSceneAttrMgr->createObjectFromJSONFile(
          sceneFilenameCandidate, true);
    } else {
      // get list of stage attributes handles that contain sceneName as a
      // substring
      auto stageList = dsStageAttrMgr->getObjectHandlesBySubstring(sceneName);
      if (!stageList.empty()) {
        // 3.  Existing, registered StageAttributes in current active dataset.
        //    In this case, a SceneInstanceAttributes is created amd registered
        //    using sceneName, referencing the StageAttributes of the same name;
        //    This sceneInstanceAttributes is returned.
        ESP_DEBUG()
            << "No existing scene instance attributes containing name"
            << sceneName << "found in Dataset :" << activeSceneDataset_ << "but"
            << stageList.size() << "StageAttributes found.  Using"
            << stageList[0]
            << "as stage and to construct a SceneInstanceAttributes with same "
               "name that will be added to Dataset.";
        // create a new SceneInstanceAttributes, and give it a
        // SceneObjectInstanceAttributes for the stage with the same name.
        sceneInstanceAttributes = makeSceneAndReferenceStage(
            datasetAttr, dsStageAttrMgr->getObjectByHandle(stageList[0]),
            dsSceneAttrMgr, sceneName);

      } else {
        // 4.  Either existing stage config/asset on disk, but not in current
        // dataset, or no stage config/asset exists with passed name.
        //    In this case, a stage attributes is loaded/created and registered,
        //    then added to current dataset, and then 3. is performed.
        ESP_DEBUG() << "Dataset :" << activeSceneDataset_
                    << "has no preloaded SceneInstanceAttributes or "
                       "StageAttributes named :"
                    << sceneName
                    << "so loading/creating a new StageAttributes with this "
                       "name, and then creating a SceneInstanceAttributes with "
                       "the same name that references this stage.";
        // create and register stage attributes
        auto stageAttributes = dsStageAttrMgr->createObject(sceneName, true);
        // create a new SceneInstanceAttributes, and give it a
        // SceneObjectInstanceAttributes for the stage with the same name.
        sceneInstanceAttributes = makeSceneAndReferenceStage(
            datasetAttr, stageAttributes, dsSceneAttrMgr, sceneName);
      }
    }
  }
  // make sure that all stage, object and lighting attributes referenced in
  // scene attributes are loaded in dataset, as well as the scene attributes
  // itself.
  datasetAttr->addNewSceneInstanceToDataset(sceneInstanceAttributes);

  return sceneInstanceAttributes;

}  // MetadataMediator::getSceneInstanceAttributesByName

attributes::SceneInstanceAttributes::ptr
MetadataMediator::makeSceneAndReferenceStage(
    const attributes::SceneDatasetAttributes::ptr& datasetAttr,
    const attributes::StageAttributes::ptr& stageAttributes,
    const managers::SceneInstanceAttributesManager::ptr& dsSceneAttrMgr,
    const std::string& sceneName) {
  ESP_CHECK(datasetAttr != nullptr && stageAttributes != nullptr &&
                dsSceneAttrMgr != nullptr,
            "Missing (at least) one of scene dataset attributes, stage "
            "attributes, or dataset scene attributes for scene '"
                << Mn::Debug::nospace << sceneName << Mn::Debug::nospace
                << "'.  Likely an invalid scene name.");
  // create scene attributes with passed name
  attributes::SceneInstanceAttributes::ptr sceneInstanceAttributes =
      dsSceneAttrMgr->createDefaultObject(sceneName, false);
  // create stage instance attributes and set its name (from stage attributes)
  sceneInstanceAttributes->setStageInstance(
      dsSceneAttrMgr->createEmptyInstanceAttributes(
          stageAttributes->getHandle()));

  // The following is to manage stage files that have navmesh and semantic scene
  // descriptor ("house file") handles in them. This mechanism has been
  // deprecated, but in order to provide backwards compatibility, we are going
  // to support these values here when we synthesize a non-existing scene
  // instance attributes only.

  // add a ref to the navmesh path from the stage attributes to scene
  // attributes, giving it an appropriately obvious name.  This entails adding
  // the path itself to the dataset, if it does not already exist there, keyed
  // by the ref that the scene attributes will use.
  std::pair<std::string, std::string> navmeshEntry =
      datasetAttr->addNavmeshPathEntry(
          sceneName, stageAttributes->getNavmeshAssetHandle(), false);
  // navmeshEntry holds the navmesh key-value in the dataset to use by this
  // scene instance.  NOTE : the key may have changed from what was passed if a
  // collision occurred with same key but different value, so we need to add
  // this key to the scene instance attributes.
  sceneInstanceAttributes->setNavmeshHandle(navmeshEntry.first);

  // add a ref to semantic scene descriptor ("house file") from stage attributes
  // to scene attributes, giving it an appropriately obvious name.  This entails
  // adding the path itself to the dataset, if it does not already exist there,
  // keyed by the ref that the scene attributes will use.
  std::pair<std::string, std::string> ssdEntry =
      datasetAttr->addSemanticSceneDescrPathEntry(
          sceneName, stageAttributes->getSemanticDescriptorFilename(), false);
  // ssdEntry holds the ssd key in the dataset to use by this scene instance.
  // NOTE : the key may have changed from what was passed if a collision
  // occurred with same key but different value, so we need to add this key to
  // the scene instance attributes.
  sceneInstanceAttributes->setSemanticSceneHandle(ssdEntry.first);

  // register SceneInstanceAttributes object
  dsSceneAttrMgr->registerObject(sceneInstanceAttributes);
  return sceneInstanceAttributes;
}  // MetadataMediator::makeSceneAndReferenceStage

std::string MetadataMediator::getFilePathForHandle(
    const std::string& assetHandle,
    const std::map<std::string, std::string>& assetMapping,
    const std::string& msgString) {
  std::map<std::string, std::string>::const_iterator mapIter =
      assetMapping.find(assetHandle);
  if (mapIter == assetMapping.end()) {
    ESP_WARNING() << msgString << ": Unable to find file path for"
                  << assetHandle << ".  Aborting.";
    return "";
  }
  return mapIter->second;
}  // MetadataMediator::getFilePathForHandle

std::string MetadataMediator::getDatasetsOverview() const {
  // reserve space for info strings for all scene datasets
  std::vector<std::string> sceneDatasetHandles =
      sceneDatasetAttributesManager_->getObjectHandlesBySubstring("");
  std::string res =
      "Datasets : \n" +
      attributes::SceneDatasetAttributes::getDatasetSummaryHeader() + "\n";
  for (const std::string& handle : sceneDatasetHandles) {
    res += sceneDatasetAttributesManager_->getObjectByHandle(handle)
               ->getDatasetSummary();
    res += '\n';
  }

  return res;
}  // MetadataMediator::getDatasetNames

std::string MetadataMediator::createDatasetReport(
    const std::string& sceneDataset) const {
  attributes::SceneDatasetAttributes::ptr ds;
  if (sceneDataset == "") {
    ds = sceneDatasetAttributesManager_->getObjectByHandle(activeSceneDataset_);

  } else if (sceneDatasetAttributesManager_->getObjectLibHasHandle(
                 sceneDataset)) {
    ds = sceneDatasetAttributesManager_->getObjectByHandle(sceneDataset);
  } else {
    // unknown dataset
    ESP_ERROR() << "Dataset" << sceneDataset
                << "is not found in the MetadataMediator.  Aborting.";
    return "Requeseted SceneDataset `" + sceneDataset + "` unknown.";
  }
  return Corrade::Utility::formatString(
      "Scene Dataset {}\n{}\n", ds->getObjectInfoHeader(), ds->getObjectInfo());
}  // MetadataMediator::const std::string MetadataMediator::createDatasetReport(

}  // namespace metadata
}  // namespace esp
