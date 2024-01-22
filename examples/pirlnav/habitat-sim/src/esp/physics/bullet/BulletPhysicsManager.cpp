// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//#include "BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h"
//#include "BulletCollision/Gimpact/btGImpactShape.h"

#include "BulletPhysicsManager.h"
#include "BulletArticulatedObject.h"
#include "BulletDynamics/Featherstone/btMultiBodyLinkCollider.h"
#include "BulletRigidObject.h"
#include "BulletURDFImporter.h"
#include "esp/assets/ResourceManager.h"
#include "esp/physics/objectManagers/ArticulatedObjectManager.h"
#include "esp/physics/objectManagers/RigidObjectManager.h"
#include "esp/sim/Simulator.h"

namespace esp {
namespace physics {

BulletPhysicsManager::BulletPhysicsManager(
    assets::ResourceManager& _resourceManager,
    const metadata::attributes::PhysicsManagerAttributes::cptr&
        _physicsManagerAttributes)
    : PhysicsManager(_resourceManager, _physicsManagerAttributes) {
  collisionObjToObjIds_ =
      std::make_shared<std::map<const btCollisionObject*, int>>();
  urdfImporter_ = std::make_unique<BulletURDFImporter>(_resourceManager);
  if (_resourceManager.getCreateRenderer()) {
    debugDrawer_ = std::make_unique<Magnum::BulletIntegration::DebugDraw>();
  }
}

BulletPhysicsManager::~BulletPhysicsManager() {
  ESP_DEBUG() << "Deconstructing BulletPhysicsManager";

  existingObjects_.clear();
  existingArticulatedObjects_.clear();
  staticStageObject_.reset();
}

void BulletPhysicsManager::removeObject(const int objectId,
                                        bool deleteObjectNode,
                                        bool deleteVisualNode) {
  removeObjectRigidConstraints(objectId);
  PhysicsManager::removeObject(objectId, deleteObjectNode, deleteVisualNode);
}

void BulletPhysicsManager::removeArticulatedObject(int objectId) {
  removeObjectRigidConstraints(objectId);
  PhysicsManager::removeArticulatedObject(objectId);
}

bool BulletPhysicsManager::initPhysicsFinalize() {
  activePhysSimLib_ = PhysicsSimulationLibrary::Bullet;

  //! We can potentially use other collision checking algorithms, by
  //! uncommenting the line below
  // btGImpactCollisionAlgorithm::registerAlgorithm(&bDispatcher_);
  bWorld_ = std::make_shared<btMultiBodyDynamicsWorld>(
      &bDispatcher_, &bBroadphase_, &bSolver_, &bCollisionConfig_);

  if (debugDrawer_) {
    debugDrawer_->setMode(
        Magnum::BulletIntegration::DebugDraw::Mode::DrawWireframe |
        Magnum::BulletIntegration::DebugDraw::Mode::DrawConstraints);
    bWorld_->setDebugDrawer(debugDrawer_.get());
  }

  // currently GLB meshes are y-up
  bWorld_->setGravity(
      btVector3(physicsManagerAttributes_->get<Magnum::Vector3>("gravity")));

  //! Create new scene node
  staticStageObject_ = physics::BulletRigidStage::create(
      &physicsNode_->createChild(), resourceManager_, bWorld_,
      collisionObjToObjIds_);

  recentNumSubStepsTaken_ = -1;
  return true;
}

// Bullet Mesh conversion adapted from:
// https://github.com/mosra/magnum-integration/issues/20
bool BulletPhysicsManager::addStageFinalize(
    const metadata::attributes::StageAttributes::ptr& initAttributes) {
  //! Initialize BulletRigidStage
  bool sceneSuccess = staticStageObject_->initialize(initAttributes);

  return sceneSuccess;
}

bool BulletPhysicsManager::makeAndAddRigidObject(
    int newObjectID,
    const esp::metadata::attributes::ObjectAttributes::ptr& objectAttributes,
    scene::SceneNode* objectNode) {
  auto ptr = physics::BulletRigidObject::create(objectNode, newObjectID,
                                                resourceManager_, bWorld_,
                                                collisionObjToObjIds_);
  bool objSuccess = ptr->initialize(objectAttributes);
  if (objSuccess) {
    existingObjects_.emplace(newObjectID, std::move(ptr));
  }
  return objSuccess;
}

int BulletPhysicsManager::addArticulatedObjectFromURDF(
    const std::string& filepath,
    bool fixedBase,
    float globalScale,
    float massScale,
    bool forceReload,
    bool maintainLinkOrder,
    const std::string& lightSetup) {
  auto& drawables = simulator_->getDrawableGroup();
  return addArticulatedObjectFromURDF(filepath, &drawables, fixedBase,
                                      globalScale, massScale, forceReload,
                                      maintainLinkOrder, lightSetup);
}

int BulletPhysicsManager::addArticulatedObjectFromURDF(
    const std::string& filepath,
    DrawableGroup* drawables,
    bool fixedBase,
    float globalScale,
    float massScale,
    bool forceReload,
    bool maintainLinkOrder,
    const std::string& lightSetup) {
  if (simulator_ != nullptr) {
    // aquire context if available
    simulator_->getRenderGLContext();
  }
  ESP_CHECK(
      urdfImporter_->loadURDF(filepath, globalScale, massScale, forceReload),
      "failed to parse/load URDF file" << filepath);

  int articulatedObjectID = allocateObjectID();

  // parse succeeded, attempt to create the articulated object
  scene::SceneNode* objectNode = &staticStageObject_->node().createChild();
  BulletArticulatedObject::ptr articulatedObject =
      BulletArticulatedObject::create(objectNode, resourceManager_,
                                      articulatedObjectID, bWorld_,
                                      collisionObjToObjIds_);

  // before initializing the URDF, import all necessary assets in advance
  urdfImporter_->importURDFAssets();

  BulletURDFImporter* u2b =
      static_cast<BulletURDFImporter*>(urdfImporter_.get());

  u2b->setFixedBase(fixedBase);

  // TODO: set these flags up better
  u2b->flags = 0;
  if (maintainLinkOrder) {
    u2b->flags |= CUF_MAINTAIN_LINK_ORDER;
  }
  u2b->initURDF2BulletCache();

  articulatedObject->initializeFromURDF(*urdfImporter_, {}, physicsNode_);

  // allocate ids for links
  for (int linkIx = 0; linkIx < articulatedObject->btMultiBody_->getNumLinks();
       ++linkIx) {
    int linkObjectId = allocateObjectID();
    articulatedObject->objectIdToLinkId_[linkObjectId] = linkIx;
    collisionObjToObjIds_->emplace(
        articulatedObject->btMultiBody_->getLinkCollider(linkIx), linkObjectId);
  }

  // attach link visual shapes
  for (size_t urdfLinkIx = 0; urdfLinkIx < u2b->getModel()->m_links.size();
       ++urdfLinkIx) {
    auto urdfLink = u2b->getModel()->getLink(urdfLinkIx);
    if (!urdfLink->m_visualArray.empty()) {
      int bulletLinkIx =
          u2b->cache->m_urdfLinkIndices2BulletLinkIndices[urdfLinkIx];
      ArticulatedLink& linkObject = articulatedObject->getLink(bulletLinkIx);
      ESP_CHECK(
          attachLinkGeometry(&linkObject, urdfLink, drawables, lightSetup),
          "BulletPhysicsManager::addArticulatedObjectFromURDF(): Failed to "
          "instance render asset (attachGeometry) for link"
              << urdfLinkIx << ".");
      linkObject.node().computeCumulativeBB();
    }
  }

  // clear the cache
  u2b->cache = nullptr;

  // base collider refers to the articulated object's id
  collisionObjToObjIds_->emplace(
      articulatedObject->btMultiBody_->getBaseCollider(), articulatedObjectID);

  existingArticulatedObjects_.emplace(articulatedObjectID,
                                      std::move(articulatedObject));

  // get a simplified name of the handle for the object
  std::string simpleArtObjHandle =
      Corrade::Utility::Path::splitExtension(
          Corrade::Utility::Path::splitExtension(
              Corrade::Utility::Path::split(filepath).second())
              .first())
          .first();

  std::string newArtObjectHandle =
      articulatedObjectManager_->getUniqueHandleFromCandidate(
          simpleArtObjHandle);
  ESP_DEBUG() << "simpleArtObjHandle :" << simpleArtObjHandle
              << " | newArtObjectHandle :" << newArtObjectHandle;

  existingArticulatedObjects_.at(articulatedObjectID)
      ->setObjectName(newArtObjectHandle);

  // 2.0 Get wrapper - name is irrelevant, do not register on create.
  ManagedArticulatedObject::ptr AObjWrapper = getArticulatedObjectWrapper();

  // 3.0 Put articulated object in wrapper
  AObjWrapper->setObjectRef(
      existingArticulatedObjects_.at(articulatedObjectID));

  // 4.0 register wrapper in manager
  articulatedObjectManager_->registerObject(AObjWrapper, newArtObjectHandle);

  return articulatedObjectID;
}  // BulletPhysicsManager::addArticulatedObjectFromURDF

esp::physics::ManagedRigidObject::ptr
BulletPhysicsManager::getRigidObjectWrapper() {
  // TODO make sure this is appropriately cast
  return rigidObjectManager_->createObject("ManagedBulletRigidObject");
}

esp::physics::ManagedArticulatedObject::ptr
BulletPhysicsManager::getArticulatedObjectWrapper() {
  // TODO make sure this is appropriately cast
  return articulatedObjectManager_->createObject(
      "ManagedBulletArticulatedObject");
}

//! Check if mesh primitive is compatible with physics
bool BulletPhysicsManager::isMeshPrimitiveValid(
    const assets::CollisionMeshData& meshData) {
  if (meshData.primitive == Magnum::MeshPrimitive::Triangles) {
    //! Only triangle mesh works
    return true;
  } else {
    switch (meshData.primitive) {
      case Magnum::MeshPrimitive::Lines:
        ESP_ERROR() << "Invalid primitive: Lines";
        break;
      case Magnum::MeshPrimitive::Points:
        ESP_ERROR() << "Invalid primitive: Points";
        break;
      case Magnum::MeshPrimitive::LineLoop:
        ESP_ERROR() << "Invalid primitive Line loop";
        break;
      case Magnum::MeshPrimitive::LineStrip:
        ESP_ERROR() << "Invalid primitive Line Strip";
        break;
      case Magnum::MeshPrimitive::TriangleStrip:
        ESP_ERROR() << "Invalid primitive Triangle Strip";
        break;
      case Magnum::MeshPrimitive::TriangleFan:
        ESP_ERROR() << "Invalid primitive Triangle Fan";
        break;
      default:
        ESP_ERROR() << "Invalid primitive" << int(meshData.primitive);
    }
    ESP_ERROR() << "Cannot load collision mesh, skipping";
    return false;
  }
}

bool BulletPhysicsManager::attachLinkGeometry(
    ArticulatedLink* linkObject,
    const std::shared_ptr<io::URDF::Link>& link,
    gfx::DrawableGroup* drawables,
    const std::string& lightSetup) {
  const bool forceFlatShading = (lightSetup == esp::NO_LIGHT_KEY);
  bool geomSuccess = false;

  for (auto& visual : link->m_visualArray) {
    bool visualSetupSuccess = true;
    // create a new child for each visual component
    scene::SceneNode& visualGeomComponent = linkObject->node().createChild();
    // cache the visual node
    linkObject->visualNodes_.push_back(&visualGeomComponent);
    visualGeomComponent.setType(esp::scene::SceneNodeType::OBJECT);
    visualGeomComponent.setTransformation(
        link->m_inertia.m_linkLocalFrame.invertedRigid() *
        visual.m_linkLocalFrame);

    // prep the AssetInfo, overwrite the filepath later
    assets::AssetInfo visualMeshInfo{assets::AssetType::UNKNOWN};
    visualMeshInfo.forceFlatShading = forceFlatShading;

    // create a modified asset if necessary for material override
    std::shared_ptr<io::URDF::Material> material =
        visual.m_geometry.m_localMaterial;
    if (material) {
      visualMeshInfo.overridePhongMaterial = assets::PhongMaterialColor();
      visualMeshInfo.overridePhongMaterial->ambientColor =
          material->m_matColor.m_rgbaColor;
      visualMeshInfo.overridePhongMaterial->diffuseColor =
          material->m_matColor.m_rgbaColor;
      visualMeshInfo.overridePhongMaterial->specularColor =
          Mn::Color4(material->m_matColor.m_specularColor);
    }

    switch (visual.m_geometry.m_type) {
      case io::URDF::GEOM_CAPSULE:
        visualMeshInfo.type = esp::assets::AssetType::PRIMITIVE;
        // should be registered and cached already
        visualMeshInfo.filepath = visual.m_geometry.m_meshFileName;
        // scale by radius as suggested by magnum docs
        visualGeomComponent.scale(
            Mn::Vector3(visual.m_geometry.m_capsuleRadius));
        // Magnum capsule is Y up, URDF is Z up
        visualGeomComponent.setTransformation(
            visualGeomComponent.transformation() *
            Mn::Matrix4::rotationX(Mn::Rad(M_PI_2)));
        break;
      case io::URDF::GEOM_CYLINDER:
        visualMeshInfo.type = esp::assets::AssetType::PRIMITIVE;
        // the default created primitive handle for the cylinder with radius 1
        // and length 2
        visualMeshInfo.filepath =
            "cylinderSolid_rings_1_segments_12_halfLen_1_useTexCoords_false_"
            "useTangents_false_capEnds_true";
        visualGeomComponent.scale(
            Mn::Vector3(visual.m_geometry.m_capsuleRadius,
                        visual.m_geometry.m_capsuleHeight / 2.0,
                        visual.m_geometry.m_capsuleRadius));
        // Magnum cylinder is Y up, URDF is Z up
        visualGeomComponent.setTransformation(
            visualGeomComponent.transformation() *
            Mn::Matrix4::rotationX(Mn::Rad(M_PI_2)));
        break;
      case io::URDF::GEOM_BOX:
        visualMeshInfo.type = esp::assets::AssetType::PRIMITIVE;
        visualMeshInfo.filepath = "cubeSolid";
        visualGeomComponent.scale(visual.m_geometry.m_boxSize * 0.5);
        break;
      case io::URDF::GEOM_SPHERE: {
        visualMeshInfo.type = esp::assets::AssetType::PRIMITIVE;
        // default sphere prim is already constructed w/ radius 1
        visualMeshInfo.filepath = "icosphereSolid_subdivs_1";
        visualGeomComponent.scale(
            Mn::Vector3(visual.m_geometry.m_sphereRadius));
      } break;
      case io::URDF::GEOM_MESH: {
        visualGeomComponent.scale(visual.m_geometry.m_meshScale);
        visualMeshInfo.filepath = visual.m_geometry.m_meshFileName;
      } break;
      case io::URDF::GEOM_PLANE:
        ESP_DEBUG() << "Trying to add visual plane, not implemented";
        // TODO:
        visualSetupSuccess = false;
        break;
      default:
        ESP_DEBUG() << "Unsupported visual type.";
        visualSetupSuccess = false;
        break;
    }

    // add the visual shape to the SceneGraph
    if (visualSetupSuccess) {
      assets::RenderAssetInstanceCreationInfo::Flags flags;
      flags |= assets::RenderAssetInstanceCreationInfo::Flag::IsRGBD;
      flags |= assets::RenderAssetInstanceCreationInfo::Flag::IsSemantic;
      assets::RenderAssetInstanceCreationInfo creation(
          visualMeshInfo.filepath, Mn::Vector3{1}, flags, lightSetup);

      geomSuccess = resourceManager_.loadAndCreateRenderAssetInstance(
                        visualMeshInfo, creation, &visualGeomComponent,
                        drawables, &linkObject->visualNodes_) != nullptr;

      // cache the visual component for later query
      if (geomSuccess) {
        linkObject->visualAttachments_.emplace_back(
            &visualGeomComponent, visual.m_geometry.m_meshFileName);
      }
    }
  }

  return geomSuccess;
}

void BulletPhysicsManager::setGravity(const Magnum::Vector3& gravity) {
  bWorld_->setGravity(btVector3(gravity));
  // After gravity change, need to reactivate all bullet objects
  for (auto& it : existingObjects_) {
    it.second->setActive(true);
  }
  for (auto& it : existingArticulatedObjects_) {
    it.second->setActive(true);
  }
}

Magnum::Vector3 BulletPhysicsManager::getGravity() const {
  return Magnum::Vector3(bWorld_->getGravity());
}

void BulletPhysicsManager::stepPhysics(double dt) {
  // We don't step uninitialized physics sim...
  if (!initialized_) {
    return;
  }
  if (dt <= 0) {
    dt = fixedTimeStep_;
  }

  // set specified control velocities
  for (auto& objectItr : existingObjects_) {
    VelocityControl::ptr velControl = objectItr.second->getVelocityControl();
    if (objectItr.second->getMotionType() == MotionType::KINEMATIC) {
      // kinematic velocity control integration
      if (velControl->controllingAngVel || velControl->controllingLinVel) {
        objectItr.second->setRigidState(velControl->integrateTransform(
            dt, objectItr.second->getRigidState()));
        objectItr.second->setActive(true);
      }
    } else if (objectItr.second->getMotionType() == MotionType::DYNAMIC) {
      if (velControl->controllingLinVel) {
        if (velControl->linVelIsLocal) {
          objectItr.second->setLinearVelocity(
              objectItr.second->node().rotation().transformVector(
                  velControl->linVel));
        } else {
          objectItr.second->setLinearVelocity(velControl->linVel);
        }
      }
      if (velControl->controllingAngVel) {
        if (velControl->angVelIsLocal) {
          objectItr.second->setAngularVelocity(
              objectItr.second->node().rotation().transformVector(
                  velControl->angVel));
        } else {
          objectItr.second->setAngularVelocity(velControl->angVel);
        }
      }
    }
  }

  // extra step to validate joint states against limits for corrective clamping
  for (auto& objectItr : existingArticulatedObjects_) {
    if (objectItr.second->getAutoClampJointLimits()) {
      static_cast<BulletArticulatedObject*>(objectItr.second.get())
          ->clampJointLimits();
    }
  }

  // ==== Physics stepforward ======
  // NOTE: worldTime_ will always be a multiple of sceneMetaData_.timestep
  int numSubStepsTaken =
      bWorld_->stepSimulation(dt, /*maxSubSteps*/ 10000, fixedTimeStep_);
  worldTime_ += numSubStepsTaken * fixedTimeStep_;
  recentNumSubStepsTaken_ = numSubStepsTaken;
  recentTimeStep_ = fixedTimeStep_;
}

void BulletPhysicsManager::setStageFrictionCoefficient(
    const double frictionCoefficient) {
  staticStageObject_->setFrictionCoefficient(frictionCoefficient);
}

void BulletPhysicsManager::setStageRestitutionCoefficient(
    const double restitutionCoefficient) {
  staticStageObject_->setRestitutionCoefficient(restitutionCoefficient);
}

double BulletPhysicsManager::getStageFrictionCoefficient() const {
  return staticStageObject_->getFrictionCoefficient();
}

double BulletPhysicsManager::getStageRestitutionCoefficient() const {
  return staticStageObject_->getRestitutionCoefficient();
}

Magnum::Range3D BulletPhysicsManager::getCollisionShapeAabb(
    const int physObjectID) const {
  auto objIter = getConstRigidObjIteratorOrAssert(physObjectID);
  return static_cast<BulletRigidObject*>(objIter->second.get())
      ->getCollisionShapeAabb();
}

Magnum::Range3D BulletPhysicsManager::getStageCollisionShapeAabb() const {
  return static_cast<BulletRigidStage*>(staticStageObject_.get())
      ->getCollisionShapeAabb();
}

void BulletPhysicsManager::debugDraw(const Magnum::Matrix4& projTrans) const {
  if (debugDrawer_) {
    debugDrawer_->setTransformationProjectionMatrix(projTrans);
    bWorld_->debugDrawWorld();
  }
}

RaycastResults BulletPhysicsManager::castRay(const esp::geo::Ray& ray,
                                             double maxDistance) {
  RaycastResults results;
  results.ray = ray;
  double rayLength = static_cast<double>(ray.direction.length());
  if (rayLength == 0) {
    ESP_ERROR() << "Cannot cast ray with zero length, aborting.";
    return results;
  }
  btVector3 from(ray.origin);
  btVector3 to(ray.origin + ray.direction * maxDistance);

  btCollisionWorld::AllHitsRayResultCallback allResults(from, to);
  bWorld_->rayTest(from, to, allResults);

  // convert to RaycastResults
  for (int i = 0; i < allResults.m_hitPointWorld.size(); ++i) {
    RayHitInfo hit;

    hit.normal = Magnum::Vector3{allResults.m_hitNormalWorld[i]};
    hit.point = Magnum::Vector3{allResults.m_hitPointWorld[i]};
    hit.rayDistance =
        (static_cast<double>(allResults.m_hitFractions[i]) * maxDistance) /
        rayLength;
    // default to -1 for "scene collision" if we don't know which object was
    // involved
    hit.objectId = -1;
    auto rawColObjIdIter =
        collisionObjToObjIds_->find(allResults.m_collisionObjects[i]);
    if (rawColObjIdIter != collisionObjToObjIds_->end()) {
      hit.objectId = rawColObjIdIter->second;
    }
    results.hits.push_back(hit);
  }
  results.sortByDistance();
  return results;
}

void BulletPhysicsManager::lookUpObjectIdAndLinkId(
    const btCollisionObject* colObj,
    int* objectId,
    int* linkId) const {
  CORRADE_INTERNAL_ASSERT(objectId);
  CORRADE_INTERNAL_ASSERT(linkId);

  *linkId = -1;
  // If the lookup fails, default to the stage. TODO: better error-handling.
  *objectId = -1;
  auto rawColObjIdIter = collisionObjToObjIds_->find(colObj);
  if (rawColObjIdIter != collisionObjToObjIds_->end()) {
    int rawObjectId = rawColObjIdIter->second;
    if (existingObjects_.count(rawObjectId) != 0u ||
        existingArticulatedObjects_.count(rawObjectId) != 0u) {
      *objectId = rawObjectId;
      return;
    } else {
      // search articulated objects to see if this is a link
      for (const auto& pair : existingArticulatedObjects_) {
        auto objIdToLinkIter = pair.second->objectIdToLinkId_.find(rawObjectId);
        if (objIdToLinkIter != pair.second->objectIdToLinkId_.end()) {
          *objectId = pair.first;
          *linkId = objIdToLinkIter->second;
          return;
        }
      }
    }
  }

  // lookup failed
}

std::vector<ContactPointData> BulletPhysicsManager::getContactPoints() const {
  std::vector<ContactPointData> contactPoints;

  auto* dispatcher = bWorld_->getDispatcher();
  int numContactManifolds = dispatcher->getNumManifolds();
  contactPoints.reserve(numContactManifolds * 4);
  for (int i = 0; i < numContactManifolds; ++i) {
    const btPersistentManifold* manifold =
        dispatcher->getInternalManifoldPointer()[i];

    int objectIdA = -2;  // stage is -1
    int objectIdB = -2;
    int linkIndexA = -1;  // -1 if not a multibody
    int linkIndexB = -1;

    const btCollisionObject* colObj0 = manifold->getBody0();
    const btCollisionObject* colObj1 = manifold->getBody1();

    lookUpObjectIdAndLinkId(colObj0, &objectIdA, &linkIndexA);
    lookUpObjectIdAndLinkId(colObj1, &objectIdB, &linkIndexB);

    // logic copied from btSimulationIslandManager::buildIslands. We count
    // manifolds as active only if related to non-sleeping bodies.
    bool isActive = ((((colObj0) != nullptr) &&
                      colObj0->getActivationState() != ISLAND_SLEEPING) ||
                     (((colObj1) != nullptr) &&
                      colObj1->getActivationState() != ISLAND_SLEEPING));

    for (int p = 0; p < manifold->getNumContacts(); ++p) {
      ContactPointData pt;
      pt.objectIdA = objectIdA;
      pt.objectIdB = objectIdB;
      const btManifoldPoint& srcPt = manifold->getContactPoint(p);
      pt.contactDistance = static_cast<double>(srcPt.getDistance());
      pt.linkIndexA = linkIndexA;
      pt.linkIndexB = linkIndexB;
      pt.contactNormalOnBInWS = Mn::Vector3(srcPt.m_normalWorldOnB);
      pt.positionOnAInWS = Mn::Vector3(srcPt.getPositionWorldOnA());
      pt.positionOnBInWS = Mn::Vector3(srcPt.getPositionWorldOnB());

      // convert impulses to forces w/ recent physics timstep
      pt.normalForce =
          static_cast<double>(srcPt.getAppliedImpulse()) / recentTimeStep_;

      pt.linearFrictionForce1 =
          static_cast<double>(srcPt.m_appliedImpulseLateral1) / recentTimeStep_;
      pt.linearFrictionForce2 =
          static_cast<double>(srcPt.m_appliedImpulseLateral2) / recentTimeStep_;

      pt.linearFrictionDirection1 = Mn::Vector3(srcPt.m_lateralFrictionDir1);
      pt.linearFrictionDirection2 = Mn::Vector3(srcPt.m_lateralFrictionDir2);

      pt.isActive = isActive;

      contactPoints.push_back(pt);
    }
  }

  return contactPoints;
}

//============ Rigid Constraints =============

int BulletPhysicsManager::createRigidConstraint(
    const RigidConstraintSettings& settings) {
  auto rigidObjAIter = existingObjects_.find(settings.objectIdA);
  bool objAIsRigid = rigidObjAIter != existingObjects_.end();
  auto artObjAIter = existingArticulatedObjects_.find(settings.objectIdA);
  bool objAIsArticulate = artObjAIter != existingArticulatedObjects_.end();

  auto rigidObjBIter = existingObjects_.find(settings.objectIdB);
  bool objBIsRigid = rigidObjBIter != existingObjects_.end();
  auto artObjBIter = existingArticulatedObjects_.find(settings.objectIdB);
  bool objBIsArticulate = artObjBIter != existingArticulatedObjects_.end();

  ESP_CHECK(objAIsRigid || objAIsArticulate,
            "::createRigidConstraint - Must provide a valid id for objectA");

  // cache the settings
  rigidConstraintSettings_.emplace(nextConstraintId_, settings);

  // setup body B in advance of bifurcation if necessary
  btRigidBody* rbB = nullptr;
  if (objBIsRigid) {
    rbB = static_cast<BulletRigidObject*>(rigidObjBIter->second.get())
              ->bObjectRigidBody_.get();
    rbB->setActivationState(DISABLE_DEACTIVATION);
  }

  // construct the constraints
  if (objAIsArticulate) {
    btMultiBody* mbA =
        static_cast<BulletArticulatedObject*>(artObjAIter->second.get())
            ->btMultiBody_.get();
    ESP_CHECK(mbA->getNumLinks() > settings.linkIdA,
              "::createRigidConstraint - linkA("
                  << settings.linkIdA
                  << ") is invalid for ArticulatedObject with"
                  << mbA->getNumLinks() << " links.");

    btMultiBody* mbB = nullptr;
    if (objBIsArticulate) {
      mbB = static_cast<BulletArticulatedObject*>(artObjBIter->second.get())
                ->btMultiBody_.get();
      ESP_CHECK(mbA->getNumLinks() > settings.linkIdA,
                "::createRigidConstraint - linkB("
                    << settings.linkIdB
                    << ") is invalid for ArticulatedObject with"
                    << mbB->getNumLinks() << " links.");
      mbB->setCanSleep(false);
    }
    mbA->setCanSleep(false);

    // construct a multibody constraint
    if (settings.constraintType == RigidConstraintType::PointToPoint) {
      // point to point constraint
      std::unique_ptr<btMultiBodyPoint2Point> p2p;
      if (mbB != nullptr) {
        // AO <-> AO constraint
        p2p = std::make_unique<btMultiBodyPoint2Point>(
            mbA, settings.linkIdA, mbB, settings.linkIdB,
            btVector3(settings.pivotA), btVector3(settings.pivotB));
      } else {
        // rigid object or global constraint
        p2p = std::make_unique<btMultiBodyPoint2Point>(
            mbA, settings.linkIdA, rbB, btVector3(settings.pivotA),
            btVector3(settings.pivotB));
      }
      bWorld_->addMultiBodyConstraint(p2p.get());
      articulatedP2PConstraints_.emplace(nextConstraintId_, std::move(p2p));
    } else {
      // fixed constraint
      std::unique_ptr<btMultiBodyFixedConstraint> fixedConstraint;
      if (mbB != nullptr) {
        // AO <-> AO constraint
        fixedConstraint = std::make_unique<btMultiBodyFixedConstraint>(
            mbA, settings.linkIdA, mbB, settings.linkIdB,
            btVector3(settings.pivotA), btVector3(settings.pivotB),
            btMatrix3x3(settings.frameA), btMatrix3x3(settings.frameB));
      } else {
        // rigid object or global constraint
        fixedConstraint = std::make_unique<btMultiBodyFixedConstraint>(
            mbA, settings.linkIdA, rbB, btVector3(settings.pivotA),
            btVector3(settings.pivotB), btMatrix3x3(settings.frameA),
            btMatrix3x3(settings.frameB));
      }
      bWorld_->addMultiBodyConstraint(fixedConstraint.get());
      articulatedFixedConstraints_.emplace(nextConstraintId_,
                                           std::move(fixedConstraint));
    }
  } else {
    ESP_CHECK(
        !objBIsArticulate,
        "::createRigidConstraint - objectA must be the ArticulatedObject for "
        "mixed typed constraints. Switch your Ids to resolve this issue.");
    btRigidBody* rbA = static_cast<BulletRigidObject*>(
                           existingObjects_.at(settings.objectIdA).get())
                           ->bObjectRigidBody_.get();
    rbA->setActivationState(DISABLE_DEACTIVATION);

    if (rbB == nullptr) {
      // use a dummy rigidbody with 0 mass to constrain to global frame.
      if (globalFrameObject == nullptr) {
        btRigidBody::btRigidBodyConstructionInfo info(0, nullptr, nullptr);
        globalFrameObject = std::make_unique<btRigidBody>(info);
      }
      rbB = globalFrameObject.get();
    }

    // construct a rigidbody constraint
    if (settings.constraintType == RigidConstraintType::PointToPoint) {
      // point to point
      std::unique_ptr<btPoint2PointConstraint> p2p =
          std::make_unique<btPoint2PointConstraint>(*rbA, *rbB,
                                                    btVector3(settings.pivotA),
                                                    btVector3(settings.pivotB));
      bWorld_->addConstraint(p2p.get());
      rigidP2PConstraints_.emplace(nextConstraintId_, std::move(p2p));
    } else {
      // fixed
      std::unique_ptr<btFixedConstraint> fixedConstraint =
          std::make_unique<btFixedConstraint>(
              *rbA, *rbB,
              btTransform(btMatrix3x3(settings.frameA),
                          btVector3(settings.pivotA)),
              btTransform(btMatrix3x3(settings.frameB),
                          btVector3(settings.pivotB)));
      bWorld_->addConstraint(fixedConstraint.get());
      rigidFixedConstraints_.emplace(nextConstraintId_,
                                     std::move(fixedConstraint));
    }
  }

  // link objects to their consraints for later deactivation/removal logic
  objectConstraints_[settings.objectIdA].push_back(nextConstraintId_);
  if (settings.objectIdB != ID_UNDEFINED) {
    objectConstraints_[settings.objectIdB].push_back(nextConstraintId_);
  }

  // use the updater to set params
  updateRigidConstraint(nextConstraintId_, settings);

  return nextConstraintId_++;
}

void BulletPhysicsManager::updateRigidConstraint(
    int constraintId,
    const RigidConstraintSettings& settings) {
  // validate that object and link ids are unchanged for update.
  auto rigidConstraintCacheIter = rigidConstraintSettings_.find(constraintId);
  ESP_CHECK(rigidConstraintCacheIter != rigidConstraintSettings_.end(),
            "::updateRigidConstraint - Provided invalid constraintId ="
                << constraintId);
  auto& cachedSettings = rigidConstraintCacheIter->second;
  ESP_CHECK(cachedSettings.objectIdA == settings.objectIdA,
            "::updateRigidConstraint - RigidConstraintSettings::objectIdA must "
            "match existing settings ("
                << settings.objectIdA << " vs." << cachedSettings.objectIdA
                << ")");
  ESP_CHECK(cachedSettings.objectIdB == settings.objectIdB,
            "::updateRigidConstraint - RigidConstraintSettings::objectIdB must "
            "match existing settings ("
                << settings.objectIdB << " vs." << cachedSettings.objectIdB
                << ")");
  ESP_CHECK(cachedSettings.linkIdA == settings.linkIdA,
            "::updateRigidConstraint - RigidConstraintSettings::linkIdA must "
            "match existing settings ("
                << settings.linkIdA << " vs." << cachedSettings.linkIdA << ")");
  ESP_CHECK(cachedSettings.linkIdB == settings.linkIdB,
            "::updateRigidConstraint - RigidConstraintSettings::linkIdB must "
            "match existing settings ("
                << settings.linkIdB << " vs." << cachedSettings.linkIdB << ")");
  ESP_CHECK(cachedSettings.constraintType == settings.constraintType,
            "::updateRigidConstraint - RigidConstraintSettings::constraintType "
            "must match existing settings ("
                << int(settings.constraintType) << " vs."
                << int(cachedSettings.constraintType) << ")");

  auto articulatedP2PConstraintIter =
      articulatedP2PConstraints_.find(constraintId);

  if (articulatedP2PConstraintIter != articulatedP2PConstraints_.end()) {
    // NOTE: oddly, pivotA cannot be set through the API for this constraint
    // type.
    ESP_CHECK(cachedSettings.pivotA == settings.pivotA,
              "::updateRigidConstraint - RigidConstraintSettings::pivotA must "
              "match existing settings for multibody P2P constraints. Instead, "
              "remove and create to update this parameter. ("
                  << settings.pivotA << " vs." << cachedSettings.pivotA << ")");
    // TODO: Either fix the Bullet API or do the add/remove for the user here.
    articulatedP2PConstraintIter->second->setPivotInB(
        btVector3(settings.pivotB));
    articulatedP2PConstraintIter->second->setMaxAppliedImpulse(
        settings.maxImpulse);
  } else {
    auto rigidP2PConstraintIter = rigidP2PConstraints_.find(constraintId);

    if (rigidP2PConstraintIter != rigidP2PConstraints_.end()) {
      rigidP2PConstraintIter->second->m_setting.m_impulseClamp =
          settings.maxImpulse;
      rigidP2PConstraintIter->second->setPivotA(btVector3(settings.pivotA));
      rigidP2PConstraintIter->second->setPivotB(btVector3(settings.pivotB));
    } else {
      auto articulatedFixedConstraintIter =
          articulatedFixedConstraints_.find(constraintId);

      if (articulatedFixedConstraintIter !=
          articulatedFixedConstraints_.end()) {
        articulatedFixedConstraintIter->second->setPivotInA(
            btVector3(settings.pivotA));
        articulatedFixedConstraintIter->second->setPivotInB(
            btVector3(settings.pivotB));
        articulatedFixedConstraintIter->second->setFrameInA(
            btMatrix3x3(settings.frameA));
        articulatedFixedConstraintIter->second->setFrameInB(
            btMatrix3x3(settings.frameB));
        articulatedFixedConstraintIter->second->setMaxAppliedImpulse(
            settings.maxImpulse);
      } else {
        auto rigidFixedConstraintIter =
            rigidFixedConstraints_.find(constraintId);
        if (rigidFixedConstraintIter != rigidFixedConstraints_.end()) {
          rigidFixedConstraintIter->second->setFrames(
              btTransform(btMatrix3x3(settings.frameA),
                          btVector3(settings.pivotA)),
              btTransform(btMatrix3x3(settings.frameB),
                          btVector3(settings.pivotB)));
          // NOTE: impulse is interpreted as force for this constraint.
          for (int i = 0; i < 6; ++i) {
            rigidFixedConstraintIter->second->setMaxMotorForce(
                i, settings.maxImpulse);
          }

        } else {
          // one of the maps should have the id if it passed the checks
          CORRADE_INTERNAL_ASSERT_UNREACHABLE();
        }
      }
    }
  }
  // cache the new settings
  rigidConstraintSettings_[constraintId] = settings;
}

void BulletPhysicsManager::removeRigidConstraint(int constraintId) {
  auto articulatedP2PConstraintIter =
      articulatedP2PConstraints_.find(constraintId);
  if (articulatedP2PConstraintIter != articulatedP2PConstraints_.end()) {
    bWorld_->removeMultiBodyConstraint(
        articulatedP2PConstraintIter->second.get());
    articulatedP2PConstraints_.erase(articulatedP2PConstraintIter);
  } else {
    auto rigidP2PConstraintIter = rigidP2PConstraints_.find(constraintId);
    if (rigidP2PConstraintIter != rigidP2PConstraints_.end()) {
      bWorld_->removeConstraint(rigidP2PConstraintIter->second.get());
      rigidP2PConstraints_.erase(rigidP2PConstraintIter);
    } else {
      auto articulatedFixedConstraintIter =
          articulatedFixedConstraints_.find(constraintId);
      if (articulatedFixedConstraintIter !=
          articulatedFixedConstraints_.end()) {
        bWorld_->removeMultiBodyConstraint(
            articulatedFixedConstraintIter->second.get());
        articulatedFixedConstraints_.erase(articulatedFixedConstraintIter);
      } else {
        auto rigidFixedConstraintIter =
            rigidFixedConstraints_.find(constraintId);
        if (rigidFixedConstraintIter != rigidFixedConstraints_.end()) {
          bWorld_->removeConstraint(rigidFixedConstraintIter->second.get());
          rigidFixedConstraints_.erase(rigidFixedConstraintIter);
        } else {
          ESP_ERROR() << "No constraint with constraintId =" << constraintId;
          return;
        }
      }
    }
  }
  rigidConstraintSettings_.erase(constraintId);
  // remove the constraint from any referencing object maps
  for (auto& itr : objectConstraints_) {
    auto conIdItr =
        std::find(itr.second.begin(), itr.second.end(), constraintId);
    if (conIdItr != itr.second.end()) {
      itr.second.erase(conIdItr);
      // when no constraints active for the object, allow it to sleep again
      if (itr.second.empty()) {
        auto artObjIter = existingArticulatedObjects_.find(itr.first);
        if (artObjIter != existingArticulatedObjects_.end()) {
          btMultiBody* mb =
              static_cast<BulletArticulatedObject*>(artObjIter->second.get())
                  ->btMultiBody_.get();
          mb->setCanSleep(true);
        } else {
          auto rigidObjIter = existingObjects_.find(itr.first);
          if (rigidObjIter != existingObjects_.end()) {
            btRigidBody* rb =
                static_cast<BulletRigidObject*>(rigidObjIter->second.get())
                    ->bObjectRigidBody_.get();
            rb->forceActivationState(ACTIVE_TAG);
            rb->activate(true);
          }
        }
      }
    }
  }
}

}  // namespace physics
}  // namespace esp
