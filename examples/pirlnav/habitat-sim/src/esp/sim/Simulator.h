// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ESP_SIM_SIMULATOR_H_
#define ESP_SIM_SIMULATOR_H_

#include <Corrade/Utility/Assert.h>

#include <utility>
#include "esp/agent/Agent.h"
#include "esp/assets/ResourceManager.h"
#include "esp/core/Esp.h"
#include "esp/core/Random.h"
#include "esp/gfx/DebugLineRender.h"
#include "esp/gfx/RenderTarget.h"
#include "esp/gfx/WindowlessContext.h"
#include "esp/metadata/MetadataMediator.h"
#include "esp/nav/PathFinder.h"
#include "esp/physics/ArticulatedObject.h"
#include "esp/physics/PhysicsManager.h"
#include "esp/physics/RigidObject.h"
#include "esp/physics/objectManagers/ArticulatedObjectManager.h"
#include "esp/physics/objectManagers/RigidObjectManager.h"
#include "esp/scene/SceneManager.h"
#include "esp/scene/SceneNode.h"
#include "esp/sensor/Sensor.h"

#include "SimulatorConfiguration.h"

namespace esp {
namespace nav {
class PathFinder;
struct NavMeshSettings;
class ActionSpacePathFinder;
}  // namespace nav
namespace scene {
class SemanticScene;
}  // namespace scene
namespace gfx {
class Renderer;
namespace replay {
class ReplayManager;
}  // namespace replay
}  // namespace gfx
}  // namespace esp

namespace esp {
namespace sim {
class Simulator {
 public:
  explicit Simulator(
      const SimulatorConfiguration& cfg,
      metadata::MetadataMediator::ptr _metadataMediator = nullptr);
  virtual ~Simulator();

  /**
   * @brief Closes the simulator and frees all loaded assets and GPU contexts.
   *
   * @warning Must reset the simulator to its "just after constructor" state for
   * python inheritance to function correctly.  Shared/unique pointers should be
   * set back to nullptr, any members set to their default values, etc.  If this
   * is not done correctly, the pattern for @ref `close` then @ref `reconfigure`
   * to create a "fresh" instance of the simulator may not work correctly
   *
   * @param destroy When set, destroy the background rendering thread and gl
   * context also. Otherwise these persist if the background rendering thread
   * was used. This is done because the OpenGL driver leaks memory on thread
   * destroy on some systems.
   */
  void close(bool destroy = true);

  void reconfigure(const SimulatorConfiguration& cfg);

  void reset();

  void seed(uint32_t newSeed);

  std::shared_ptr<gfx::Renderer> getRenderer() { return renderer_; }
  std::shared_ptr<scene::SemanticScene> getSemanticScene() {
    return resourceManager_->getSemanticScene();
  }

  /**
   * @brief Return a view of the currently set Semantic scene colormap.
   */
  const std::vector<Mn::Vector3ub>& getSemanticSceneColormap() const {
    return resourceManager_->getSemanticSceneColormap();
  }

  inline void getRenderGLContext() {
    // acquire GL context from background thread, if background rendering
    // enabled.
    if (renderer_) {
      renderer_->acquireGlContext();
    }
  }

  /** @brief check if the semantic scene exists.*/
  bool semanticSceneExists() const {
    return resourceManager_->semanticSceneExists();
  }

  /**
   * @brief get the current active scene graph
   */
  scene::SceneGraph& getActiveSceneGraph() {
    CORRADE_INTERNAL_ASSERT(std::size_t(activeSceneID_) < sceneID_.size());
    return sceneManager_->getSceneGraph(activeSceneID_);
  }

  /** @brief Check to see if there is a SemanticSceneGraph for rendering */
  bool semanticSceneGraphExists() const {
    return std::size_t(activeSemanticSceneID_) < sceneID_.size();
  }

  /** @brief get the semantic scene's SceneGraph for rendering */
  scene::SceneGraph& getActiveSemanticSceneGraph() {
    CORRADE_INTERNAL_ASSERT(semanticSceneGraphExists());
    return sceneManager_->getSceneGraph(activeSemanticSceneID_);
  }
  std::shared_ptr<gfx::replay::ReplayManager> getGfxReplayManager() {
    return gfxReplayMgr_;
  }

  void saveFrame(const std::string& filename);

  /**
   * @brief The ID of the CUDA device of the OpenGL context owned by the
   * simulator.  This will only be nonzero if the simulator is built in
   * --headless mode on linux
   */
  int gpuDevice() const {
    CORRADE_ASSERT(config_.createRenderer,
                   "Simulator::gpuDevice() : cannot get gpu device when "
                   "createRenderer flag is false",
                   0);
    if (context_ == nullptr) {
      return 0;
    }
    return context_->gpuDevice();
  }

  // === Physics Simulator Functions ===
  // TODO: support multi-scene physics (default sceneID=0 currently).

  /**
   * @brief Return manager for construction and access to asset attributes for
   * the current dataset.
   */
  const metadata::managers::AssetAttributesManager::ptr&
  getAssetAttributesManager() const {
    return metadataMediator_->getAssetAttributesManager();
  }
  /**
   * @brief Return manager for construction and access to light attributes and
   * layouts for the current dataset.
   */
  const metadata::managers::LightLayoutAttributesManager::ptr&
  getLightLayoutAttributesManager() const {
    return metadataMediator_->getLightLayoutAttributesManager();
  }

  /**
   * @brief Return manager for construction and access to object attributes and
   * layouts for the current dataset.
   */
  const metadata::managers::ObjectAttributesManager::ptr&
  getObjectAttributesManager() const {
    return metadataMediator_->getObjectAttributesManager();
  }
  /**
   * @brief Return manager for construction and access to physics world
   * attributes.
   */
  const metadata::managers::PhysicsAttributesManager::ptr&
  getPhysicsAttributesManager() const {
    return metadataMediator_->getPhysicsAttributesManager();
  }
  /**
   * @brief Return manager for construction and access to scene attributes for
   * the current dataset.
   */
  const metadata::managers::StageAttributesManager::ptr&
  getStageAttributesManager() const {
    return metadataMediator_->getStageAttributesManager();
  }

  /**
   * @brief Get current active dataset name from @ref metadataMediator_.
   */
  std::string getActiveSceneDatasetName() {
    return metadataMediator_->getActiveSceneDatasetName();
  }

  /**
   * @brief Set current active dataset name from @ref metadataMediator_.
   * @param _dsHandle The desired dataset to switch to. If has not been loaded,
   * an attempt will be made to load it.
   */
  void setActiveSceneDatasetName(const std::string& _dsHandle) {
    metadataMediator_->setActiveSceneDatasetName(_dsHandle);
  }

  /** @brief Return the library implementation type for the simulator currently
   * in use. Use to check for a particular implementation.
   * @return The implementation type of this simulator.
   */
  const esp::physics::PhysicsManager::PhysicsSimulationLibrary&
  getPhysicsSimulationLibrary() const {
    return physicsManager_->getPhysicsSimulationLibrary();
  }

  /** @brief Render any debugging visualizations provided by the underlying
   * physics simulator implementation. By default does nothing. See @ref
   * BulletPhysicsManager::debugDraw.
   * @param projTrans The composed projection and transformation matrix for the
   * render camera.
   */
  void physicsDebugDraw(const Magnum::Matrix4& projTrans) const {
    physicsManager_->debugDraw(projTrans);
  }

  /**
   * @brief Get a copy of a stage's template when the stage was instanced.
   *
   * Use this to query the stage's properties when it was initialized.
   */
  metadata::attributes::StageAttributes::cptr getStageInitializationTemplate(
      int sceneID = 0) const {
    if (sceneHasPhysics(sceneID)) {
      return physicsManager_->getStageInitAttributes();
    }
    return nullptr;
  }

  /**
   * @brief Build a map keyed by semantic color/id referencing a vector
   * connected component-based Semantic objects.
   */
  std::unordered_map<uint32_t, std::vector<scene::CCSemanticObject::ptr>>
  buildSemanticCCObjects() const {
    // build report with current stage attributes
    if (semanticSceneMeshLoaded_) {
      return resourceManager_->buildSemanticCCObjects(
          physicsManager_->getStageInitAttributes());
    }
    ESP_WARNING()
        << "Unable to build semantic CC objects since no semantic mesh exists.";
    return {};
  }

  /**
   * @brief Build a report on vertex color mappings to semantic scene descriptor
   * object colors, to show whether any verts have unknown colors and whether
   * any semantic object colors are not present in the mesh.
   */
  std::vector<std::string> buildVertexColorMapReport() const {
    if (semanticSceneMeshLoaded_) {
      return resourceManager_->buildVertexColorMapReport(
          physicsManager_->getStageInitAttributes());
    }
    ESP_WARNING() << "Unable to build vertex-to-color mapping report since no "
                     "semantic mesh exists.";
    return {};
  }

  /**
   * @brief Builds a @ref esp::metadata::SceneInstanceAttributes describing the
   * current scene configuration, and saves it to a JSON file, using @p
   * saveFileName .
   * @param saveFilename The name to use to save the current scene instance.
   * @return whether successful or not.
   */
  bool saveCurrentSceneInstance(const std::string& saveFilename,
                                int sceneID = 0) const {
    if (sceneHasPhysics(sceneID)) {
      ESP_DEBUG() << "Attempting to save current scene layout as "
                     "SceneInstanceAttributes with filename :"
                  << saveFilename;
      return metadataMediator_->getSceneInstanceAttributesManager()
          ->saveManagedObjectToFile(buildCurrentStateSceneAttributes(),
                                    saveFilename);
    }
    return false;
  }  // saveCurrentSceneInstance

  /**
   * @brief Builds a @ref esp::metadata::SceneInstanceAttributes describing
   * the current scene configuration, and saves it to a JSON file, using the
   * current scene attributes' filename, or an incremented version if @p
   * overwrite == false.
   * @param overwrite Whether to overrite an existing file with the same name,
   * should one exist.
   * @return whether successful or not.
   */
  bool saveCurrentSceneInstance(bool overwrite = false, int sceneID = 0) const {
    if (sceneHasPhysics(sceneID)) {
      ESP_DEBUG() << "Attempting to save current scene layout as "
                     "SceneInstanceAttributes.";
      return metadataMediator_->getSceneInstanceAttributesManager()
          ->saveManagedObjectToFile(buildCurrentStateSceneAttributes(),
                                    overwrite);
    }
    return false;
  }  // saveCurrentSceneInstance

  /**
   * @brief Get the IDs of the physics objects instanced in a physical scene.
   * See @ref esp::physics::PhysicsManager::getExistingObjectIDs.
   * @param sceneID !! Not used currently !! Specifies which physical scene to
   * query.
   * @return A vector of ID keys into @ref
   * esp::physics::PhysicsManager::existingObjects_.
   */
  std::vector<int> getExistingObjectIDs(int sceneID = 0) {
    if (sceneHasPhysics(sceneID)) {
      return physicsManager_->getExistingObjectIDs();
    }
    return std::vector<int>();  // empty if no simulator exists
  }

  /**
   * @brief Turn on/off rendering for the bounding box of the object's visual
   * component.
   *
   * Assumes the new @ref esp::gfx::Drawable for the bounding box should be
   * added to the active @ref esp::gfx::SceneGraph's default drawable group. See
   * @ref esp::gfx::SceneGraph::getDrawableGroup().
   *
   * @param drawBB Whether or not the render the bounding box.
   * @param objectId The object ID and key identifying the object in @ref
   * esp::physics::PhysicsManager::existingObjects_.
   * @param sceneID !! Not used currently !! Specifies which physical scene of
   * the object.
   */
  void setObjectBBDraw(bool drawBB, int objectId, int sceneID = 0) {
    if (sceneHasPhysics(sceneID)) {
      getRenderGLContext();
      auto& drawables = getDrawableGroup(sceneID);
      physicsManager_->setObjectBBDraw(objectId, &drawables, drawBB);
    }
  }

  //===============================================================================//
  // Voxel Field API

#ifdef ESP_BUILD_WITH_VHACD
  /**
   * @brief Creates a voxelization for a particular object. Initializes the
   * voxelization with a boundary voxel grid using VHACD's voxelization library.
   *
   * @param objectId The object ID and key identifying the object in @ref
   * esp::physics::PhysicsManager::existingObjects_.
   * @param resolution The approximate number of voxels for the voxel grid that
   * is created.
   */
  void createObjectVoxelization(int objectId, int resolution = 1000000) {
    physicsManager_->generateVoxelization(objectId, resolution);
  }
#endif

  /**
   * @brief Turn on/off rendering for the voxel grid of the object's visual
   * component.
   *
   * If a voxel grid for the object has not been created, it will make one with
   * default arguments using @ref createObjectVoxelization().
   *
   * @param drawV Whether or not the render the voxel grid.
   * @param objectId The object ID and key identifying the object in @ref
   * esp::physics::PhysicsManager::existingObjects_.
   * @param gridName The name of the voxel grid to be visualized.
   */
  void setObjectVoxelizationDraw(bool drawV,
                                 int objectId,
                                 const std::string& gridName = "Boundary") {
    auto& drawables = getDrawableGroup();
    physicsManager_->setObjectVoxelizationDraw(objectId, gridName, &drawables,
                                               drawV);
  }

  /**
   * @brief Returns the VoxelWrapper for a particular object.
   *
   * @param objectId The object ID and key identifying the object in @ref
   * esp::physics::PhysicsManager::existingObjects_.
   * @return A shared ptr to the object's VoxelWrapper .
   */
  std::shared_ptr<esp::geo::VoxelWrapper> getObjectVoxelization(int objectId) {
    return physicsManager_->getObjectVoxelization(objectId);
  }

#ifdef ESP_BUILD_WITH_VHACD
  /**
   * @brief Creates a voxelization for the scene. Initializes the voxelization
   * with a boundary voxel grid using VHACD's voxelization library.
   *
   * @param resolution The approximate number of voxels for the voxel grid that
   * is created.
   */
  void createStageVoxelization(int resolution = 1000000) {
    physicsManager_->generateStageVoxelization(resolution);
  }
#endif

  /**
   * @brief Turn on/off rendering for the voxel grid of the scene's visual
   * component.
   *
   * If a voxel grid for the scene has not been created, it will make one with
   * default arguments using @ref createStageVoxelization().
   *
   * @param drawV Whether or not the render the voxel grid.
   * @param gridName The name of the voxel grid to be visualized.
   */
  void setStageVoxelizationDraw(bool drawV,
                                const std::string& gridName = "Boundary") {
    auto& drawables = getDrawableGroup();
    physicsManager_->setStageVoxelizationDraw(gridName, &drawables, drawV);
  }

  /**
   * @brief Returns the VoxelWrapper for a particular object.
   * @return A shared ptr to the object's VoxelWrapper .
   */
  std::shared_ptr<esp::geo::VoxelWrapper> getStageVoxelization() {
    return physicsManager_->getStageVoxelization();
  }

  /**
   * @brief Discrete collision check for contact between an object and the
   * collision world.
   * @param objectId The object ID and key identifying the object in @ref
   * esp::physics::PhysicsManager::existingObjects_.
   * @param sceneID !! Not used currently !! Specifies which physical scene of
   * the object.
   * @return Whether or not the object is in contact with any other collision
   * enabled objects.
   */
  bool contactTest(int objectID, int sceneID = 0) {
    if (sceneHasPhysics(sceneID)) {
      return physicsManager_->contactTest(objectID);
    }
    return false;
  }

  /**
   * @brief Perform discrete collision detection for the scene.
   */
  void performDiscreteCollisionDetection() {
    physicsManager_->performDiscreteCollisionDetection();
  };

  /**
   * @brief Query physics simulation implementation for contact point data from
   * the most recent collision detection cache.
   *
   * @return a vector with each entry corresponding to a single contact point.
   */
  std::vector<esp::physics::ContactPointData> getPhysicsContactPoints() {
    return physicsManager_->getContactPoints();
  }

  /**
   * @brief Query the number of contact points that were active during the
   * collision detection check.
   *
   * @return the number of active contact points.
   */
  int getPhysicsNumActiveContactPoints() {
    return physicsManager_->getNumActiveContactPoints();
  }

  /**
   * @brief See BulletPhysicsManager.h getNumActiveOverlappingPairs
   */
  int getPhysicsNumActiveOverlappingPairs() {
    return physicsManager_->getNumActiveOverlappingPairs();
  }

  /**
   * @brief See BulletPhysicsManager.h getStepCollisionSummary
   */
  std::string getPhysicsStepCollisionSummary() {
    return physicsManager_->getStepCollisionSummary();
  }

  /**
   * @brief Set the stage to collidable or not.
   */
  void setStageIsCollidable(bool collidable) {
    if (sceneHasPhysics(activeSceneID_)) {
      physicsManager_->setStageIsCollidable(collidable);
    }
  }

  /**
   * @brief Get whether or not the stage is collision active.
   */
  bool getStageIsCollidable() {
    if (sceneHasPhysics(activeSceneID_)) {
      return physicsManager_->getStageIsCollidable();
    }
    return false;
  }

  /**
   * @brief returns the wrapper manager for the currently created rigid objects.
   * @return RigidObject wrapper manager.
   */
  std::shared_ptr<esp::physics::RigidObjectManager> getRigidObjectManager()
      const {
    if (sceneHasPhysics(activeSceneID_)) {
      return physicsManager_->getRigidObjectManager();
    }
    return nullptr;
  }  // getRigidObjectManager

  /**
   * @brief returns the wrapper manager for the currently created articulated
   * objects.
   * @return ArticulatedObject wrapper manager
   */
  std::shared_ptr<esp::physics::ArticulatedObjectManager>
  getArticulatedObjectManager() const {
    if (sceneHasPhysics(activeSceneID_)) {
      return physicsManager_->getArticulatedObjectManager();
    }
    return nullptr;
  }

  /**
   * @brief Raycast into the collision world of a scene.
   *
   * Note: A default @ref physics::PhysicsManager has no collision world, so
   * physics must be enabled for this feature.
   *
   * @param ray The ray to cast. Need not be unit length, but returned hit
   * distances will be in units of ray length.
   * @param maxDistance The maximum distance along the ray direction to search.
   * In units of ray length.
   * @param sceneID !! Not used currently !! Specifies which physical scene of
   * the object.
   * @return Raycast results sorted by distance.
   */
  esp::physics::RaycastResults castRay(const esp::geo::Ray& ray,
                                       double maxDistance = 100.0,
                                       int sceneID = 0) {
    if (sceneHasPhysics(sceneID)) {
      return physicsManager_->castRay(ray, maxDistance);
    }
    return esp::physics::RaycastResults();
  }

  /**
   * @brief the physical world has a notion of time which passes during
   * animation/simulation/action/etc... Step the physical world forward in time
   * by a desired duration. Note that the actual duration of time passed by this
   * step will depend on simulation time stepping mode (todo). See @ref
   * esp::physics::PhysicsManager::stepPhysics.
   * @todo timestepping options?
   * @param dt The desired amount of time to advance the physical world.
   * @return The new world time after stepping. See @ref
   * esp::physics::PhysicsManager::worldTime_.
   */
  double stepWorld(double dt = 1.0 / 60.0);

  /**
   * @brief Get the current time in the simulated world. This is always 0 if no
   * @ref esp::physics::PhysicsManager is initialized. See @ref stepWorld. See
   * @ref esp::physics::PhysicsManager::getWorldTime.
   * @return The amount of time, @ref esp::physics::PhysicsManager::worldTime_,
   * by which the physical world has advanced.
   */
  double getWorldTime();

  /**
   * @brief Get the last physics timestep in seconds
   *
   * @return The timestep.
   */
  double getPhysicsTimeStep();

  /**
   * @brief Get the simplified name of the @ref
   * esp::metadata::attributes::SceneInstanceAttributes used to create the scene
   * currently being simulated/displayed.
   */
  std::string getCurSceneInstanceName() const {
    if (curSceneInstanceAttributes_ == nullptr) {
      return "NONE";
    }
    return curSceneInstanceAttributes_->getSimplifiedHandle();
  }

  /**
   * @brief Set the gravity in a physical scene.
   */
  void setGravity(const Magnum::Vector3& gravity, int sceneID = 0) {
    if (sceneHasPhysics(sceneID)) {
      physicsManager_->setGravity(gravity);
    }
  }

  /**
   * @brief Get the gravity in a physical scene.
   */
  Magnum::Vector3 getGravity(int sceneID = 0) const {
    if (sceneHasPhysics(sceneID)) {
      return physicsManager_->getGravity();
    }
    return Magnum::Vector3();
  }

  std::shared_ptr<esp::gfx::DebugLineRender> getDebugLineRender() {
    // We only create this if/when used (lazy creation)
    if (!debugLineRender_) {
      debugLineRender_ = std::make_shared<esp::gfx::DebugLineRender>();
    }
    return debugLineRender_;
  }

  /**
   * @brief Compute the navmesh for the simulator's current active scene and
   * assign it to the referenced @ref nav::PathFinder.
   * @param pathfinder The pathfinder object to which the recomputed navmesh
   * will be assigned.
   * @param navMeshSettings The @ref nav::NavMeshSettings instance to
   * parameterize the navmesh construction.
   * @return Whether or not the navmesh recomputation succeeded.
   */
  bool recomputeNavMesh(nav::PathFinder& pathfinder,
                        const nav::NavMeshSettings& navMeshSettings,
                        bool includeStaticObjects = false);

  /**
   * @brief Get the joined mesh data for all objects in the scene
   * @param includeStaticObjects flag to include static objects
   * @return A shared ptr assets::MeshData with required mesh
   */
  assets::MeshData::ptr getJoinedMesh(bool includeStaticObjects = false);

  /**
   * @brief Get the joined semantic mesh data for all objects in the scene
   * @param[out] objectIds will be populated with the object ids for the
   * semantic mesh
   * @return A shared ptr assets::MeshData with required mesh and populate the
   * objectIds
   */
  assets::MeshData::ptr getJoinedSemanticMesh(
      std::vector<std::uint16_t>& objectIds);

  /**
   * @brief Set visualization of the current NavMesh @ref pathfinder_ on or off.
   *
   * @param visualize Whether or not to visualize the navmesh.
   * @return Whether or not the NavMesh visualization is active.
   */
  bool setNavMeshVisualization(bool visualize);

  /**
   * @brief Query active state of the current NavMesh @ref pathfinder_
   * visualization.
   */
  bool isNavMeshVisualizationActive();

  /**
   * @brief Return a ref to a new drawables in the currently active scene, for
   * object creation.  Eventually support multi-scene ID
   * @param sceneID The scene to get the drawables for.  Currently not used.
   */
  inline esp::gfx::DrawableGroup& getDrawableGroup(
      CORRADE_UNUSED const int sceneID) {
    // TODO eventually use passed sceneID
    return sceneManager_->getSceneGraph(activeSceneID_).getDrawables();
  }

  /**
   * @brief Return a ref to a new drawables in the currently active scene, for
   * object creation.
   */
  inline esp::gfx::DrawableGroup& getDrawableGroup() {
    return getDrawableGroup(activeSceneID_);
  }

  /**
   * @brief Compute a trajectory visualization for the passed points.
   * @param trajVisName The name to use for the trajectory visualization
   * @param pts The points of a trajectory, in order
   * @param colorVec Array of colors for trajectory tube.
   * @param numSegments The number of the segments around the circumference of
   * the tube. Must be greater than or equal to 3.
   * @param radius The radius of the tube.
   * @param smooth Whether to smooth the points in the trajectory or not. Will
   * build a much bigger mesh
   * @param numInterp The number of interpolations between each trajectory
   * point, if smoothed
   * @return The ID of the object created for the visualization
   */
  int addTrajectoryObject(const std::string& trajVisName,
                          const std::vector<Mn::Vector3>& pts,
                          const std::vector<Mn::Color3>& colorVec,
                          int numSegments = 3,
                          float radius = .001,
                          bool smooth = false,
                          int numInterp = 10) {
    if (sceneHasPhysics(activeSceneID_)) {
      return physicsManager_->addTrajectoryObject(
          trajVisName, pts, colorVec, numSegments, radius, smooth, numInterp);
    }
    return ID_UNDEFINED;
  }
  /**
   * @brief Remove a trajectory visualization by name.
   * @param trajVisName The name of the trajectory visualization to remove.
   * @return whether successful or not.
   */
  bool removeTrajVisByName(const std::string& trajVisName) {
    if (sceneHasPhysics(activeSceneID_)) {
      return physicsManager_->removeTrajVisByName(trajVisName);
    }
    return false;
  }

  /**
   * @brief Remove a trajectory visualization by object ID.
   * @param trajVisObjID The object ID of the trajectory visualization to
   * remove.
   * @return whether successful or not.
   */
  bool removeTrajVisByID(int trajVisObjID) {
    if (sceneHasPhysics(activeSceneID_)) {
      return physicsManager_->removeTrajVisByID(trajVisObjID);
    }
    return false;
  }

  agent::Agent::ptr getAgent(int agentId);

  agent::Agent::ptr addAgent(const agent::AgentConfiguration& agentConfig,
                             scene::SceneNode& agentParentNode);
  agent::Agent::ptr addAgent(const agent::AgentConfiguration& agentConfig);

  /**
   * @brief Initialize sensor and attach to sceneNode of a particular object
   * @param objectId    Id of the object to which a sensor will be initialized
   * at its node
   * @param sensorSpec  SensorSpec of sensor to be initialized
   * @return            handle to sensor initialized
   *
   */
  esp::sensor::Sensor& addSensorToObject(
      int objectId,
      const esp::sensor::SensorSpec::ptr& sensorSpec);

  /**
   * @brief Displays observations on default frame buffer for a
   * particular sensor of an agent
   * @param agentId    Id of the agent for which the observation is to
   *                   be returned
   * @param sensorId   Id of the sensor for which the observation is to
   *                   be returned
   */
  bool displayObservation(int agentId, const std::string& sensorId);

  /**
   * @brief
   * get the render target of a particular sensor of an agent
   * @return pointer to the render target if it is a valid visual sensor,
   * otherwise nullptr
   * @param agentId    Id of the agent for which the observation is to
   *                   be returned
   * @param sensorId   Id of the sensor for which the observation is to
   *                   be returned
   */
  gfx::RenderTarget* getRenderTarget(int agentId, const std::string& sensorId);

  /**
   * @brief draw observations to the frame buffer stored in that
   * particular sensor of an agent. Unlike the @displayObservation, it will
   * not display the observation on the default frame buffer
   * @param agentId    Id of the agent for which the observation is to
   *                   be returned
   * @param sensorId   Id of the sensor for which the observation is to
   *                   be returned
   */
  bool drawObservation(int agentId, const std::string& sensorId);

  /**
   * @brief visualize the undisplayable observations such as depth, semantic, to
   * the frame buffer stored in the @ref SensorInfoVisualizer
   * Note: it will not display the observation on the default frame buffer
   * @param[in] agentId    Id of the agent for which the observation is to
   *                   be returned
   * @param[in] sensorId   Id of the sensor for which the observation is to
   *                   be returned
   * @param[in] colorMapOffset the offset of the color map
   * @param[in] colorMapScale the scale of the color map
   * See details in @ref TextureVisualizerShader::setColorMapTransformation for
   * more info.
   *
   * NOTE: it assumes:
   * -) it is a non-rgb sensor (such as a depth or semantic sensor);
   * -) the drawObservation is called;
   * -) when the render target is bound to the sensor, "VisualizeTexture" is
   * enabled. See @ref Renderer::bindRenderTarget and @ref
   * Renderer::RenderTargetBindingFlag for more info
   * @return false if the sensor's observation cannot be visualized.
   */
  bool visualizeObservation(int agentId,
                            const std::string& sensorId,
                            float colorMapOffset,
                            float colorMapScale);

  bool visualizeObservation(int agentId, const std::string& sensorId);

  bool getAgentObservation(int agentId,
                           const std::string& sensorId,
                           sensor::Observation& observation);
  int getAgentObservations(
      int agentId,
      std::map<std::string, sensor::Observation>& observations);

  bool getAgentObservationSpace(int agentId,
                                const std::string& sensorId,
                                sensor::ObservationSpace& space);
  int getAgentObservationSpaces(
      int agentId,
      std::map<std::string, sensor::ObservationSpace>& spaces);

  nav::PathFinder::ptr getPathFinder() { return pathfinder_; }
  void setPathFinder(nav::PathFinder::ptr pf);

  /**
   * @brief Enable or disable frustum culling (enabled by default)
   * @param val true = enable, false = disable
   */
  void setFrustumCullingEnabled(bool val) { frustumCulling_ = val; }

  /**
   * @brief Get status, whether frustum culling is enabled or not
   * @return true if enabled, otherwise false
   */
  bool isFrustumCullingEnabled() const { return frustumCulling_; }

  /**
   * @brief Get a copy of an existing @ref gfx::LightSetup by its key.
   *
   * @param key The string key of the @ref gfx::LightSetup.
   */
  gfx::LightSetup getLightSetup(const std::string& key = DEFAULT_LIGHTING_KEY) {
    return *resourceManager_->getLightSetup(key);
  }

  /**
   * @brief Get a copy of the currently set existing @ref gfx::LightSetup.
   *
   * @param key The string key of the @ref gfx::LightSetup.
   */
  gfx::LightSetup getCurrentLightSetup() {
    return *resourceManager_->getLightSetup(config_.sceneLightSetupKey);
  }

  /**
   * @brief Register a @ref gfx::LightSetup with a key name.
   *
   * If this name already exists, the @ref gfx::LightSetup is updated and all
   * @ref esp::gfx::Drawable s using this setup are updated.
   *
   * @param lightSetup The @ref gfx::LightSetup this key will now reference.
   * @param key Key to identify this @ref gfx::LightSetup.
   */
  void setLightSetup(gfx::LightSetup lightSetup,
                     const std::string& key = DEFAULT_LIGHTING_KEY) {
    resourceManager_->setLightSetup(std::move(lightSetup), key);
  }

  //============= Object Rigid Constraint API =============

  /**
   * @brief Create a rigid constraint between two objects or an object and the
   * world.
   *
   * Note: requires Bullet physics to be enabled.
   *
   * @param settings The datastructure defining the constraint parameters.
   *
   * @return The id of the newly created constraint or ID_UNDEFINED if failed.
   */
  int createRigidConstraint(const physics::RigidConstraintSettings& settings) {
    return physicsManager_->createRigidConstraint(settings);
  }

  /**
   * @brief Update the settings of a rigid constraint.
   *
   * Note: requires Bullet physics to be enabled.
   *
   * @param constraintId The id of the constraint to update.
   * @param settings The new settings of the constraint.
   */
  void updateRigidConstraint(int constraintId,
                             const physics::RigidConstraintSettings& settings) {
    physicsManager_->updateRigidConstraint(constraintId, settings);
  }

  /**
   * @brief Remove a rigid constraint by id.
   *
   * Note: requires Bullet physics to be enabled.
   *
   * @param constraintId The id of the constraint to remove.
   */
  void removeRigidConstraint(int constraintId) {
    physicsManager_->removeRigidConstraint(constraintId);
  }

  /**
   * @brief Get a copy of the settings for an existing rigid constraint.
   *
   * Note: requires Bullet physics to be enabled.
   *
   * @param constraintId The id of the constraint.
   *
   * @return The settings of the constraint.
   */
  physics::RigidConstraintSettings getRigidConstraintSettings(
      int constraintId) const {
    return physicsManager_->getRigidConstraintSettings(constraintId);
  }

  //============= END - Object Rigid Constraint API =============

  /**
   * @brief Getter for PRNG.
   *
   * Use this where-ever possible so that habitat won't be effected by
   * python random or numpy.random modules.
   */
  core::Random::ptr random() { return random_; }

  /**
   * @brief Get this simulator's MetadataMediator
   */
  metadata::MetadataMediator::ptr getMetadataMediator() const {
    return metadataMediator_;
  }

  /**
   * @brief Set this simulator's MetadataMediator
   */
  void setMetadataMediator(metadata::MetadataMediator::ptr _metadataMediator) {
    metadataMediator_ = std::move(_metadataMediator);
    // set newly added MM to have current Simulator Config
    metadataMediator_->setSimulatorConfiguration(this->config_);
  }

  /**
   * @brief Load and add a render asset instance to the current scene graph(s).
   * @param assetInfo the asset to load
   * @param creation how to create the instance
   */
  scene::SceneNode* loadAndCreateRenderAssetInstance(
      const assets::AssetInfo& assetInfo,
      const assets::RenderAssetInstanceCreationInfo& creation);

#ifdef ESP_BUILD_WITH_VHACD
  /**
   * @brief Runs convex hull decomposition on a specified file. Creates an
   * object attributes referencing a newly created convex hull asset, and
   * returns the attribute's handle.
   *
   * @param filename The MeshMetaData filename to be converted.
   * @param params VHACD params that specify resolution, vertices per convex
   * hull, etc.
   * @param renderChd Specifies whether or not to render the coinvex hull asset,
   * or to render the original render asset.
   * @param saveChdToObj Specifies whether or not to save the newly created
   * convex hull asset to an obj file.
   * @return The handle of the newly created object attributes.
   */
  std::string convexHullDecomposition(
      const std::string& filename,
      const assets::ResourceManager::VHACDParameters& params =
          assets::ResourceManager::VHACDParameters(),
      bool renderChd = false,
      bool saveChdToObj = false);
#endif

  /**
   * @brief For the current active scene, update the shaow map drawable group
   */
  void updateShadowMapDrawableGroup();
  /**
   * @brief compute the shadow maps for the active scene graph, and store the
   * results in the resource manager
   */
  void computeShadowMaps(float lightNearPlane, float lightFarPlane);
  /**
   * @brief propergate shadow maps to the drawables
   * NOTE: so far only pbr drawable and shader support the shadow maps
   */
  void setShadowMapsToDrawables();

 protected:
  Simulator() = default;

  /**
   * @brief if Navemesh visualization is active, reset the visualization.
   */
  void resetNavMeshVisIfActive() {
    if (isNavMeshVisualizationActive()) {
      // if updating pathfinder_ instance, refresh the visualization.
      setNavMeshVisualization(false);  // first clear the old instance
      setNavMeshVisualization(true);
    }
  }

  /**
   * @brief Builds a scene instance and populates it with initial object
   * layout, if appropriate, based on @ref
   * esp::metadata::attributes::SceneInstanceAttributes referenced by @p
   * activeSceneName .
   * @param activeSceneName The name of the desired SceneInstanceAttributes to
   * use to instantiate a scene.
   * @return Whether successful or not.
   */
  bool createSceneInstance(const std::string& activeSceneName);

  /**
   * @brief Instance the stage for the current scene based on
   * curSceneInstanceAttributes_, the currently active scene's @ref
   * esp::metadata::SceneInstanceAttributes
   * @param curSceneInstanceAttributes The attributes describing the current
   * scene instance.
   * @return whether stage creation is completed successfully
   */
  bool instanceStageForSceneAttributes(
      const metadata::attributes::SceneInstanceAttributes::cptr&
          curSceneInstanceAttributes);

  /**
   * @brief Instance all the objects in the scene based on
   * curSceneInstanceAttributes_, the currently active scene's @ref
   * esp::metadata::SceneInstanceAttributes.
   * @param curSceneInstanceAttributes The attributes describing the current
   * scene instance.
   * @return whether object creation and placement is completed successfully
   */
  bool instanceObjectsForSceneAttributes(
      const metadata::attributes::SceneInstanceAttributes::cptr&
          curSceneInstanceAttributes);

  /**
   * @brief Instance all the articulated objects in the scene based
   * oncurSceneInstanceAttributes_, the currently active scene's @ref
   * esp::metadata::SceneInstanceAttributes.
   * @param curSceneInstanceAttributes The attributes describing the current
   * scene instance.
   * @return whether articulated object creation and placement is completed
   * successfully
   */
  bool instanceArticulatedObjectsForSceneAttributes(
      const metadata::attributes::SceneInstanceAttributes::cptr&
          curSceneInstanceAttributes);

  /**
   * @brief Build a @ref metadata::attributes::SceneInstanceAttributes
   * describing the current scene's present state.
   */
  metadata::attributes::SceneInstanceAttributes::ptr
  buildCurrentStateSceneAttributes() const;

  /**
   * @brief sample a random valid AgentState in passed agentState
   * @param agentState [out] The placeholder for the sampled agent state.
   */
  void sampleRandomAgentState(agent::AgentState& agentState);

  bool isValidScene(int sceneID) const {
    return sceneID >= 0 && static_cast<std::size_t>(sceneID) < sceneID_.size();
  }

  bool sceneHasPhysics(int sceneID) const {
    return isValidScene(sceneID) && physicsManager_ != nullptr;
  }

  /**
   * @brief TEMPORARY until sim access to objects is completely removed.  This
   * method will return an object's wrapper if the passsed @p sceneID and @p
   * objID are both valid.  This wrapper will then be used by the calling
   * function to access components of the object.
   * @param sceneID The ID of the scene to query
   * @param objID The ID of the desired object
   * @return A smart pointer to the wrapper referencing the desired object, or
   * nullptr if DNE.
   */
  esp::physics::ManagedRigidObject::ptr queryRigidObjWrapper(int sceneID,
                                                             int objID) const {
    if (!sceneHasPhysics(sceneID)) {
      return nullptr;
    }
    return getRigidObjectManager()->getObjectCopyByID(objID);
  }

  /**
   * @brief TEMPORARY until sim access to objects is completely removed.  This
   * method will return an object's wrapper if the passsed @p sceneID and @p
   * objID are both valid.  This wrapper will then be used by the calling
   * function to access components of the object.
   * @param sceneID The ID of the scene to query
   * @param objID The ID of the desired object
   * @return A smart pointer to the wrapper referencing the desired object, or
   * nullptr if DNE.
   */
  esp::physics::ManagedArticulatedObject::ptr queryArticulatedObjWrapper(
      int sceneID,
      int objID) const {
    if (!sceneHasPhysics(sceneID)) {
      return nullptr;
    }
    return getArticulatedObjectManager()->getObjectCopyByID(objID);
  }

  void reconfigureReplayManager(bool enableGfxReplaySave);

  gfx::WindowlessContext::uptr context_ = nullptr;
  std::shared_ptr<gfx::Renderer> renderer_ = nullptr;
  // CANNOT make the specification of resourceManager_ above the context_!
  // Because when deconstructing the resourceManager_, it needs
  // the GL::Context
  // If you switch the order, you will have the error:
  // GL::Context::current(): no current context from Magnum
  // during the deconstruction
  std::unique_ptr<assets::ResourceManager> resourceManager_ = nullptr;

  /**
   * @brief Owns and manages the metadata/attributes managers
   */
  metadata::MetadataMediator::ptr metadataMediator_ = nullptr;

  /**
   * @brief Configuration describing currently active scene
   */
  metadata::attributes::SceneInstanceAttributes::cptr
      curSceneInstanceAttributes_ = nullptr;

  scene::SceneManager::uptr sceneManager_ = nullptr;

  int activeSceneID_ = ID_UNDEFINED;
  int activeSemanticSceneID_ = ID_UNDEFINED;

  /**
   * @brief Whether or not the loaded scene has a semantic scene mesh loaded.
   */
  bool semanticSceneMeshLoaded_ = false;

  std::vector<int> sceneID_;

  std::shared_ptr<physics::PhysicsManager> physicsManager_ = nullptr;

  std::shared_ptr<esp::gfx::replay::ReplayManager> gfxReplayMgr_;

  core::Random::ptr random_;
  SimulatorConfiguration config_;

  std::vector<agent::Agent::ptr> agents_;

  nav::PathFinder::ptr pathfinder_;
  // state indicating frustum culling is enabled or not
  //
  // TODO:
  // Such state, frustumCulling_ has also been defined in frontend (py)
  // See: examples/settings.py, src_python/habitat_sim/simulator.py for more
  // information ideally, to avoid inconsistency at any time, and reduce
  // maintenance cost this state should be defined in just one place.e.g., only
  // in the frontend Currently, we need it defined here, because sensor., e.g.,
  // PinholeCamera rquires it when drawing the observation
  bool frustumCulling_ = true;

  //! NavMesh visualization variables
  int navMeshVisPrimID_ = esp::ID_UNDEFINED;
  esp::scene::SceneNode* navMeshVisNode_ = nullptr;

  /**
   * @brief Tracks whether or not the simulator was initialized
   * to load textures.  Because we cache mesh loading, this should
   * *not* be changed without calling close() first
   */
  Corrade::Containers::Optional<bool> requiresTextures_;

  std::shared_ptr<esp::gfx::DebugLineRender> debugLineRender_;

  ESP_SMART_POINTERS(Simulator)
};

}  // namespace sim
}  // namespace esp

#endif  // ESP_SIM_SIMULATOR_H_
