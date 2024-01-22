# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set(DEPS_DIR "${CMAKE_CURRENT_LIST_DIR}/../deps")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}")

# Find Corrade first so we can use CORRADE_TARGET_*
if(NOT USE_SYSTEM_MAGNUM)
  # These are enabled by default but we don't need them right now -- disabling
  # for slightly faster builds. If you need any of these, simply delete a line.
  set(WITH_INTERCONNECT OFF CACHE BOOL "" FORCE)
  # Ensure Corrade should be built statically if Magnum is.
  set(BUILD_PLUGINS_STATIC ON CACHE BOOL "" FORCE)
  set(BUILD_STATIC ON CACHE BOOL "" FORCE)
  set(BUILD_STATIC_PIC ON CACHE BOOL "" FORCE)
  add_subdirectory("${DEPS_DIR}/corrade")
endif()
find_package(Corrade REQUIRED Utility)

# OpenMP
find_package(OpenMP)
# We don't find_package(OpenGL REQUIRED) here, but let Magnum do that instead
# as it sets up various things related to GLVND.

include_directories("deps")

# Eigen. Use a system package, if preferred.
if(USE_SYSTEM_EIGEN)
  find_package(Eigen3 REQUIRED)
  include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
else()
  include_directories(SYSTEM "${DEPS_DIR}/eigen")
  set(EIGEN3_INCLUDE_DIR "${DEPS_DIR}/eigen")
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${DEPS_DIR}/eigen/cmake")
endif()

if(NOT IMGUI_DIR)
  set(IMGUI_DIR "${DEPS_DIR}/imgui")
endif()

# tinyxml2
include_directories("${DEPS_DIR}/tinyxml2")
add_subdirectory("${DEPS_DIR}/tinyxml2")

# RapidJSON. Use a system package, if preferred.
if(USE_SYSTEM_RAPIDJSON)
  find_package(RapidJSON CONFIG REQUIRED)
  include_directories(SYSTEM ${RapidJSON_INCLUDE_DIR})
else()
  include_directories(SYSTEM "${DEPS_DIR}/rapidjson/include")
endif()

# Why all the weird options in the set() calls below?
# See https://stackoverflow.com/questions/3766740/overriding-a-default-option-value-in-cmake-from-a-parent-cmakelists-txt

if(BUILD_ASSIMP_SUPPORT)
  # Assimp. Use a system package, if preferred.
  if(NOT USE_SYSTEM_ASSIMP)
    set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "ASSIMP_BUILD_ASSIMP_TOOLS" FORCE)
    set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "ASSIMP_BUILD_TESTS" FORCE)
    set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "BUILD_SHARED_LIBS" FORCE)
    # The following is important to avoid Assimp appending `d` to all our
    # binaries. Works only with Assimp >= 5.0.0, and after 5.0.1 this option is
    # prefixed with ASSIMP_, so better set both variants to future-proof this.
    set(INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
    set(ASSIMP_INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
    # Otherwise Assimp may attempt to find minizip on the filesystem using
    # pkgconfig, but in a poor way that expects just yelling -lminizip at the
    # linker without any link directories would work. It won't. (The variable
    # is not an option() so no need to CACHE it.)
    set(ASSIMP_BUILD_MINIZIP ON)
    add_subdirectory("${DEPS_DIR}/assimp")

    # Help FindAssimp locate everything
    set(ASSIMP_INCLUDE_DIR "${DEPS_DIR}/assimp/include" CACHE STRING "" FORCE)
    set(ASSIMP_LIBRARY_DEBUG assimp CACHE STRING "" FORCE)
    set(ASSIMP_LIBRARY_RELEASE assimp CACHE STRING "" FORCE)
    add_library(Assimp::Assimp ALIAS assimp)
  endif()
  find_package(Assimp REQUIRED)
endif()

# v-hacd
if(BUILD_WITH_VHACD)
  set(NO_OPENCL ON CACHE BOOL "NO_OPENCL" FORCE)
  set(NO_OPENMP ON CACHE BOOL "NO_OPENMP" FORCE)
  # adding /src/VHACD_Lib instead of /src since /src contains unneccesary test files
  add_subdirectory("${DEPS_DIR}/v-hacd/src/VHACD_Lib")
endif()

# audio
if(BUILD_WITH_AUDIO)
  find_library(
    RLRAudioPropagation_LIBRARY RLRAudioPropagation
    PATHS ${DEPS_DIR}/rlr-audio-propagation/RLRAudioPropagationPkg/libs/linux/x64
  )
endif()

# recast
set(RECASTNAVIGATION_DEMO OFF CACHE BOOL "RECASTNAVIGATION_DEMO" FORCE)
set(RECASTNAVIGATION_TESTS OFF CACHE BOOL "RECASTNAVIGATION_TESTS" FORCE)
set(RECASTNAVIGATION_EXAMPLES OFF CACHE BOOL "RECASTNAVIGATION_EXAMPLES" FORCE)
# Temp BUILD_SHARED_LIBS override for Recast, has to be reset back after to avoid affecting Bullet (which needs shared) and potentially other submodules
set(_PREV_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
include(GNUInstallDirs)
add_subdirectory("${DEPS_DIR}/recastnavigation/Recast")
add_subdirectory("${DEPS_DIR}/recastnavigation/Detour")
set(BUILD_SHARED_LIBS ${_PREV_BUILD_SHARED_LIBS})
# Needed so that Detour doesn't hide the implementation of the method on dtQueryFilter
target_compile_definitions(Detour PUBLIC DT_VIRTUAL_QUERYFILTER)

if(BUILD_PYTHON_BINDINGS)
  # Before calling find_package(PythonInterp) search for python executable not
  # in the default paths to pick up activated virtualenv/conda python
  find_program(
    PYTHON_EXECUTABLE
    # macOS still defaults to `python` being Python 2, so look for `python3`
    # first
    NAMES python3 python
    PATHS ENV PATH # look in the PATH environment variable
    NO_DEFAULT_PATH # do not look anywhere else...
  )

  # Let the Find module do proper version checks on what we found (it uses the
  # same PYTHON_EXECUTABLE variable, will pick it up from the cache)
  find_package(PythonInterp 3.7 REQUIRED)

  message(STATUS "Bindings being generated for python at ${PYTHON_EXECUTABLE}")

  # Pybind11. Use a system package, if preferred. This needs to be before Magnum
  # so the bindings can properly detect pybind11 added as a subproject.
  if(USE_SYSTEM_PYBIND11)
    find_package(pybind11 REQUIRED)
  else()
    add_subdirectory("${DEPS_DIR}/pybind11")
  endif()
endif()

if(BUILD_WITH_BULLET AND NOT USE_SYSTEM_BULLET)
  # The below block except for the visiblity patch verbatim copied from
  # https://doc.magnum.graphics/magnum/namespaceMagnum_1_1BulletIntegration.html

  # Disable Bullet tests and demos
  set(BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_BULLET2_DEMOS OFF CACHE BOOL "" FORCE)
  set(BUILD_CPU_DEMOS OFF CACHE BOOL "" FORCE)
  set(BUILD_OPENGL3_DEMOS OFF CACHE BOOL "" FORCE)
  # While not needed for Magnum, you might actually want some of those
  set(BUILD_ENET OFF CACHE BOOL "" FORCE)
  set(BUILD_CLSOCKET OFF CACHE BOOL "" FORCE)
  set(BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
  set(BUILD_BULLET3 OFF CACHE BOOL "" FORCE)
  # This is needed in case BUILD_EXTRAS is enabled, as you'd get a CMake syntax
  # error otherwise
  set(PKGCONFIG_INSTALL_PREFIX "lib${LIB_SUFFIX}/pkgconfig/")

  # caches CXX_FLAGS so we can reset them at the end
  set(_PREV_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

  # Bullet's buildsystem doesn't correctly express dependencies between static
  # libs, causing linker errors on Magnum side. If you have CMake 3.13, the
  # Find module is able to correct that on its own, otherwise you need to
  # enable BUILD_SHARED_LIBS to build as shared.
  if((NOT CORRADE_TARGET_EMSCRIPTEN) AND CMAKE_VERSION VERSION_LESS 3.13)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    # however the whole Habitat is built with -fvisibility=hidden and Bullet
    # doesn't export any of its symbols and relies on symbols being visible by
    # default. Which means we have to compile it without hidden visibility.
    # ... and because we have to build shared libs, we need exported symbols,
    string(REPLACE "-fvisibility=hidden" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  else()
    # On Emscripten we require 3.13, so there it's fine (and there we can't use
    # shared libs)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  endif()
  ## Bullet Optimisation Bug
  # We need to define this macro for bullet to bypass an early-out optimisation that
  # was added to bullet via this PR https://github.com/bulletphysics/bullet3/pull/4190 ,
  # specifically here :
  #      https://github.com/erwincoumans/bullet3/blob/28b951c128b53e1dcf26271dd47b88776148a940/src/BulletCollision/CollisionDispatch/btConvexConcaveCollisionAlgorithm.cpp#L106
  # that causes rigid objects to never come to rest.
  # This needs to be further examined on bullet side
  add_definitions(-DBT_DISABLE_CONVEX_CONCAVE_EARLY_OUT=1)
  add_subdirectory(${DEPS_DIR}/bullet3 EXCLUDE_FROM_ALL)
  set(CMAKE_CXX_FLAGS ${_PREV_CMAKE_CXX_FLAGS})
endif()

# Magnum. Use a system package, if preferred.
if(NOT USE_SYSTEM_MAGNUM)
  # Magnum is already set to be build statically when Corrade is above.

  # These are enabled by default but we don't need them right now -- disabling
  # for slightly faster builds. If you need any of these, simply delete a line.
  set(WITH_TEXT OFF CACHE BOOL "" FORCE)
  set(WITH_TEXTURETOOLS OFF CACHE BOOL "" FORCE)

  # These are not enabled by default but we need them
  set(WITH_ANYSCENEIMPORTER ON CACHE BOOL "WITH_ANYSCENEIMPORTER" FORCE)
  if(BUILD_ASSIMP_SUPPORT)
    set(WITH_ASSIMPIMPORTER ON CACHE BOOL "WITH_ASSIMPIMPORTER" FORCE)
  endif()
  set(WITH_GLTFIMPORTER ON CACHE BOOL "" FORCE)
  set(WITH_ANYIMAGEIMPORTER ON CACHE BOOL "WITH_ANYIMAGEIMPORTER" FORCE)
  set(WITH_ANYIMAGECONVERTER ON CACHE BOOL "WITH_ANYIMAGECONVERTER" FORCE)
  set(WITH_PRIMITIVEIMPORTER ON CACHE BOOL "" FORCE)
  set(WITH_STANFORDIMPORTER ON CACHE BOOL "" FORCE)
  set(WITH_STBIMAGEIMPORTER ON CACHE BOOL "WITH_STBIMAGEIMPORTER" FORCE)
  set(WITH_STBIMAGECONVERTER ON CACHE BOOL "WITH_STBIMAGECONVERTER" FORCE)
  set(WITH_EMSCRIPTENAPPLICATION OFF CACHE BOOL "WITH_EMSCRIPTENAPPLICATION" FORCE)
  set(WITH_GLFWAPPLICATION OFF CACHE BOOL "WITH_GLFWAPPLICATION" FORCE)
  set(WITH_EIGEN ON CACHE BOOL "WITH_EIGEN" FORCE) # Eigen integration
  set(WITH_IMGUI ON CACHE BOOL "WITH_IMGUI" FORCE) # ImGui integration
  if(BUILD_PYTHON_BINDINGS)
    set(WITH_PYTHON ON CACHE BOOL "" FORCE) # Python bindings
  endif()
  # We only support WebGL2
  if(CORRADE_TARGET_EMSCRIPTEN)
    set(TARGET_GLES2 OFF CACHE BOOL "" FORCE)
  endif()
  if(BUILD_TEST)
    set(WITH_OPENGLTESTER ON CACHE BOOL "" FORCE)
  endif()

  # Basis Universal. The repo is extremely huge and so instead of a Git
  # submodule we bundle just the transcoder files, and only a subset of the
  # formats (BC7 mode 6 has > 1 MB tables, ATC/FXT1/PVRTC2 are quite rare and
  # not supported by Magnum).
  set(BASIS_UNIVERSAL_DIR "${DEPS_DIR}/basis-universal")
  # Disabling Zstd for now, when it's actually needed we bundle a submodule
  set(CMAKE_DISABLE_FIND_PACKAGE_Zstd ON)
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DBASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY=0 -DBASISD_SUPPORT_ATC=0 -DBASISD_SUPPORT_FXT1=0 -DBASISD_SUPPORT_PVRTC2=0"
  )
  set(WITH_BASISIMPORTER ON CACHE BOOL "" FORCE)

  if(BUILD_BASIS_COMPRESSOR)
    # ImageConverter tool for basis
    set(WITH_IMAGECONVERTER ON CACHE BOOL "" FORCE)
    set(WITH_BASISIMAGECONVERTER ON CACHE BOOL "" FORCE)
  endif()

  # OpenEXR. Use a system package, if preferred.
  if(NOT USE_SYSTEM_OPENEXR)
    # Disable unneeded functionality
    set(PYILMBASE_ENABLE OFF CACHE BOOL "" FORCE)
    set(IMATH_INSTALL_PKG_CONFIG OFF CACHE BOOL "" FORCE)
    set(IMATH_INSTALL_SYM_LINK OFF CACHE BOOL "" FORCE)
    set(OPENEXR_INSTALL OFF CACHE BOOL "" FORCE)
    set(OPENEXR_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(OPENEXR_INSTALL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(OPENEXR_INSTALL_PKG_CONFIG OFF CACHE BOOL "" FORCE)
    set(OPENEXR_INSTALL_TOOLS OFF CACHE BOOL "" FORCE)
    set(OPENEXR_BUILD_UTILS OFF CACHE BOOL "" FORCE)
    # Otherwise OpenEXR uses C++14, and before OpenEXR 3.0.2 also forces C++14
    # on all libraries that link to it.
    set(OPENEXR_CXX_STANDARD 11 CACHE STRING "" FORCE)
    # OpenEXR implicitly bundles Imath. However, without this only the first
    # CMake run will pass and subsequent runs will fail.
    set(CMAKE_DISABLE_FIND_PACKAGE_Imath ON)
    # Disable threading on Emscripten. Brings more problems than is currently
    # worth-
    if(CORRADE_TARGET_EMSCRIPTEN)
      set(OPENEXR_ENABLE_THREADING OFF CACHE BOOL "" FORCE)
    endif()
    # These variables may be used by other projects, so ensure they're reset
    # back to their original values after. OpenEXR forces CMAKE_DEBUG_POSTFIX
    # to _d, which isn't desired outside of that library.
    set(_PREV_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(_PREV_BUILD_TESTING ${BUILD_TESTING})
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTING OFF)
    set(CMAKE_DEBUG_POSTFIX "" CACHE STRING "" FORCE)
    add_subdirectory("${DEPS_DIR}/openexr" EXCLUDE_FROM_ALL)
    set(BUILD_SHARED_LIBS ${_PREV_BUILD_SHARED_LIBS})
    set(BUILD_TESTING ${_PREV_BUILD_TESTING})
    unset(CMAKE_DEBUG_POSTFIX CACHE)

    set(WITH_OPENEXRIMPORTER ON CACHE BOOL "" FORCE)
    set(WITH_OPENEXRIMAGECONVERTER ON CACHE BOOL "" FORCE)
  endif()

  if(BUILD_WITH_BULLET)
    # Build Magnum's BulletIntegration
    set(WITH_BULLET ON CACHE BOOL "" FORCE)
  else()
    set(WITH_BULLET OFF CACHE BOOL "" FORCE)
  endif()

  if(BUILD_GUI_VIEWERS)
    if(CORRADE_TARGET_EMSCRIPTEN)
      set(WITH_EMSCRIPTENAPPLICATION ON CACHE BOOL "WITH_EMSCRIPTENAPPLICATION" FORCE)
    else()
      if(NOT USE_SYSTEM_GLFW)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        # These two will be off-by-default when GLFW 3.4 gets released
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        add_subdirectory("${DEPS_DIR}/glfw")
      endif()
      set(WITH_GLFWAPPLICATION ON CACHE BOOL "WITH_GLFWAPPLICATION" FORCE)
    endif()
  endif()
  if(APPLE)
    set(WITH_WINDOWLESSCGLAPPLICATION ON CACHE BOOL "WITH_WINDOWLESSCGLAPPLICATION"
                                               FORCE
    )
  elseif(WIN32)
    set(WITH_WINDOWLESSWGLAPPLICATION ON CACHE BOOL "WITH_WINDOWLESSWGLAPPLICATION"
                                               FORCE
    )
  elseif(CORRADE_TARGET_EMSCRIPTEN)
    set(WITH_WINDOWLESSEGLAPPLICATION ON CACHE INTERNAL "WITH_WINDOWLESSEGLAPPLICATION"
                                               FORCE
    )
  elseif(UNIX)
    if(BUILD_GUI_VIEWERS)
      set(WITH_WINDOWLESSGLXAPPLICATION ON CACHE INTERNAL
                                                 "WITH_WINDOWLESSGLXAPPLICATION" FORCE
      )
      set(WITH_WINDOWLESSEGLAPPLICATION OFF CACHE INTERNAL
                                                  "WITH_WINDOWLESSEGLAPPLICATION" FORCE
      )
    else()
      set(WITH_WINDOWLESSGLXAPPLICATION OFF CACHE INTERNAL
                                                  "WITH_WINDOWLESSGLXAPPLICATION" FORCE
      )
      set(WITH_WINDOWLESSEGLAPPLICATION ON CACHE INTERNAL
                                                 "WITH_WINDOWLESSEGLAPPLICATION" FORCE
      )
    endif()
  endif()
  add_subdirectory("${DEPS_DIR}/magnum")
  add_subdirectory("${DEPS_DIR}/magnum-plugins")
  add_subdirectory("${DEPS_DIR}/magnum-integration")
  if(BUILD_PYTHON_BINDINGS)
    add_subdirectory("${DEPS_DIR}/magnum-bindings")
  endif()

endif()

if(NOT CORRADE_TARGET_EMSCRIPTEN)
  add_library(atomic_wait STATIC ${DEPS_DIR}/atomic_wait/atomic_wait.cpp)
  target_include_directories(atomic_wait PUBLIC ${DEPS_DIR}/atomic_wait)
endif()
