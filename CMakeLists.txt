cmake_minimum_required(VERSION 3.16)
project(sisyphus
    VERSION 0.1.0
    DESCRIPTION "SISYPHUS: Spectral Interpolation System Yielding Precise High-order Univariate Splines"
    LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Find FFTW3
# -----------------------------------------------------------------------------
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 fftw3)
endif()

if(NOT FFTW3_FOUND)
    find_library(FFTW3_LIBRARY fftw3 REQUIRED)
    find_path(FFTW3_INCLUDE_DIR fftw3.h REQUIRED)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARY})
    set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
endif()

# -----------------------------------------------------------------------------
# Header-only interface target
# -----------------------------------------------------------------------------
add_library(sisyphus INTERFACE)
add_library(sisyphus::spectral ALIAS sisyphus)
target_include_directories(sisyphus INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(sisyphus INTERFACE ${FFTW3_LIBRARIES} m)
target_compile_options(sisyphus INTERFACE ${FFTW3_CFLAGS_OTHER})

# Allow users to bring their own FFTW3 include path
if(FFTW3_INCLUDE_DIRS)
    target_include_directories(sisyphus INTERFACE ${FFTW3_INCLUDE_DIRS})
endif()

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(BUILD_TESTING "Build test suite" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -----------------------------------------------------------------------------
# Install rules
# -----------------------------------------------------------------------------
include(GNUInstallDirs)
install(FILES include/sisyphus.h include/sisyphus.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS sisyphus
        EXPORT sisyphus-targets)
install(EXPORT sisyphus-targets
        FILE sisyphus-config.cmake
        NAMESPACE sisyphus::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sisyphus)
