cmake_minimum_required(VERSION 3.12)
project(depth_clustering_python)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem regex program_options)

# Optional PCL support
find_package(PCL QUIET COMPONENTS common io)

# Optional Qt support (for visualization)
find_package(Qt5 QUIET COMPONENTS Core Widgets OpenGL)

# Optional ROS support
find_package(catkin QUIET COMPONENTS roscpp sensor_msgs std_msgs nav_msgs message_filters tf)

# Include directories
include_directories(
    src
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

if(PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    add_definitions(-DPCL_FOUND)
endif()

if(Qt5_FOUND)
    include_directories(${Qt5Core_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS})
    add_definitions(-DQT_FOUND)
endif()

if(catkin_FOUND)
    include_directories(${catkin_INCLUDE_DIRS})
    add_definitions(-DROS_FOUND)
endif()

# Create the Python module
pybind11_add_module(depth_clustering
    src/python_bindings.cpp
    # Core source files
    src/utils/rich_point.cpp
    src/utils/cloud.cpp
    src/utils/bbox.cpp
    src/utils/folder_reader.cpp
    src/utils/velodyne_utils.cpp
    src/projections/projection_params.cpp
    src/projections/cloud_projection.cpp
    src/projections/ring_projection.cpp
    src/projections/spherical_projection.cpp
    src/ground_removal/depth_ground_remover.cpp
    src/communication/identifiable.cpp
    src/image_labelers/abstract_image_labeler.cpp
    src/image_labelers/diff_helpers/angle_diff.cpp
    src/image_labelers/diff_helpers/line_dist_diff.cpp
    src/visualization/cloud_saver.cpp
    src/visualization/visualizer.cpp
)

# Link libraries
target_link_libraries(depth_clustering PRIVATE
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
)

if(PCL_FOUND)
    target_link_libraries(depth_clustering PRIVATE
        ${PCL_LIBRARIES}
    )
endif()

if(Qt5_FOUND)
    target_link_libraries(depth_clustering PRIVATE
        ${Qt5Core_LIBRARIES}
        ${Qt5Widgets_LIBRARIES}
        ${Qt5OpenGL_LIBRARIES}
    )
endif()

if(catkin_FOUND)
    target_link_libraries(depth_clustering PRIVATE
        ${catkin_LIBRARIES}
    )
endif()

# Set properties
set_target_properties(depth_clustering PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

# Compiler-specific options
if(MSVC)
    target_compile_definitions(depth_clustering PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
else()
    target_compile_definitions(depth_clustering PRIVATE VERSION_INFO="${EXAMPLE_VERSION_INFO}")
endif()

# Installation
install(TARGETS depth_clustering
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install Python package
install(DIRECTORY depth_clustering/
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/python3/site-packages/depth_clustering
    FILES_MATCHING PATTERN "*.py"
)

# Install examples
install(DIRECTORY examples/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/depth_clustering/examples
    FILES_MATCHING PATTERN "*.py"
)
