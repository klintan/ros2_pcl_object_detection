cmake_minimum_required(VERSION 3.5.0)

project(pcl_object_detection)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


#find_package(Boost REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(tf2_ros REQUIRED)


include_directories(include ${cv_bridge_INCLUDE_DIRS})

add_library(pcl_object_detection SHARED
  src/pcl_object_detection.cpp)
target_compile_definitions(pcl_object_detection
  PRIVATE "COMPOSITION_BUILDING_DLL")

ament_target_dependencies(pcl_object_detection
    "rclcpp"
    "sensor_msgs"
    "std_msgs"
    "cv_bridge"
    "OpenCV"
    "pcl_conversions"
    "PCL"
  geometry_msgs
  pcl_ros
  shape_msgs
  tf2_ros
)

rclcpp_components_register_nodes(pcl_object_detection "pcl_object_detection::PclObjectDetection")
set(node_plugins "${node_plugins}pcl_object_detection::PclObjectDetection;$<TARGET_FILE:pcl_object_detection>\n")

add_executable(pcl_object_detection_node
    src/pcl_object_detection_node.cpp)
target_link_libraries(pcl_object_detection_node
    pcl_object_detection
)
    
ament_target_dependencies(pcl_object_detection_node
    "rclcpp"
)


install(TARGETS
    pcl_object_detection
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)


install(TARGETS
    pcl_object_detection_node
    DESTINATION lib/${PROJECT_NAME}
)


# Install launch files.
install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
