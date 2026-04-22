cmake_minimum_required(VERSION 3.5)
project(bt_plugins)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(behaviortree_cpp_v3 REQUIRED)

# Create shared BT plugin library
add_library(bt_human_nodes SHARED
  src/is_human_detected.cpp
)

# ✅ Crucial: Export symbols for BT plugin registration
target_compile_definitions(bt_human_nodes PRIVATE "BT_PLUGIN_EXPORT")

# Include headers (future-proofing)
target_include_directories(bt_human_nodes PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Link dependencies
ament_target_dependencies(bt_human_nodes
  rclcpp
  std_msgs
  pluginlib
  behaviortree_cpp_v3
)

# Install the plugin library
install(TARGETS bt_human_nodes
  LIBRARY DESTINATION lib
)

# Install the plugin XML file
install(FILES bt_plugins.xml
  DESTINATION share/${PROJECT_NAME}
)

# Register plugin with pluginlib
pluginlib_export_plugin_description_file(behaviortree_cpp_v3 bt_plugins.xml)

# Package declaration
ament_package()
