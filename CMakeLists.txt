cmake_minimum_required(VERSION 2.8.3)
project(mask_rcnn_ros)

find_package(catkin REQUIRED COMPONENTS std_msgs sensor_msgs message_generation)

catkin_python_setup()

add_message_files(
  FILES
    Result.msg
)


generate_messages(
  DEPENDENCIES std_msgs sensor_msgs
)



catkin_package(CATKIN_DEPENDS message_runtime)

#############
## Install ##
#############

install(PROGRAMS
  nodes/mask_rcnn_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  msg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


#############
## Testing ##
#############


## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
