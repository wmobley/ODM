if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

set(cmake_lists "${SOURCE_DIR}/CMakeLists.txt")
file(READ "${cmake_lists}" pypopsift_cmake)

if(NOT pypopsift_cmake MATCHES "install\\(TARGETS pypopsift")
    file(APPEND "${cmake_lists}" "\ninstall(TARGETS pypopsift LIBRARY DESTINATION bin/opensfm/opensfm)\n")
endif()
