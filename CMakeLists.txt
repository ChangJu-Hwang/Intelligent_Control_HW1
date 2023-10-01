cmake_minimum_required(VERSION 3.5)

set(MY_TARGET ANN)

project ( ${MY_TARGET} )

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "RELEASE")
ENDIF()

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

find_package(ament_cmake REQUIRED)

include_directories("include")
file(GLOB SOURCES "src/*.cpp")
add_executable( ${MY_TARGET} ${SOURCES})

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCUDE_DIRS})
target_link_libraries( ${MY_TARGET} ${YAML_CPP_LIBRARIES})

find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
include_directories( ${Boost_INCLUDE_DIRS} )
target_link_libraries(${MY_TARGET} ${Boost_LIBRARIES})