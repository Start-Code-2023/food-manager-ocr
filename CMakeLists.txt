cmake_minimum_required(VERSION 3.26)
project(ReceiptOCR)

# Set C++ standard to C++23
set(CMAKE_CXX_STANDARD 23)

# Enable compiler optimization level 3
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Include the nlohmann/json.hpp directory
include_directories(${CMAKE_SOURCE_DIR}/nlohmann)

# Find required libraries
find_package(OpenCV REQUIRED)
find_package(rapidfuzz REQUIRED)
find_package(CURL REQUIRED)

# Create an executable for ReceiptOCR
add_executable(ReceiptOCR main.cpp ReceiptOCR/receiptOCR.h)

# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Link the libraries to the ReceiptOCR executable
target_link_libraries(ReceiptOCR
        ${OpenCV_LIBS}
        rapidfuzz::rapidfuzz
        -llept
        -ltesseract
        ${CURL_LIBRARIES})

# Create an interface library for nlohmann/json.hpp
add_library(nlohmann_json INTERFACE)

# Set include directories for the nlohmann_json library
target_include_directories(nlohmann_json INTERFACE ${CMAKE_SOURCE_DIR}/nlohmann)

# Link the nlohmann_json library to the ReceiptOCR executable
target_link_libraries(ReceiptOCR nlohmann_json)
