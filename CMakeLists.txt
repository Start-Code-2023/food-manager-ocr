cmake_minimum_required(VERSION 3.26)
project(ReceiptOCR)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(./nlohmann/json.hpp)

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ./nlohmann/json.hpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(rapidfuzz REQUIRED)

find_package(CURL REQUIRED)

add_executable(ReceiptOCR main.cpp
        string/titlecase.h
        string/titlecase.cpp)

target_link_libraries(ReceiptOCR ${OpenCV_LIBS})
target_link_libraries(ReceiptOCR nlohmann_json)
target_link_libraries(ReceiptOCR rapidfuzz::rapidfuzz)
target_link_libraries(ReceiptOCR -llept -ltesseract)
target_include_directories(ReceiptOCR PRIVATE ${CURL_INCLUDE_DIRS})
target_link_libraries(ReceiptOCR ${CURL_LIBRARIES})