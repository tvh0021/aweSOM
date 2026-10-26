cmake_minimum_required(VERSION 3.14)
project(awesom_classify LANGUAGES CXX)

# Header-only interface target for consumers.
add_library(awesom_classify INTERFACE)
target_include_directories(awesom_classify INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(awesom_classify INTERFACE cxx_std_17)
# No FMA contraction: acc += d*d must round like the Python reference.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(awesom_classify INTERFACE -ffp-contract=off)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  target_compile_options(awesom_classify INTERFACE -ffp-contract=off -fp-model=precise)
elseif(MSVC)
  target_compile_options(awesom_classify INTERFACE /fp:precise)
endif()

add_executable(test_classify test_classify.cpp)
target_link_libraries(test_classify PRIVATE awesom_classify)
