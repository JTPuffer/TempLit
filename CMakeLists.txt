cmake_minimum_required(VERSION 3.20)

project(GraphCompiler LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GRAPH_TEST_BACKEND "CPU" CACHE STRING "Backend for backend-agnostic tests: CPU or METAL")
set_property(CACHE GRAPH_TEST_BACKEND PROPERTY STRINGS CPU METAL)
option(GRAPH_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" ON)

set(PROJECT_WARNINGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
)
if(GRAPH_ENABLE_SANITIZERS)
  add_compile_options(
      -fsanitize=address,undefined
      -fno-omit-frame-pointer
  )

  add_link_options(
      -fsanitize=address,undefined
  )
endif()
add_executable(train src/train.cpp src/model_utils/dataset.cpp
                     src/model_utils/bpe_tokeniser.cpp
                     src/model_utils/loader.cpp
                     src/model_utils/archive.cpp
                     src/my_metal/metal.cpp src/my_metal/runtime.cpp)
target_include_directories(train PRIVATE 
	include
)
target_include_directories(train SYSTEM PRIVATE
	${CMAKE_SOURCE_DIR}/external/metal-cpp
)
target_compile_options(train PRIVATE ${PROJECT_WARNINGS}
                                     -ftemplate-depth=4096
                                     -fbracket-depth=8192
                                     -gline-tables-only)
target_compile_features(train PRIVATE cxx_std_20)

add_executable(generate src/generate.cpp src/model_utils/dataset.cpp
                              src/model_utils/bpe_tokeniser.cpp
                              src/model_utils/loader.cpp
                              src/model_utils/archive.cpp
                              src/my_metal/metal.cpp src/my_metal/runtime.cpp)
target_include_directories(generate PRIVATE 
	include
)
target_include_directories(generate SYSTEM PRIVATE
	${CMAKE_SOURCE_DIR}/external/metal-cpp
)
target_compile_options(generate PRIVATE ${PROJECT_WARNINGS}
                                              -ftemplate-depth=4096
                                              -fbracket-depth=8192
                                              -gline-tables-only)
target_compile_features(generate PRIVATE cxx_std_20)

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_program(XCRUN_EXECUTABLE xcrun REQUIRED)

set(METAL_SOURCE ${CMAKE_SOURCE_DIR}/include/my_metal/math.metal)
set(METAL_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/metal)
set(METAL_AIR ${METAL_BUILD_DIR}/Metal_Math.air)
set(METAL_MATH_LIBRARY ${METAL_BUILD_DIR}/Metal_Math.metallib)

add_custom_command(
  OUTPUT ${METAL_AIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${METAL_BUILD_DIR}
  COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -c
          ${METAL_SOURCE} -o ${METAL_AIR}
  DEPENDS ${METAL_SOURCE}
  VERBATIM
)

add_custom_command(
  OUTPUT ${METAL_MATH_LIBRARY}
  COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib
          ${METAL_AIR} -o ${METAL_MATH_LIBRARY}
  DEPENDS ${METAL_AIR}
  VERBATIM
)

add_custom_target(metal_math_library DEPENDS ${METAL_MATH_LIBRARY})

function(configure_metal_target target)
  add_dependencies(${target} metal_math_library)
  target_include_directories(${target} SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/external/metal-cpp
  )
  target_link_libraries(${target} PRIVATE
    ${FOUNDATION_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
  )
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${METAL_MATH_LIBRARY}
      $<TARGET_FILE_DIR:${target}>/Metal_Math.metallib
  )
endfunction()

configure_metal_target(train)
configure_metal_target(generate)


enable_testing()

add_executable(test_expr tests/test_expr.cpp)
target_include_directories(test_expr PRIVATE include)
target_compile_features(test_expr PRIVATE cxx_std_20)
add_test(NAME expr COMMAND test_expr)

function(add_graph_backend_test target source test_name)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE include tests)
  target_compile_features(${target} PRIVATE cxx_std_20)

  if(GRAPH_TEST_BACKEND STREQUAL "METAL")
    target_sources(${target} PRIVATE src/my_metal/metal.cpp
                                     src/my_metal/runtime.cpp)
    target_compile_definitions(${target} PRIVATE GRAPH_TEST_BACKEND_METAL)
    configure_metal_target(${target})
  elseif(GRAPH_TEST_BACKEND STREQUAL "CPU")
    target_sources(${target} PRIVATE src/CPU/cpu.cpp)
    target_compile_definitions(${target} PRIVATE GRAPH_TEST_BACKEND_CPU)
  else()
    message(FATAL_ERROR "GRAPH_TEST_BACKEND must be CPU or METAL")
  endif()

  add_test(NAME ${test_name} COMMAND ${target})
endfunction()

add_graph_backend_test(test_math tests/test_math.cpp math)

add_graph_backend_test(test_forward tests/test_forward.cpp forward)

add_graph_backend_test(test_backward tests/test_backward.cpp backward)

add_graph_backend_test(test_passes tests/test_passes_cache.cpp passes)
add_graph_backend_test(test_meta tests/test_meta.cpp meta)

add_executable(test_tensor tests/test_tensor.cpp)
target_include_directories(test_tensor PRIVATE include)
target_compile_features(test_tensor PRIVATE cxx_std_20)
add_test(NAME tensor COMMAND test_tensor)

add_executable(test_model_storage tests/test_model_storage.cpp
                                  src/model_utils/archive.cpp
                                  src/CPU/cpu.cpp)
target_include_directories(test_model_storage PRIVATE include)
target_compile_features(test_model_storage PRIVATE cxx_std_20)
add_test(NAME model_storage COMMAND test_model_storage)

add_executable(test_dataset tests/test_dataset.cpp src/model_utils/dataset.cpp
                            src/model_utils/loader.cpp)
target_include_directories(test_dataset PRIVATE include)
target_compile_features(test_dataset PRIVATE cxx_std_20)
add_test(NAME dataset COMMAND test_dataset)


add_graph_backend_test(test_basic_nn tests/test_basic_nn.cpp basic_nn)

add_graph_backend_test(test_mlp tests/test_mlp.cpp mlp)

add_executable(test_metal tests/test_metal.cpp src/my_metal/metal.cpp
                          src/my_metal/runtime.cpp src/CPU/cpu.cpp)
target_include_directories(test_metal PRIVATE
    include
)
target_include_directories(test_metal SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/external/metal-cpp
)
target_compile_features(test_metal PRIVATE cxx_std_20)
configure_metal_target(test_metal)
add_test(NAME metal COMMAND test_metal)

add_executable(test_metal_runtime tests/test_metal_runtime.cpp
                                  src/my_metal/metal.cpp
                                  src/my_metal/runtime.cpp)
target_include_directories(test_metal_runtime PRIVATE include)
target_include_directories(test_metal_runtime SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/external/metal-cpp
)
target_compile_features(test_metal_runtime PRIVATE cxx_std_20)
configure_metal_target(test_metal_runtime)
add_test(NAME metal_runtime COMMAND test_metal_runtime)

add_executable(test_attention tests/test_attention.cpp src/my_metal/metal.cpp
                              src/my_metal/runtime.cpp)
target_include_directories(test_attention PRIVATE include)
target_include_directories(test_attention SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/external/metal-cpp
)
target_compile_features(test_attention PRIVATE cxx_std_20)
target_compile_options(test_attention PRIVATE ${PROJECT_WARNINGS})
configure_metal_target(test_attention)
add_test(NAME attention COMMAND test_attention)


add_executable(test_bpe tests/test_bpe.cpp src/model_utils/bpe_tokeniser.cpp)
target_include_directories(test_bpe PRIVATE include)
target_compile_features(test_bpe PRIVATE cxx_std_20)
target_compile_options(test_bpe PRIVATE ${PROJECT_WARNINGS})
add_test(NAME bpe COMMAND test_bpe)

add_executable(train_bpe dataset/train_bpe.cpp
                         src/model_utils/bpe_tokeniser.cpp
                         src/model_utils/dataset.cpp)
target_include_directories(train_bpe PRIVATE include)
target_compile_features(train_bpe PRIVATE cxx_std_20)
target_compile_options(train_bpe PRIVATE ${PROJECT_WARNINGS})

add_executable(tokenise_dataset dataset/tokenise_dataset.cpp
                                src/model_utils/bpe_tokeniser.cpp
                                src/model_utils/dataset.cpp)
target_include_directories(tokenise_dataset PRIVATE include)
target_compile_features(tokenise_dataset PRIVATE cxx_std_20)
target_compile_options(tokenise_dataset PRIVATE ${PROJECT_WARNINGS})
