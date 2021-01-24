cmake_minimum_required(VERSION 3.10)
project(lego)
set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS 1)

set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen/include")

add_subdirectory(dynet_hltc_fork)

# The compiled lego library is pretty big (~500MB), so linking against Lego library is frustratingly slow.
# To solve this problem, use shared library instead of static library
set(BUILD_SHARED_LIBS 1)

add_library(${PROJECT_NAME}
        src/backprop_trainable_parameter.cpp
        src/bilinear_model.cpp
        src/boolean_operations_impl.cpp
        src/lambda_transducer_model.cpp
        src/const_value_model.cpp
        src/dense_model.cpp
        src/dropout_model.cpp
        src/dynet_computation_graph.cpp
        src/embedding_table_model.cpp
        src/generic_rnn_model.cpp
        src/io_operations_impl.cpp
        src/lego_boolean_operations.cpp
        src/lego_initialize.cpp
        src/lego_io_operations.cpp
        src/lego_list_operations.cpp
        src/lego_tensor.cpp
        src/lego_tensor_operations.cpp
        src/lego_training_pipeline.cpp
        src/list_operations_impl.cpp
        src/rnn_cells.cpp
        src/symbol_id_converter_model.cpp
        src/tbd_transducer.cpp
        src/tensor_operations_impl.cpp
        src/transducer_dataset.cpp
        src/transducer_instance.cpp
        src/transducer_model.cpp
        src/transducer_optimizer.cpp
        src/transducer_structure.cpp
        src/transducer_typed_value.cpp
        src/transducer_variant.cpp
        src/value_placeholder.cpp
        src/lambda_transducer_value_cache.cpp
        src/compute_value_behavior.cpp
        src/lego_guard.cpp
        src/composed_transducer_model.cpp
        src/lego_param_naming_guard.cpp
        )




if(NOT MSVC)
    find_package(Threads REQUIRED)
    link_libraries(Threads::Threads)
endif()

install(DIRECTORY src/include/
        DESTINATION "include")

install(DIRECTORY cereal/include/
        DESTINATION "include")

set(INTERNAL_PROGRAM_SOURCES src/main src/examples/xor_demo.cpp src/examples/dynamic_xor_demo.cpp src/examples/lstm_demo.cpp src/examples/rnn_diy_demo.cpp src/examples/recursion_demo.cpp)
# Create a program cmake-target for every $INTERNAL_PROGRAM_SOURCES
foreach(PROGRAM_SOURCE ${INTERNAL_PROGRAM_SOURCES})
    # Ensure that program source starts with `src/`
    string(REGEX MATCH "^src/" STARTS_WITH_SRC ${PROGRAM_SOURCE})
    if(NOT STARTS_WITH_SRC)
        message(FATAL_ERROR "All program sources must starts with src/ got: ${PROGRAM_SOURCE}")
    endif()

    # Assemble target name from the program source
    # For example, if program source is `src/examples/demo.cpp`
    # then target name will be `demo`
    get_filename_component(TARGET_NAME "${PROGRAM_SOURCE}" NAME_WE)

    add_executable(${TARGET_NAME} ${PROGRAM_SOURCE})
    target_link_libraries(${TARGET_NAME} ${PROJECT_NAME} dynet)
endforeach()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cereal/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dynet_hltc_fork)
