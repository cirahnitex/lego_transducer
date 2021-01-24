//
// Created by Dekai WU and YAN Yuchen on 20200427.
//

#ifndef LEGO_LEGO_TRANSDUCER_HPP
#define LEGO_LEGO_TRANSDUCER_HPP
#include "lego_boolean_operations.hpp"
#include "lego_guard.hpp"
#include "lego_initialize.hpp"
#include "lego_io_operations.hpp"
#include "lego_list_operations.hpp"
#include "lego_primitive_types.hpp"
#include "lego_serialization_helper.hpp"
#include "lego_tensor.hpp"
#include "lego_tensor_operations.hpp"
#include "lego_training_pipeline.hpp"
#include "lego_param_naming_guard.hpp"
#include "parallel_array_map.hpp"
#include "transducer_dataset.hpp"
#include "transducer_instance.hpp"
#include "transducer_model.hpp"
#include "transducer_optimizer.hpp"
#include "transducer_structure.hpp"
#include "transducer_typed_value.hpp"

/**
 * \defgroup global_configurations Global configurations
 *
 * \brief Functions related to global behaviors of the entire library
 *
 */

/**
 * \defgroup transducer Transducer
 *
 * \brief A transducer transforms input data to output data
 *
 * Transducers is used mainly in two ways: evaluating and composing.
 *   * <b>Transducing</b>: A transducer takes one or more input data (of type tg::typed_value), and computes an output data (of type tg::typed_value).
 *   * <b>Composing</b>: Connect a "parent" transducer with a list of "child" transducers, forming a new transducer that takes the output of the "child" transducers as the input to the "parent" transducer. This is analogous to composing functions in functional programming.
 *
 */

/**
 * \defgroup data_values Data values
 * \brief The data types that get passed between transducers
 *
 * This group include
 *   - tg::value_t: The type describing data values that gets passed between transducers.
 *   - the more specific data types that gets wrapped by tg::value_t, such as integers, scalars, tensors and symbols
 *
 */

/**
 * \defgroup boolean_operations Boolean operations
 * \brief Functions related to booleans
 *
 * All functions in this file composes transducers instead of computing values immediately. In other words,
 * these functions takes one or more transducer models (that will produce the inputs when evaluated) and returns a transducer model
 * (that will produce the outputs when evaluated)
 */

/**
 * \defgroup io_operations IO operations
 * \brief Functions for printing messages
 *
 */

/**
 * \defgroup tensor_operations Tensor operations
 * \brief Tensor arithmetic functions
 *
 * All functions in this group composes transducers instead of computing values immediately. In other words,
 * these functions takes one or more transducer models (that will produce the inputs when evaluated) and returns a transducer model
 * (that will produce the outputs when evaluated)
 */

/**
 * \defgroup list_operations List operations
 * \brief Functions related to manipulating lists
 *
 * All functions in this group composes transducers instead of computing values immediately. In other words,
 * these functions takes one or more transducer models (that will produce the inputs when evaluated) and returns a transducer model
 * (that will produce the outputs when evaluated)
 */

/**
 * \defgroup training Training
 * \brief Functions and classes related to backprop training
 */

/**
 * \defgroup parallel_computing Parallel computing
 * \brief Invoke transducers using parallel computation for general purposes
 *
 * This group include functions for general purpose parallel computations.
 * If you want to perform parallel computation other than the built-in backprop training, please look at functions in this group.
 *
 * @{
 */

/**
 * \defgroup utilities Utilities
 * \brief Other utility functions and classes that doesn't fit into any group
 *
 */
#endif //LEGO_LEGO_TRANSDUCER_HPP
