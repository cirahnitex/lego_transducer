//
// Created by Dekai WU and YAN Yuchen on 20200423.
//

#ifndef LEGO_TRANSDUCER_INSTANCE_HPP
#define LEGO_TRANSDUCER_INSTANCE_HPP

#include "transducer_typed_value.hpp"
#include <stdexcept>
#include "transducer_model.hpp"

namespace tg {

  class transducer_model;
  class transducer_dataset;
  class transducer_variant;
  class training_pipeline;

  /**
   * \ingroup transducer
   *
   * \brief A transducer instance is a function object that performs the transduction task when invoked.
   *
   * Unlike a transducer model, it takes concrete input values and returns concrete output values.
   *
   * It can be created from a tg::transducer_model by calling transducer_model::instantiate()
   */
  class transducer_instance {
    std::shared_ptr<transducer_variant> impl;
    friend tg::training_pipeline;
  public:
    transducer_instance() = default;

    transducer_instance(const transducer_instance&) = default;

    transducer_instance(transducer_instance&&) noexcept = default;

    transducer_instance& operator=(const transducer_instance&) = default;

    transducer_instance& operator=(transducer_instance&&) noexcept = default;

    explicit transducer_instance(std::shared_ptr<transducer_variant> impl);

    /**
     * \brief Construct a constant value transducer instance
     *
     * It always returns a constant value.
     *
     * \param v the constant value
     */
    explicit transducer_instance(value_t v);

    /**
     * \brief Invoke this transducer (if this is a nullary transducer)
     * \return the output
     */
    value_t operator()();

    /**
     * \brief Invoke this transducer on 1 input (if this is an unary transducer)
     * \param in0 input#0
     * \return the output
     */
    value_t operator()(const value_t& in0);

    /**
     * \brief Invoke the transducer on 2 inputs (if this is a binary transducer)
     * \param in0 input#0
     * \param in1 input#1
     * \return the output
     */
    value_t operator()(const value_t& in0, const value_t& in1);

    /**
     * \brief Invoke the transducer on N inputs (if this is an N-ary transducer)
     * \tparam T = typed_value
     * \param ins input#0, intput#1, input#2, ... , input#(N-1)
     * \return the output
     */
    template<typename... T>
    value_t operator()(const T& ... ins) {
      std::vector<value_t> args{value_t(ins)...};
      return apply(args);
    }

    /**
     * \brief Invoke the transducer on N inputs (if this is an N-ary transducer)
     *
     * This is the non-variadic version of operator()
     *
     * \param ins the list of inputs
     * \return the output
     */
    value_t apply(const std::vector<value_t>& ins);

    std::vector<value_t> batch_apply(const std::shared_ptr<const transducer_dataset>& dataset);

    /**
     * \brief Invoke the transducer on N inputs, and compute backward pass
     *
     * The transducer must return tensor of shape {1}, representing the loss
     *
     * \param ins input#0, intput#1, input#2, ... , input#(N-1)
     * \return the output loss
     */
    template<typename... T>
    scalar_t backward(const T& ... ins) {
      std::vector<value_t> args{value_t(ins)...};
      return apply_backward(args);
    }

    /**
     * \brief The non-variadic version of backward()
     *
     * \param ins The input dataset
     * \return The output loss
     */
    scalar_t apply_backward(const std::vector<value_t>& ins);

    scalar_t batch_backward(const std::shared_ptr<const transducer_dataset>& dataset);

    static scalar_t dynamic_batch_backward(const std::vector<tg::dynamic_transducer_application>& loss_applications);

    static std::vector<value_t> dynamic_batch_apply(const std::vector<tg::dynamic_transducer_application>& applications);

    /**
     * \brief Get a transducer_model from this transducer instance
     * \return The transducer model
     */
    transducer_model prototype();

  };
}



#endif //LEGO_TRANSDUCER_INSTANCE_HPP
