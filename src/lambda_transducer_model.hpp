//
// Created by Dekai WU and YAN Yuchen on 20200423.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_LAMBDA_TRANSDUCER_MODEL_HPP
#define LEGO_LAMBDA_TRANSDUCER_MODEL_HPP

#include "include/value_placeholder.hpp"
#include "compute_value_behavior.hpp"
#include "lambda_transducer_value_cache.hpp"

#include <variant>
#include <mutex>
#include <memory>

namespace tg {

  class transducer_variant;

  /**
   * \brief This is a global (thread local) stack holding the lambda transducers that are under construction.
   *
   * When constructing transducers using lambda syntax, it is useful to know what is the current transducer that is being constructed. For example, when constructing a value placeholder, the value placeholder needs to register itself to its owner transducer.
   */
  class lambda_transducer_model_construction_guard {
    static thread_local std::deque<lambda_transducer_model*> unfinished_transducers;
  public:

    /**
     * \brief Create this guard when constructing a composed transducer with lambda syntax.
     * \param transducer The transducer that is being constructed.
     */
    explicit lambda_transducer_model_construction_guard(lambda_transducer_model* transducer);
    ~lambda_transducer_model_construction_guard();

    /**
     * \brief Get the current transducer being constructed.
     * \return
     */
    static lambda_transducer_model* top();

    static unsigned long size();
  };


  class lambda_transducer_model  {
    /**
     * \brief The depth of this transducer model (only applicable when defining using lambda syntax)
     *
     * When constructing transducer with the lambda syntax, a composed transducer can be defined within a composed transducer.
     *
     * This value indicates the level of "nesting" this current transducer is in. The top level transducer has depth=0,
     * and each nesting increases this value by 1.
     *
     */
    unsigned long nesting_depth_m{};

    unsigned long arity_m{};

    /**
     * \brief A list of local values that should be computed as intermediate steps when applying this transducer
     * Sorted in topological order, where the arguments to this transducer are at the front and the return value sits in the last.
     */
    std::vector<compute_value_behavior> locals_m;

    /**
     * \brief The value placeholder for return value
     */
    value_placeholder ret_m;

    /**
     * \brief A flag indicating whether this transducer has lazy operations within (for example, lazy_ifelse)
     *
     * If a transducer does not have lazy operations, the execution of the transducer can be done in sequential way, from the input to the output. Which is faster when compared with working all the way from the output to input backwards in the usual way.
     *
     */
    bool has_lazy_operation{false};

    friend value_placeholder;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(nesting_depth_m, arity_m, locals_m, ret_m, has_lazy_operation);
    }

    lambda_transducer_model() = default;
    lambda_transducer_model(const lambda_transducer_model&) = default;
    lambda_transducer_model(lambda_transducer_model&& x) noexcept = default;
    lambda_transducer_model& operator=(const lambda_transducer_model&) = default;
    lambda_transducer_model& operator=(lambda_transducer_model&&) noexcept = default;

    template<int ARITY, typename F>
    static lambda_transducer_model from_lambda_fn(const F& lambda_fn) {
      auto ret = lambda_transducer_model();
      ret.arity_m = ARITY;
      ret.nesting_depth_m = lambda_transducer_model_construction_guard::size();
      lambda_transducer_model_construction_guard _(&ret);

      std::vector<value_placeholder> input_placeholders;
      input_placeholders.reserve(ARITY);
      for(unsigned long i=0; i<ARITY; ++i) {
        input_placeholders.push_back(ret.make_value_placeholder_from_input());
      }

      ret.ret_m = std::apply(lambda_fn, generate_type_consistent_tuple<ARITY>(
        [&](unsigned long i) {
          return input_placeholders[i];
        }));

      return ret;
    }

    [[nodiscard]] unsigned long nesting_depth() const;

    [[nodiscard]] bool is_arity(unsigned long arity) const;

    [[nodiscard]] unsigned long num_locals() const;

    [[nodiscard]] std::string default_name() const;

    template<typename ...Args>
    value_t transduce(const Args&... args) {
      constexpr auto argc = sizeof...(args);
      if(argc != arity_m) {
        throw std::runtime_error("Argument count error");
      }

      lambda_transducer_value_cache _(this, args...);
      return evaluate_value_placeholder(ret_m);
    }

    std::vector<std::shared_ptr<transducer_variant>> nested_transducers();

    /**
     * \brief Create a value placeholder holding input (argument to this transducer)
     *
     * Input placeholders needs to be created before any other value placeholders for a transducer.
     *
     * \return The value placeholder holding input
     */
    value_placeholder make_value_placeholder_from_input();

    /**
     * \brief Create a value placeholder owned by this transducer, holding a constant value.
     * \param val The constant value
     * \return The created value placeholder
     */
    value_placeholder make_value_placeholder_from_constant(value_t val);

    /**
     * \brief Create a value placeholder owned by this transducer, that should be computed by invoking other transducers
     *
     * If the transducer to invoke is a TBD transducer, then this value placeholder will hold a weak pointer to the transducer to avoid memory leak.
     *
     * \param transducer The transducer to invoke
     * \param inputs The inputs to the transducer to invoke
     * \return The created value placeholder
     */
    template<typename ...Args>
    value_placeholder make_value_placeholder_from_transducing(std::shared_ptr<transducer_variant> transducer, Args ...inputs);

    /**
     * \brief Create a value placeholder that wraps its inputs into a list
     *
     * This is a special treatment for the make list operation so that this operation is not restricted by the maximum arity constraint.
     *
     * \param inputs
     * \return
     */
    value_placeholder make_value_placeholder_from_make_list(std::vector<value_placeholder> inputs);

    /**
     * \brief Evaluate a value placeholder (under the current transducing scope)
     * \param vp The value placeholder to evaluate
     * \return The evaluated value
     */
    static const value_t& evaluate_value_placeholder(const value_placeholder& vp);

  private:

    static value_t evaluate_compute_behavior(const compute_value_behavior& behavior);

    value_placeholder make_value_placeholder_impl(compute_value_behavior behavior);
  };


}


#endif //LEGO_LAMBDA_TRANSDUCER_MODEL_HPP
