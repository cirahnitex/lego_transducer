//
// Created by Dekai WU and YAN Yuchen on 20201126.
//
/// \cond SHOW_INTERNAL_IMPL
#ifndef LEGO_TRANSDUCE_TIME_SCOPE_HPP_
#define LEGO_TRANSDUCE_TIME_SCOPE_HPP_
#include "include/value_placeholder.hpp"
namespace tg {
  class lambda_transducer_model;
  /**
   * \brief Caches the computed local values of a lambda transducer.
   *
   * There is a global stack holding all instances of tg::lambda_transducer_value_cache in the current callstack, instead of each transducer holding its own. This is because, the same transducer may be invoked multiple times in the call stack (for example, in recursion). Thus, it is ambiguous for a transducer to hold its own value cache.
   *
   */
  struct lambda_transducer_value_cache {
    /**
     * \brief Point to the caller transducer
     */
    lambda_transducer_value_cache* parent;

    /**
     * \brief Points to this callee transducer
     */
    lambda_transducer_model* transducer;

    /**
     * \brief The computed local values for this callee transducer
     */
    std::vector<std::pair<bool, value_t>> values_cache;

    /**
     * \brief Globally points to the cache of the current transducer being invoked.
     */
    static thread_local lambda_transducer_value_cache* top;

    /**
     * \brief Need to create this scope guard before transducing
     * \param transducer
     * \param args The values that gets passed to this transducer as inputs
     */
    template<typename ...value_T>
    lambda_transducer_value_cache(lambda_transducer_model* transducer, value_T ...args):
    parent(top), transducer(transducer),
    values_cache(std::initializer_list<std::pair<bool, value_t>>{std::make_pair(true, std::move(args))...})
    {
      resize_values_cache_to_transducer_num_locals();
      top = this;
    }

    ~lambda_transducer_value_cache();

    static lambda_transducer_value_cache* get_scope_by_nesting_depth(unsigned long nesting_depth);

  private:
    void resize_values_cache_to_transducer_num_locals();
  };


}


#endif
