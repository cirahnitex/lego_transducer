//
// Created by Dekai WU and YAN Yuchen on 20200515.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_DYNET_COMPUTATION_GRAPH_HPP
#define LEGO_DYNET_COMPUTATION_GRAPH_HPP
#include <dynet/dynet.h>
namespace tg {
  /**
   * Manages a global dynet::ComputationGraph object singleton.
   *
   * A dynet::ComputationGraph is where all symbolic tensors (dynet::Expression) are stored.
   * dynet::ComputationGraph has some design flaws, this class provides bandaid fixes:
   *
   * ** FLAW #1 **
   * Dynet framework does not support multiple dynet::ComputationGraph instances,
   * but it does not provide interfaces to access it as a singleton.
   * Instead it relies on the user to have only one dynet::ComputationGraph at once.
   * This class stores a global singleton and provides singleton accessor.
   *
   * ** FLAW #2 **
   * Also, the dynet::ComputationGraph::renew() is bugged, it does not reset itself correctly.
   * So instead of calling dynet::ComputationGraph::renew(),
   * we must destroy the old graph and create a new one.
   *
   * This class also takes care of that.
   */
  class dynet_computation_graph {
    thread_local static dynet::ComputationGraph* pcg;
  public:

    /**
     * \brief Get the singleton dynet::ComputationGraph instance.
     * \return the pointer to the dynet::ComputationGraph instance.
     */
    static dynet::ComputationGraph* p();


    /**
     * \brief Discard the computation graph, invalidating all symbolic tensors.
     *
     * Please call value_t::evaluate() to evaluate all symbolic tensors that you need before discarding the computation graph.
     */
    static void discard();

  };
}


#endif
