//
// Created by Dekai WU and YAN Yuchen on 20200515.
//

#include "dynet_computation_graph.hpp"
#include "include/transducer_typed_value.hpp"

using namespace tg;
using namespace std;

thread_local dynet::ComputationGraph* tg::dynet_computation_graph::pcg{};

dynet::ComputationGraph* tg::dynet_computation_graph::p() {
  if(!pcg) {
    pcg = new dynet::ComputationGraph();

    // dynet may run into weird "double free or corruption" crash when executing activation functions on tensor input nodes
    // for example,
    //     std::vector<float> arr{1,2,3,4};
    //     dynet::ComputationGraph cg;
    //     auto x = dynet::input(cg, {4}, &arr);
    //     x = dynet::tanh(x);
    //     cg.incremental_forward(x);
    // the above code will cause "double free or corruption".
    //
    // However, this crash does not trigger when there are some other dynet::Expression in the CG
    // So, we always add some dummy nodes in the CG, the sole purpose for these dummy node are to avoid the crash.
    volatile auto _ = dynet::zeros(*pcg, {1}) + 1;
  }
  return pcg;
}


void dynet_computation_graph::discard() {
  if(!pcg) return;
  delete pcg;
  pcg = nullptr;
}

