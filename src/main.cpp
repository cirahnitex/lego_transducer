#include "include/lego_transducer.hpp"
#include <dynet/dynet.h>
#include "include/parallel_array_map.hpp"
#include "dynet_computation_graph.hpp"
using namespace std;
using namespace tg;



int main() {

  cout << "[INFO] initializing..." << endl;

  lego_initialize(512, {GPU_5, GPU_6});

  cout << "initialize complete" << endl;

  tg::value_t x(tensor_t({1,2,3}));

  transducer_model my_double([&](const value_placeholder& x) {
    return x + x;
  });

  cout << my_double(x) << endl;

  return 0;
}
