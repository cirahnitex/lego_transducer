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


  auto numerical_encoder = make_numerical_encoder(10, 1, 8);

  for(unsigned i = 0; i< 10; ++i) {
    tg::value_t x(tensor_t({(float)i}));

    cout << numerical_encoder(x) << endl;
  }



  return 0;
}
