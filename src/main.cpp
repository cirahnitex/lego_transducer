#include "include/lego_transducer.hpp"
#include <dynet/dynet.h>
#include "include/parallel_array_map.hpp"
#include "dynet_computation_graph.hpp"
using namespace std;
using namespace tg;



int main() {

  cout << "[INFO] initializing..." << endl;

  lego_initialize(512, {CPU});

  cout << "initialize complete" << endl;


  auto numerical_encoder = make_numerical_encoder(0, 10, 1, 8);

  transducer_model distort_and_reconstruct([&](const value_placeholder& x)->value_placeholder {
    return numerical_encoder(x + random_normal({8}, 0, 0.1));
  });

  for(unsigned i = 0; i< 10; ++i) {
    tg::value_t x(tensor_t({(float)i}));

    auto embedding = numerical_encoder(x);
    cout << embedding << endl;
    auto reconstructed = distort_and_reconstruct(embedding);
    cout << reconstructed << endl;
  }



  return 0;
}
