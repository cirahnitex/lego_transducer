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


  value_t x(tensor_t({-1,2,3}));

  cout << tg::abs(x) << endl;



  return 0;
}
