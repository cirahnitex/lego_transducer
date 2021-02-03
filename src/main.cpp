#include "include/lego_transducer.hpp"
#include <dynet/dynet.h>
#include "include/parallel_array_map.hpp"
using namespace std;
using namespace tg;



int main() {
  lego_initialize();


  transducer_model tmp([&](){

    auto x = value_placeholder::constant(tensor_t({1,2,3,4,5,6}, {2,3}));

    return tensor_reshape(x, {1, -1});
  });

  cout << tmp() << endl;

  return 0;
}
