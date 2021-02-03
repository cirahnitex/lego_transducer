#include "include/lego_transducer.hpp"
#include <dynet/dynet.h>
#include "include/parallel_array_map.hpp"
using namespace std;
using namespace tg;



int main() {
  lego_initialize();

  value_t x(tensor_t({1,2,3,4,5,6}, {1, 6})); // value holding a 1x6 tensor
  value_t y = tensor_reshape(x, {2, -1}); // a value holding a 2x3 tensor

  return 0;
}
