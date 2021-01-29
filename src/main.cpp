#include "include/lego_transducer.hpp"
#include <dynet/dynet.h>
#include "include/parallel_array_map.hpp"
using namespace std;
using namespace tg;


int main() {
  lego_initialize();

  value_t x(tensor_t({1,2,3}));
  value_t y(tensor_t({0,0,0}));

  cout << tensor_squared_distance(x, y) << endl;

  return 0;
}
