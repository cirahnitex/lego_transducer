#include "include/lego_transducer.hpp"
#include <dynet/dynet.h>
#include "include/parallel_array_map.hpp"
using namespace std;
using namespace tg;


int main() {
  lego_initialize();

transducer_model add_one([&](const value_placeholder& x)->value_placeholder {
  return x + 1;
});

transducer_model list_add_one([&](const value_placeholder& xs)->value_placeholder {
  return list_map(add_one, xs);
});

cout << list_add_one(make_list(1,2,3)) << endl; // 4 5 6

  return 0;
}
