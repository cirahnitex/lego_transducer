//
// Created by Dekai WU and YAN Yuchen on 20201127.
//

#include "../include/lego_transducer.hpp"
using namespace std;
using namespace tg;

int main() {
  lego_initialize();


  transducer_model find_maximum;
  find_maximum = transducer_model([&](const value_placeholder& list)->value_placeholder {
    return lazy_ifelse(
      list_size(list) == value_placeholder::constant(1),
      list[0],
      cmax(list[0], find_maximum(list_slice(list,1, -1))));
  });

  cout << find_maximum(value_t::make_list(1, 2,4,3,5)) << endl;

  return 0;
}

