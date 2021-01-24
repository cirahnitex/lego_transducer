//
// Created by Dekai WU and YAN Yuchen on 20201126.
//

#include "lambda_transducer_value_cache.hpp"
#include "transducer_variant.hpp"
#include "include/wallclock_timer.hpp"
using namespace tg;
using namespace std;

thread_local lambda_transducer_value_cache* lambda_transducer_value_cache::top = nullptr;

lambda_transducer_value_cache::~lambda_transducer_value_cache() {
  top = top->parent;
}



lambda_transducer_value_cache *lambda_transducer_value_cache::get_scope_by_nesting_depth(unsigned long depth) {
  lambda_transducer_value_cache* ret = top;
  while(ret != nullptr) {
    if(ret->transducer->nesting_depth() == depth) {

      return ret;
    }
    ret = ret->parent;
  }
  return nullptr;
}

void lambda_transducer_value_cache::resize_values_cache_to_transducer_num_locals() {
  values_cache.resize(transducer->num_locals());
}

template<class> inline constexpr bool always_false_v = false;
