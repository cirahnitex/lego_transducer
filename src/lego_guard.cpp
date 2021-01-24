//
// Created by Dekai WU and YAN Yuchen on 20201203.
//

#include "include/lego_guard.hpp"

namespace tg {
  DEFINE_THREAD_LOCAL_GUARD_IMPL(lego_training_guard)
  DEFINE_THREAD_LOCAL_GUARD_IMPL(show_cg_construction_time_guard)
  DEFINE_THREAD_LOCAL_GUARD_IMPL(immediate_computation_guard)
}

