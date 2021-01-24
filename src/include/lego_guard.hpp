//
// Created by Dekai WU and YAN Yuchen on 20190927.
//

#ifndef LEGO_GUARD_MACRO_HPP
#define LEGO_GUARD_MACRO_HPP

#define DEFINE_THREAD_LOCAL_GUARD(guard_name) \
class guard_name { \
  static thread_local volatile unsigned num_instances; \
public: \
  static bool is_guarded() { \
    return num_instances > 0; \
  } \
  guard_name() {num_instances++;} \
  guard_name(const guard_name&) = delete; \
  guard_name(guard_name&&) noexcept = delete; \
  guard_name &operator=(const guard_name&) = delete; \
  guard_name &operator=(guard_name&&) noexcept = delete; \
  ~guard_name() {num_instances--;} \
}; \

#define DEFINE_THREAD_LOCAL_GUARD_IMPL(guard_name) thread_local volatile unsigned guard_name::num_instances = 0;
namespace tg {

  /**
   * \addtogroup global_configurations
   * @{
   */

  /**
   * \brief A global guard indicates that whether a backprop trainer is performing training.
   *
   * This guard is consumed by transducers who cares about whether the model is training or not, such as a dropout layer.
   *
   */
  DEFINE_THREAD_LOCAL_GUARD(lego_training_guard)

  /**
   * \brief When guarded, shows the time it takes to (1) construct the CG and (2) execute CG
   * \internal
   */
  DEFINE_THREAD_LOCAL_GUARD(show_cg_construction_time_guard)

  /**
   * \brief When guarded, all symbolic tensor computations are evaluated immediately.
   *
   * Upon every evaluation, if NaN or Inf value occurs, an error will be thrown.
   *
   * This is used to narrow down where NaN and Inf values are originated from.
   * TODO: implement the usage of this NaN debugging
   */
  DEFINE_THREAD_LOCAL_GUARD(immediate_computation_guard)
  /// @}
}


#endif //LEGO_GUARD_MACRO_HPP
