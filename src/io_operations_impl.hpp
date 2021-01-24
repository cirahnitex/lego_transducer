//
// Created by Dekai WU and YAN Yuchen on 20200511.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_IO_OPERATIONS_IMPL_HPP
#define LEGO_IO_OPERATIONS_IMPL_HPP
#include "include/transducer_typed_value.hpp"

namespace tg {
  class trace_op {
    std::string prefix_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(prefix_m);
    }
    trace_op() = default;
    trace_op(const trace_op&) = default;
    trace_op(trace_op&&) noexcept = default;
    trace_op& operator=(const trace_op&) = default;
    trace_op& operator=(trace_op&&) noexcept = default;
    explicit trace_op(std::string prefix);

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class identity_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
    }
    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };
}


#endif
