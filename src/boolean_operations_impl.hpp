//
// Created by Dekai WU and YAN Yuchen on 20200627.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_BOOLEAN_OPERATIONS_IMPL_HPP
#define LEGO_BOOLEAN_OPERATIONS_IMPL_HPP

#include "include/transducer_typed_value.hpp"

namespace tg {

  class to_boolean_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {}

    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };

  class lazy_ifelse_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {}

    value_t lazy_transduce(const std::function<value_t()>& in0, const std::function<value_t()>& in1,
                      const std::function<value_t()>& in2);

    std::string default_name() const;
  };

  class eager_ifelse_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
    }

    value_t transduce(const value_t& in0, const value_t& in1, const value_t& in2);

    std::string default_name() const;
  };

  class soft_ifelse_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1, const value_t& in2);

    std::string default_name() const;
  };

  class eq_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);

    std::string default_name() const;
  };

  class ne_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);


    std::string default_name() const;
  };

  class lt_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);


    std::string default_name() const;
  };


  class gt_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);


    std::string default_name() const;
  };


  class logical_and_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);


    std::string default_name() const;
  };

  class logical_or_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);


    std::string default_name() const;
  };

  class logical_not_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };
}

#endif //LEGO_BOOLEAN_OPERATIONS_IMPL_HPP
