//
// Created by Dekai WU and YAN Yuchen on 20200506.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_LIST_OPERATIONS_IMPL_HPP
#define LEGO_LIST_OPERATIONS_IMPL_HPP

#include "include/transducer_typed_value.hpp"

namespace tg {

  class transducer_variant;


  class list_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
    }

    list_op() = default;

    list_op(const list_op&) = default;

    list_op(list_op&&) noexcept = default;

    list_op& operator=(const list_op&) = default;

    list_op& operator=(list_op&&) noexcept = default;

    std::string default_name() const {
      return "list";
    }

    template<typename ...Args>
    value_t transduce(const Args&... args) {
      return value_t(std::vector<value_t>{args...});
    }
  };

  class list_zip_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
    }

    list_zip_op() = default;

    list_zip_op(const list_zip_op&) = default;

    list_zip_op(list_zip_op&&) noexcept = default;

    list_zip_op& operator=(const list_zip_op&) = default;

    list_zip_op& operator=(list_zip_op&&) noexcept = default;


    std::string default_name() const;

    value_t apply(const std::vector<value_t>& ins);

    template<typename ...Args>
    value_t transduce(const Args&... args) {
      return apply(std::vector<value_t>{args...});
    }
  };


  class list_reverse_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    list_reverse_op() = default;

    list_reverse_op(const list_reverse_op&) = default;

    list_reverse_op(list_reverse_op&&) noexcept = default;

    list_reverse_op& operator=(const list_reverse_op&) = default;

    list_reverse_op& operator=(list_reverse_op&&) noexcept = default;

    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };


  class mappify_op {
    std::shared_ptr<transducer_variant> mapper_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar);

    mappify_op() = default;

    mappify_op(const mappify_op&) = default;

    mappify_op(mappify_op&&) noexcept = default;

    mappify_op& operator=(const mappify_op&) = default;

    mappify_op& operator=(mappify_op&&) noexcept = default;

    explicit mappify_op(std::shared_ptr<transducer_variant> mapper);

    template<typename ...Args>
    value_t transduce(const Args& ...ins);

    bool is_arity(unsigned long arity) const;

    std::string default_name() const;

    std::vector<std::shared_ptr<transducer_variant>> nested_transducers();

  };

  class list_reduce_op {
    std::shared_ptr<transducer_variant> reducer_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar);

    list_reduce_op() = default;

    list_reduce_op(const list_reduce_op&) = default;

    list_reduce_op(list_reduce_op&&) noexcept = default;

    list_reduce_op& operator=(const list_reduce_op&) = default;

    list_reduce_op& operator=(list_reduce_op&&) noexcept = default;

    explicit list_reduce_op(std::shared_ptr<transducer_variant> reducer);

    value_t transduce(const value_t& xs, const value_t& init_val);

    std::string default_name() const;

    std::vector<std::shared_ptr<transducer_variant>> nested_transducers();


  };

  class list_filter_op {
    std::shared_ptr<transducer_variant> filter_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar);

    list_filter_op() = default;
    list_filter_op(const list_filter_op&) = default;
    list_filter_op(list_filter_op&&) noexcept = default;
    list_filter_op& operator=(const list_filter_op&) = default;
    list_filter_op& operator=(list_filter_op&&) noexcept = default;

    explicit list_filter_op(std::shared_ptr<transducer_variant> filter);

    value_t transduce(const value_t& xs);

    std::string default_name() const;

    std::vector<std::shared_ptr<transducer_variant>> nested_transducers();
  };

  class list_select_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    list_select_op() = default;

    list_select_op(const list_select_op&) = default;

    list_select_op(list_select_op&&) noexcept = default;

    list_select_op& operator=(const list_select_op&) = default;

    list_select_op& operator=(list_select_op&&) noexcept = default;

    value_t transduce(const value_t& in0, const value_t& in1);

    std::string default_name() const;
  };

  class list_select_many_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    list_select_many_op() = default;

    list_select_many_op(const list_select_many_op&) = default;

    list_select_many_op(list_select_many_op&&) noexcept = default;

    list_select_many_op& operator=(const list_select_many_op&) = default;

    list_select_many_op& operator=(list_select_many_op&&) noexcept = default;

    value_t transduce(const value_t& list, const value_t& indices);


    std::string default_name() const;
  };

  class list_slice_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    list_slice_op() = default;

    list_slice_op(const list_slice_op&) = default;

    list_slice_op(list_slice_op&&) noexcept = default;

    list_slice_op& operator=(const list_slice_op&) = default;

    list_slice_op& operator=(list_slice_op&&) noexcept = default;

    value_t transduce(const value_t& list, const value_t& start, const value_t& end);


    std::string default_name() const;
  };

  class list_push_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    list_push_op() = default;

    list_push_op(const list_push_op&) = default;

    list_push_op(list_push_op&&) noexcept = default;

    list_push_op& operator=(const list_push_op&) = default;

    list_push_op& operator=(list_push_op&&) noexcept = default;

    value_t transduce(const value_t& in0, const value_t& in1);

    std::string default_name() const;
  };

  class list_size_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class list_empty_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };


  class list_concat_op {
  public:

    template<typename Archive>
    void serialize(Archive& ar) {
    }
    list_concat_op() = default;
    list_concat_op(const list_concat_op&) = default;
    list_concat_op(list_concat_op&&) noexcept = default;
    list_concat_op& operator=(const list_concat_op&) = default;
    list_concat_op& operator=(list_concat_op&&) noexcept = default;

    template<typename ...Args>
    value_t transduce(const Args&... args) {
      constexpr unsigned long argc = sizeof...(Args);
      if constexpr (argc == 0) return value_t::make_list();
      if constexpr (argc == 1) return std::get<0>(std::forward_as_tuple(args...));

      std::initializer_list<value_t> ins{args...};
      std::vector<value_t> ret;

      for(auto&& in:ins) {
        auto&& i_list = in.as_list();
        ret.insert(ret.end(), i_list.begin(), i_list.end());
      }

      return value_t(ret);
    }

    std::string default_name() const {
      return "list_concat";
    }
  };
}

#endif //LEGO_LIST_OPERATIONS_IMPL_HPP
