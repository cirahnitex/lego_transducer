//
// Created by Dekai WU and YAN Yuchen on 20200714.
//
/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_TRANSDUCER_VARIANT_HPP
#define LEGO_TRANSDUCER_VARIANT_HPP

#include <variant>
#include "tensor_operations_impl.hpp"
#include "class_traits_helper.cmacro.hpp"
#include "dense_model.hpp"
#include "bilinear_model.hpp"
#include "boolean_operations_impl.hpp"
#include "lambda_transducer_model.hpp"
#include "const_value_model.hpp"
#include "dropout_model.hpp"
#include "embedding_table_model.hpp"
#include "generic_rnn_model.hpp"
#include "io_operations_impl.hpp"
#include "list_operations_impl.hpp"
#include "symbol_id_converter_model.hpp"
#include "tbd_transducer.hpp"
#include "composed_transducer_model.hpp"

namespace tg {

  NULLARY_MEMBER_FUNCTION_DETECTOR(is_arity_0, transduce, value_t)
  MEMBER_FUNCTION_DETECTOR(is_arity_1, transduce, value_t, const value_t&)
  MEMBER_FUNCTION_DETECTOR(is_arity_2, transduce, value_t, const value_t&, const value_t&)
  MEMBER_FUNCTION_DETECTOR(is_arity_3, transduce, value_t, const value_t&, const value_t&, const value_t&)
  MEMBER_FUNCTION_DETECTOR(is_arity_4, transduce, value_t, const value_t&, const value_t&, const value_t&, const value_t&)
  MEMBER_FUNCTION_DETECTOR(is_arity_5, transduce, value_t, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&)
  MEMBER_FUNCTION_DETECTOR(is_arity_6, transduce, value_t, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&)
  MEMBER_FUNCTION_DETECTOR(is_arity_7, transduce, value_t, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&)
  MEMBER_FUNCTION_DETECTOR(is_arity_8, transduce, value_t, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&)
  template<typename T>
  struct has_dynamic_arity {
    template<typename U>
    static constexpr bool
    test(std::void_t<decltype(std::declval<U>().is_arity(0))>*) {return true;}

    template<typename U>
    static constexpr bool
    test(...) {return false;}

    static constexpr bool value = test<T>(nullptr);
  };

  template<typename T>
  inline constexpr bool has_dynamic_arity_v = has_dynamic_arity<T>::value;



  MEMBER_FUNCTION_DETECTOR(is_ternary_lazy, lazy_transduce, value_t, const std::function<value_t()>&, const std::function<value_t()>&, const std::function<value_t()>&)
  
  MEMBER_FUNCTION_DETECTOR(has_nested_transducers, nested_transducers, std::vector<std::shared_ptr<transducer_variant>>)

  class transducer_variant :public std::enable_shared_from_this<transducer_variant> {
  public:
    using variant_t = std::variant<
      tensor_concat_op, tensor_concat_op_static_axis, split_op, max_index_of_tensor1d_op, log_softmax_op, neg_op, minus_op, add_op, matmult_op, divide_op, tensor_pick_op, tensor_slice_op, tanh_op, relu_op, leaky_relu_op, elu_op, selu_op, gelu_op,
      sigmoid_op, cmult_op, sqrt_op, pow_op,
      list_sum_op, list_max_op, binary_max_op, binary_min_op, list_min_op, pickneglogsoftmax_op, tensor_sum_op, tensor_average_op, tensor_std_op, tensor_axis_sum_op, tensor_axis_average_op, tensor_axis_std_op, tensor_axis_max_op, tensor_axis_min_op, random_normal_op, random_uniform_op, random_bernoulli_op, pickneglogsigmoid_op, log_op,

      bilinear_model, lazy_ifelse_op, eager_ifelse_op,
      to_boolean_op, soft_ifelse_op, eq_op, ne_op, lt_op, gt_op, logical_and_op, logical_or_op, logical_not_op,

      lambda_transducer_model,

      const_value_model,

      dense_model, n_ary_dense_model,

      dropout_model, axis_synchronized_dropout_model,

      embedding_table_model,

      generic_rnn_model, generic_stacked_rnn_model, generic_bidirectional_rnn_model, generic_stacked_bidirectional_rnn_model,

      trace_op, identity_op,

      list_op, list_zip_op, list_reverse_op, mappify_op, list_reduce_op, list_select_op, list_select_many_op, list_slice_op, list_push_op, list_size_op, list_empty_op,

      dict_model,

      tbd_transducer,

      minus_scalar_op<float>, minus_scalar_op<long>, scalar_minus_op<float>, scalar_minus_op<long>, add_scalar_op<float>, add_scalar_op<long>, cmult_scalar_op<float>, cmult_scalar_op<long>, divide_scalar_op<float>, divide_scalar_op<long>,

      biaffine_model,

      n_ary_concat_op_static_axis, list_concat_op, composed_transducer_model, list_filter_op, tensor_reshape_op, tensor_transpose_op, softmax_op, exp_op, tensor_l2_norm_op
    >;

    variant_t v;

    /**
     * \brief Stores the display name defined by the user
     */
    std::string user_defined_display_name;

    template<typename Archive>
    void serialize(Archive& ar) {
      ar(v, user_defined_display_name);
    }

    transducer_variant() = default;
    transducer_variant(const transducer_variant& x) = default;
    transducer_variant(transducer_variant&& x) noexcept = default;
    transducer_variant& operator=(const transducer_variant&) = default;
    transducer_variant& operator=(transducer_variant&&) noexcept = default;

    template<typename T>
    explicit transducer_variant(T t): v(std::move(t)) {}

    struct transducer_traits_t {

      bool has_dynamic_arity;
      std::tuple<bool,bool,bool,bool,bool,bool,bool,bool,bool> static_arities;
      bool is_lazy;

      constexpr transducer_traits_t(bool has_dynamic_arity, std::tuple<bool,bool,bool,bool,bool,bool,bool,bool,bool> static_arities, bool is_lazy) :
        has_dynamic_arity(has_dynamic_arity), static_arities(static_arities), is_lazy(is_lazy) {

      }

    };

    template<typename transducer_T>
    static constexpr transducer_traits_t transducer_traits() {
      using T = std::decay_t<transducer_T>;

      constexpr bool _is_ternary_lazy = is_ternary_lazy_v<T>;

      if constexpr (_is_ternary_lazy) {
        return transducer_traits_t(false, std::make_tuple(false, false, false, true, false, false, false, false, false), true);
      }
      else {
        return transducer_traits_t(has_dynamic_arity_v<T>, std::make_tuple(
          is_arity_0_v<T>, is_arity_1_v<T>,is_arity_2_v<T>,is_arity_3_v<T>,
          is_arity_4_v<T>, is_arity_5_v<T>, is_arity_6_v<T>,is_arity_7_v<T>,is_arity_8_v<T>
        ), false);
      }
    }

    value_t _apply(const std::vector<value_t>& ins);

    template<typename ...Args>
    value_t transduce(const Args& ...args);

    template<typename ...Args>
    value_t transduce_placeholder(const Args& ...args);


    std::string name() const;

    void rename(const std::string& name);

    bool is_arity(unsigned long arity) const;

    template<typename _Visitor>
    constexpr decltype(auto) visit(_Visitor&& visitor) {
      return std::visit(visitor, v);
    }

    template<typename _Visitor>
    constexpr decltype(auto) visit(_Visitor&& visitor) const {
      return std::visit(visitor, v);
    }

    /**
     * \brief Find a (possibly nested) transducer by a given name.
     *
     * Search into the topology of this transducer, find the transducer by a given name.
     *
     * A custom name can be assigned via rename()
     *
     * If there are multiple matches, only the first match will be returned.
     *
     * \param name The name to search for
     * \return the transducer with the given name. nullptr if not found.
     */
    std::shared_ptr<transducer_variant> find_transducer_by_name(const std::string& name);
    
    /**
     * \brief List the directly nested transducers in this transducer.
     * 
     * A transducers may have nested transducers. For example, a composed transducer have the self transducer and a list of operand transducers; a mappified transducer have the transducer to be mapped.
     * 
     * Querying the child transducers is a subroutine for searching and aggregating, like search a transducer by name or aggregate all the parameter norms.
     *
     * This function only lists the direct child of this transducer. You may need to recursively go deeper if you want to find all the descendants.
     * 
     * \return The list of transducer that is directly nested in this transducer.
     */
    std::vector<std::shared_ptr<transducer_variant>> nested_transducers();

  };


}



#endif //LEGO_TRANSDUCER_VARIANT_HPP
