//
// Created by Dekai WU and YAN Yuchen on 20200526.
//

#ifndef LEGO_GENERATE_TYPE_CONSISTENT_TUPLE_HPP
#define LEGO_GENERATE_TYPE_CONSISTENT_TUPLE_HPP
#include <tuple>

namespace _private_gAiBnCTRmK {
  template<typename F, size_t... Is>
  static auto gen_tuple_impl(F func, std::index_sequence<Is...>) {
    return std::make_tuple(func(Is)...);
  }
}

namespace tg {

  /**
   * \ingroup utilities
   *
   * \brief A helper function to generate tuples whose items are of the same type.
   *
   * Specifically, it returns the tuple <f(0), f(1), ..., f(N-1)> for some N that is a constant-expression.
   * For example, this is how you can make a tuple of the first 10 squares: 0, 1, 4, ..., 81
   *
   *     auto squares = gen_tuple<10>([](size_t i){ return i*i;});
   *
   * \param f the transformation function that takes the tuple element index and returns the tuple element
   * \return the tuple whose elements are <f(0), f(1), ..., f(N-1)>
   */
  template<size_t N, typename F>
  static auto generate_type_consistent_tuple(F f) {
    return _private_gAiBnCTRmK::gen_tuple_impl(f, std::make_index_sequence<N>{});
  }

  /**
   * \ingroup utilities
   * \brief Get the following type std::tuple<T, T, ..., T>
   * \tparam N The number of elements the tuple holds
   * \tparam T The type that the tuple holds
   */
  template <size_t N, typename T>
  class type_consistent_tuple {
    template <typename = std::make_index_sequence<N>>
    struct impl;

    template <size_t... Is>
    struct impl<std::index_sequence<Is...>> {
      template <size_t >
      using wrap = T;

      using type = std::tuple<wrap<Is>...>;
    };

  public:
    using type = typename impl<>::type;
  };

  template <size_t N, typename T>
  using type_consistent_tuple_t = typename type_consistent_tuple<N, T>::type;
}

#endif //LEGO_GENERATE_TYPE_CONSISTENT_TUPLE_HPP
