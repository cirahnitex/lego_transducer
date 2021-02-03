//
// Created by Dekai WU and YAN Yuchen on 20200506.
//

#ifndef LEGO_LEGO_LIST_OPERATIONS_HPP
#define LEGO_LEGO_LIST_OPERATIONS_HPP
#include "value_placeholder.hpp"
#include "transducer_typed_value.hpp"
#include "transducer_model.hpp"
namespace tg {

  /**
   * \addtogroup list_operations
   * @{
   */


  /**
   * \brief Wrap one or more items in a list
   *
   * Actually you could give zero items to make an empty list.
   *
   * \tparam value_placeholder_T = value_placeholder
   * \param args Items to be wrapped in a list
   * \return The list
   */
  extern transducer_model make_list;


  /**
   * \brief Zip multiple lists of the same length
   *
   * Similar to how zip works in python.
   * It returns a list of lists, where the first item in each passed lists is paired together, and then the second item in each passed lists are paired together etc.
   *
   * \tparam value_placeholder_T = value_placeholder
   * \param args each argument is a list
   * \return the zipped list.
   */
  extern transducer_model list_zip;

  /**
   * \brief Reverse a list.
   * \param list_to_reverse The list to be reversed.
   * \return The reversed list.
   */
  extern transducer_model list_reverse;

  /**
   * \brief The internal helper to implement tg::list_map
   *
   * It takes a mapper transducer and returns a transducer that can perform list map on a list.
   */
  transducer_model _mappify(const transducer_model& mapper_fn);

  /**
   * \brief Performs a list map
   *
   * Performs a functional programming <b>map</b>. It calls your provided transducer on every element in your provided list.
   *
   * Performing list map of higher arity is also supported.
   *   - If you provide a binary function, you need to provided two lists.
   *   - If you provide a ternary function, you need to provide three lists.
   *
   * \param mapper_fn The transducer to apply to every element in the list.
   * \param args The lists to map on
   * \return The returned values of applying the transducer to every item in the list
   */
  template<typename ...Args>
  decltype(auto) list_map(const transducer_model& mapper_fn, Args... lists) {
    return _mappify(mapper_fn)(lists...);
  }


  /**
   * \brief The internal helper to implement tg::list_reduce
   *
   * It takes a reducer and returns a transducer that can perform reduce operation on a list.
   */
  transducer_model _reducify(const transducer_model& reducer);

  /**
   * \brief Applies a aggregation function on every item in a list in order.
   * \param xs the list whose items will be applied aggregation function on.
   * \param reducer the aggregation function that
   *                * takes (1) the previous aggregated value and (2) the current list item
   *                * returns the new aggregated value
   * \param init_value the initial value before any aggregation happens.
   * \return A model which outputs the final aggregated value
   */
  value_placeholder list_reduce(const transducer_model& reducer, const value_placeholder& xs, const value_placeholder& init_value);
  value_t list_reduce(const transducer_model& reducer, const value_t& xs, const value_t& init_value);

  /**
   * \brief The internal helper to implement tg::list_filter
   *
   * It takes a filter transducer and returns a transducer that can perform list filter operation on a list.
   */
  transducer_model _filterify(const transducer_model& filter);

  /**
   * \brief Selects all elements in a list that passes a filter
   *
   * The filter is a transducer, that takes a list element and returns true if the test is passed.
   *
   * \param filter The filter transducer to test the elements.
   * \param xs The original list to be filtered
   * \return A new list containing all elements that passes the filter
   */
  value_placeholder list_filter(const transducer_model& filter, const value_placeholder& xs);
  value_t list_filter(const transducer_model& filter, const value_t& xs);

  /**
   * \brief Append a value at the end of a list
   * \param list The list to append value to
   * \param val The value to append
   * \return The appended list
   */
  extern transducer_model push_back;

  /**
   * \brief Concatenate multiple lists into a list
   * \param lists The lists to concatenate
   * \return The concatenated list
   */
  extern transducer_model list_concat;

  /**
   * \brief Get the size of a list
   * \param list The list to compute size
   * \return The list size
   */
  extern transducer_model list_size;

  /**
   * \brief Check if a list is empty
   * \param list The list to check if is empty
   * \return 1 if the list is empty, 0 otherwise
   */
  extern transducer_model list_is_empty;

  /**
   * \brief Selects a consecutive slice from a list
   * \param list The list to slice from
   * \param start The starting index (inclusive)
   * \param end The ending index (exclusive)
   * \return The slice of the list
   */
  value_placeholder list_slice(const value_placeholder& list, unsigned long start, unsigned long end);
  value_t list_slice(const value_t& list, unsigned long start, unsigned long end);
  value_placeholder list_slice(const value_placeholder& list, const value_placeholder& start, const value_placeholder& end);

  /**
   * \brief Selects multiple items from a list
   * \param list The list to select from
   * \param indices The indices of the items you wish to select
   * \return A list containing selected items
   */
  value_placeholder list_select_many(const value_placeholder& list, const std::vector<unsigned long>& indices);
  value_t list_select_many(const value_t& list, const std::vector<unsigned long>& indices);
  value_placeholder list_select_many(const value_placeholder& list, const value_placeholder& indices);
  /// @}
}

#endif //LEGO_LEGO_LIST_OPERATIONS_HPP
