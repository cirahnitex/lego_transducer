//
// Created by Dekai WU and YAN Yuchen on 20200506.
//

#include "include/lego_list_operations.hpp"
#include "list_operations_impl.hpp"
#include "transducer_variant.hpp"
#include "include/transducer_model.hpp"

using namespace std;
using namespace tg;


transducer_model tg::_mappify(const transducer_model& mapper_fn) {
  return transducer_model(make_shared<transducer_variant>(mappify_op(mapper_fn._get_impl())));
}

transducer_model tg::_reducify(const transducer_model& reducer) {
  return transducer_model(make_shared<transducer_variant>(list_reduce_op(reducer._get_impl())));
}

value_t tg::list_reduce(const transducer_model& reducer, const value_t& xs, const value_t& init_value) {
  return _reducify(reducer)(xs, init_value);
}

value_placeholder tg::list_reduce(const transducer_model& reducer, const value_placeholder& xs,
                                  const value_placeholder& init_value) {
  return _reducify(reducer)(xs, init_value);
}

transducer_model tg::_filterify(const transducer_model& filter) {
  return transducer_model(make_shared<transducer_variant>(list_filter_op(filter._get_impl())));
}

value_placeholder tg::list_filter(const transducer_model& filter, const value_placeholder& xs) {
  return _filterify(filter)(xs);
}

value_t tg::list_filter(const transducer_model& filter, const value_t& xs) {
  return _filterify(filter)(xs);
}

namespace tg {
  transducer_model push_back(make_shared<transducer_variant>(list_push_op()));
  transducer_model list_reverse(make_shared<transducer_variant>(list_reverse_op()));
  transducer_model make_list(make_shared<transducer_variant>(list_op()));
  transducer_model list_zip(make_shared<transducer_variant>(list_zip_op()));
  transducer_model list_concat(make_shared<transducer_variant>(list_concat_op()));
  transducer_model list_size(make_shared<transducer_variant>(list_size_op()));
  transducer_model list_is_empty(make_shared<transducer_variant>(list_empty_op()));
}

value_placeholder tg::list_slice(const value_placeholder& list, const value_placeholder& start,
                                 const value_placeholder& end) {
  static transducer_model model(make_shared<transducer_variant>(list_slice_op()));
  return model(list, start, end);
}

value_placeholder tg::list_slice(const value_placeholder& list, unsigned long start, unsigned long end) {
  return list_slice(list, value_placeholder::constant(start), value_placeholder::constant(end));
}
value_t tg::list_slice(const value_t& list, unsigned long start, unsigned long end) {
  return list.slice(start, end);
}

value_placeholder tg::list_select_many(const value_placeholder& list, const value_placeholder& indices) {
  static transducer_model op(make_shared<transducer_variant>(list_select_many_op()));
  return op(list, indices);
}

value_placeholder tg::list_select_many(const value_placeholder& list, const std::vector<unsigned long>& indices) {
  return list_select_many(list, value_placeholder::constant(indices));
}

value_t tg::list_select_many(const value_t& list, const std::vector<unsigned long>& indices) {
  return list.select_many(indices);
}

