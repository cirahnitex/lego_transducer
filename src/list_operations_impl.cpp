//
// Created by Dekai WU and YAN Yuchen on 20200506.
//

#include "list_operations_impl.hpp"
#include "include/transducer_instance.hpp"
#include "include/parallel_array_map.hpp"
#include "dynet_computation_graph.hpp"
#include "transducer_variant.hpp"

using namespace std;
using namespace tg;


string mappify_op::default_name() const {
  return "mappified";
}

bool mappify_op::is_arity(unsigned long arity) const {
  return mapper_m->is_arity(arity);
}

mappify_op::mappify_op(std::shared_ptr<transducer_variant> mapper): mapper_m(move(mapper)) {

}

template<typename ...Args>
value_t mappify_op::transduce(const Args& ...ins) {

  constexpr auto argc = sizeof...(ins);

  if constexpr(argc == 0) {
    throw std::runtime_error("Cannot invoke nullary mappify");
  }
  else {
    unsigned long length = std::max({ins.as_list().size()...});

    if(length == 0) return value_t(vector<value_t>());

    vector<value_t> ys;
    ys.reserve(length);
    for (unsigned long i=0; i<length; ++i) {
      ys.push_back(mapper_m->transduce(ins.as_list()[i]...));
    }

    return value_t(ys);
  }
}

template value_t mappify_op::transduce();
template value_t mappify_op::transduce(const value_t&);
template value_t mappify_op::transduce(const value_t&, const value_t&);
template value_t mappify_op::transduce(const value_t&, const value_t&, const value_t&);
template value_t mappify_op::transduce(const value_t&, const value_t&, const value_t&, const value_t&);
template value_t mappify_op::transduce(const value_t&, const value_t&, const value_t&, const value_t&, const value_t&);
template value_t mappify_op::transduce(const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&);
template value_t mappify_op::transduce(const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&);
template value_t mappify_op::transduce(const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&);

template<typename Archive>
void mappify_op::serialize(Archive& ar) {
  ar(mapper_m);
}

std::vector<std::shared_ptr<transducer_variant>> mappify_op::nested_transducers() {
  return {mapper_m};
}


template void mappify_op::serialize<cereal::BinaryOutputArchive>( cereal::BinaryOutputArchive & );
template void mappify_op::serialize<cereal::BinaryInputArchive>( cereal::BinaryInputArchive & );

string list_reduce_op::default_name() const {
  return "list_reduce";
}

value_t list_reduce_op::transduce(const value_t& xs, const value_t& init_val)  {
  auto ret = init_val;
  for (auto&& x:xs.as_list()) {
    ret = reducer_m->transduce(ret, x);
  }
  return ret;
}

list_reduce_op::list_reduce_op(std::shared_ptr<transducer_variant> reducer): reducer_m(move(reducer)) {

}


template<typename Archive>
void list_reduce_op::serialize(Archive& ar) {
  ar(reducer_m);
}


std::vector<std::shared_ptr<transducer_variant>> list_reduce_op::nested_transducers() {
  return {reducer_m};
}

template void list_reduce_op::serialize<cereal::BinaryOutputArchive>( cereal::BinaryOutputArchive & );
template void list_reduce_op::serialize<cereal::BinaryInputArchive>( cereal::BinaryInputArchive & );

value_t list_select_op::transduce(const value_t& in0, const value_t& in1) {
  return in0.select(in1.as_integer());
}

string list_select_op::default_name() const {
  return "list_select";
}

value_t list_push_op::transduce(const value_t& in0, const value_t& in1) {
  auto ret = in0.as_list();
  auto&& item = in1;
  ret.push_back(item);
  return value_t(ret);
}

string list_push_op::default_name() const {
  return "push_back";
}


string list_zip_op::default_name() const {
  return "list_zip";
}

value_t list_zip_op::apply(const std::vector<value_t>& ins) {

  vector<value_t> ret;

  unsigned long length = ins[0].as_list().size();
  for (unsigned long i = 1; i < ins.size(); ++i) {
    if (ins[i].as_list().size() != length) {
      stringstream ss;
      ss << "Failed to apply " << default_name() << ": input lists must be of the same length.";
      throw_with_nested(std::runtime_error(ss.str()));
    }
  }

  for (unsigned long i = 0; i < length; ++i) {
    std::vector<value_t> entry;
    for (auto&& in:ins) {
      entry.push_back(in.as_list()[i]);
    }
    ret.emplace_back(entry);
  }

  return value_t(ret);
}


value_t list_reverse_op::transduce(const value_t& in0) {
  auto&& input_list = in0.as_list();
  vector<value_t> ret(input_list.rbegin(), input_list.rend());
  return value_t(input_list);
}

string list_reverse_op::default_name() const {
  return "list_reverse";
}

value_t list_select_many_op::transduce(const value_t& list, const value_t& indices) {

  vector<unsigned long> _ids;
  auto&& _indices = indices.as_list();
  _ids.reserve(_indices.size());
  for(auto&& i : _indices) {
    _ids.push_back(i.as_integer());
  }
  auto ret =  list.select_many(_ids);

  return ret;
}

string list_select_many_op::default_name() const {
  return "list_select_many";
}

value_t list_slice_op::transduce(const value_t& in0, const value_t& in1, const value_t& in2) {
  auto&& start = in1.as_integer();
  auto&& end = in2.as_integer();
  return in0.slice(start, end);
}

string list_slice_op::default_name() const {
  return "list_slice";
}


value_t list_size_op::transduce(const value_t& in0) {
  return value_t(in0.as_list().size());
}

string list_size_op::default_name() const {
  return "list_size";
}

value_t list_empty_op::transduce(const value_t& in0) {
  return value_t(in0.as_list().empty());
}

string list_empty_op::default_name() const {
  return "list_empty?";
}

list_filter_op::list_filter_op(std::shared_ptr<transducer_variant> filter):filter_m(move(filter)) {

}

value_t list_filter_op::transduce(const value_t& xs) {
  vector<value_t> ret;
  for(auto&& item:xs.as_list()) {
    if(filter_m->transduce(item).as_integer()) {
      ret.push_back(item);
    }
  }
  return value_t(move(ret));
}

std::string list_filter_op::default_name() const {
  return "list_filter";
}

std::vector<std::shared_ptr<transducer_variant>> list_filter_op::nested_transducers() {
  return {filter_m};
}

template<typename Archive>
void list_filter_op::serialize(Archive& ar) {
  ar(filter_m);
}

template void list_filter_op::serialize<cereal::BinaryOutputArchive>( cereal::BinaryOutputArchive & );
template void list_filter_op::serialize<cereal::BinaryInputArchive>( cereal::BinaryInputArchive & );
