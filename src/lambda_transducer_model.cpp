//
// Created by Dekai WU and YAN Yuchen on 20200423.
//

#include "lambda_transducer_model.hpp"
#include "transducer_variant.hpp"
using namespace std;
using namespace tg;

bool tg::lambda_transducer_model::is_arity(unsigned long arity) const {
  return arity_m == arity;
}

string tg::lambda_transducer_model::default_name() const {
  return "<lambda>";
}

std::vector<std::shared_ptr<transducer_variant>> tg::lambda_transducer_model::nested_transducers() {
  std::vector<std::shared_ptr<transducer_variant>> ret;
  for(auto&& local:locals_m) {
    std::visit([&](auto&& v){
      using V = std::decay_t<decltype(v)>;
      if constexpr (std::is_same_v<V, compute_value_input_impl> || std::is_same_v<V, compute_value_const_impl> || std::is_same_v<V, compute_value_make_list_impl>) {
      }
      else {
        std::visit([&](auto&& transducer) {
          using transducer_T = std::decay_t<decltype(transducer)>;
          if constexpr (std::is_same_v<transducer_T, std::shared_ptr<transducer_variant>>) {
            ret.push_back(transducer);
          }
        },v.transducer);
      }
    }, local.impl);
  }
  return ret;
}


thread_local std::deque<lambda_transducer_model*> lambda_transducer_model_construction_guard::unfinished_transducers{}; /* NOLINT*/

lambda_transducer_model_construction_guard::lambda_transducer_model_construction_guard(
  lambda_transducer_model* transducer) {
  unfinished_transducers.push_back(transducer);
}

lambda_transducer_model_construction_guard::~lambda_transducer_model_construction_guard() {
  unfinished_transducers.pop_back();
}

lambda_transducer_model* lambda_transducer_model_construction_guard::top() {
  if(unfinished_transducers.empty()) return nullptr;
  return unfinished_transducers.back();
}

unsigned long lambda_transducer_model_construction_guard::size() {
  return unfinished_transducers.size();
}

template <typename T>
inline constexpr bool always_false_v = false;

value_placeholder lambda_transducer_model::make_value_placeholder_impl(compute_value_behavior behavior) {
  auto idx = locals_m.size();
  locals_m.push_back(std::move(behavior));
  return value_placeholder(nesting_depth_m, idx);
}

value_placeholder lambda_transducer_model::make_value_placeholder_from_input() {
  return make_value_placeholder_impl(compute_value_behavior(compute_value_input_impl{}));
}

value_placeholder lambda_transducer_model::make_value_placeholder_from_constant(value_t val) {
  return make_value_placeholder_impl(compute_value_behavior(std::move(val)));
}


unsigned long lambda_transducer_model::nesting_depth() const {
  return nesting_depth_m;
}

unsigned long lambda_transducer_model::num_locals() const {
  return locals_m.size();
}

value_placeholder lambda_transducer_model::make_value_placeholder_from_make_list(
  std::vector<value_placeholder> inputs) {
  return make_value_placeholder_impl(compute_value_behavior(compute_value_make_list_impl(std::move(inputs))));
}

template<typename ...Args>
value_placeholder lambda_transducer_model::make_value_placeholder_from_transducing(
  std::shared_ptr<transducer_variant> transducer, Args ...inputs) {
  constexpr unsigned long argc = sizeof...(inputs);
  return transducer->visit([&](auto&& v)->value_placeholder {
    using V = std::decay_t<decltype(v)>;
    constexpr auto traits = transducer_variant::transducer_traits<V>();
    if constexpr (std::is_same_v<V, tbd_transducer>) {

      return make_value_placeholder_impl(compute_value_behavior(compute_value_transduce_impl<argc>{
        .transducer = std::weak_ptr<transducer_variant>(transducer),
        .inputs = make_tuple(move(inputs)...)
      }));
    }
    else {

      if constexpr (traits.is_lazy) {
        has_lazy_operation = true;
      }

      return make_value_placeholder_impl(compute_value_behavior(compute_value_transduce_impl<argc>{
        .transducer = move(transducer),
        .inputs = make_tuple(move(inputs)...)
      }));
    }
  });
}

template value_placeholder lambda_transducer_model::make_value_placeholder_from_transducing(std::shared_ptr<transducer_variant> transducer);
template value_placeholder lambda_transducer_model::make_value_placeholder_from_transducing(std::shared_ptr<transducer_variant> transducer, value_placeholder);
template value_placeholder lambda_transducer_model::make_value_placeholder_from_transducing(std::shared_ptr<transducer_variant> transducer, value_placeholder, value_placeholder);
template value_placeholder lambda_transducer_model::make_value_placeholder_from_transducing(std::shared_ptr<transducer_variant> transducer, value_placeholder, value_placeholder, value_placeholder);
template value_placeholder lambda_transducer_model::make_value_placeholder_from_transducing(std::shared_ptr<transducer_variant> transducer, value_placeholder, value_placeholder, value_placeholder, value_placeholder);
template value_placeholder lambda_transducer_model::make_value_placeholder_from_transducing(std::shared_ptr<transducer_variant> transducer, value_placeholder, value_placeholder, value_placeholder, value_placeholder, value_placeholder);
template value_placeholder lambda_transducer_model::make_value_placeholder_from_transducing(std::shared_ptr<transducer_variant> transducer, value_placeholder, value_placeholder, value_placeholder, value_placeholder, value_placeholder, value_placeholder);
template value_placeholder lambda_transducer_model::make_value_placeholder_from_transducing(std::shared_ptr<transducer_variant> transducer, value_placeholder, value_placeholder, value_placeholder, value_placeholder, value_placeholder, value_placeholder, value_placeholder);
template value_placeholder lambda_transducer_model::make_value_placeholder_from_transducing(std::shared_ptr<transducer_variant> transducer, value_placeholder, value_placeholder, value_placeholder, value_placeholder, value_placeholder, value_placeholder, value_placeholder, value_placeholder);

value_t lambda_transducer_model::evaluate_compute_behavior(const compute_value_behavior& behavior) {
  return std::visit([&](auto&& v)->value_t {
    using V = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<V, compute_value_input_impl>) {
      throw std::runtime_error("Unfulfilled input placeholder");
    }
    else if constexpr (std::is_same_v<V, compute_value_const_impl>) {
      return v;
    }
    else if constexpr (std::is_same_v<V, compute_value_make_list_impl>) {
      const std::vector<value_placeholder>& vps = v.inputs;
      std::vector<value_t> ret;
      ret.reserve(vps.size());
      for(auto&& vp:vps) {
        ret.push_back(evaluate_value_placeholder(vp));
      }
      return value_t(std::move(ret));
    }
    else {
      return std::visit([&](auto&& transducer)->value_t {
        using transducer_t = std::decay_t<decltype(transducer)>;
        if constexpr (std::is_same_v<transducer_t, std::shared_ptr<transducer_variant>>) {
          auto helper = [&](auto&& ...args) {return transducer->transduce_placeholder(args...);};
          return std::apply(helper, v.inputs);
        }
        else {
          auto helper = [&](auto&& ...args) {return transducer.lock()->transduce_placeholder(args...);};
          return std::apply(helper, v.inputs);
        }
      }, v.transducer);
    }
  }, behavior.impl);
}

const value_t & lambda_transducer_model::evaluate_value_placeholder(const value_placeholder& vp) {

  auto scope = lambda_transducer_value_cache::get_scope_by_nesting_depth(vp.owner_nesting_depth);

  if(!scope) {
    throw std::runtime_error("Value cannot be obtained in the current scope");
  }

  auto& [computed, value] = scope->values_cache[vp.value_idx];

  // Return the value if it is in cache
  if(computed) return value;

  auto owner = scope->transducer;

  // Use the sequential evaluation strategy if the transducer has no lazy operations
  if(!owner->has_lazy_operation) {
    for(unsigned long idx_i = owner->arity_m; idx_i < owner->num_locals(); ++idx_i) {
      auto& [computed, value] = scope->values_cache[idx_i];
      value = evaluate_compute_behavior(owner->locals_m[idx_i]);
      computed = true;
      if(idx_i == vp.value_idx) return value;
    }
  }

  // Compute the value according to computation behavior
  value = evaluate_compute_behavior(owner->locals_m[vp.value_idx]);
  computed = true;
  return value;
}
