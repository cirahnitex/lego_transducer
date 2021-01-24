//
// Created by Dekai WU and YAN Yuchen on 20200423.
//

#include "include/transducer_instance.hpp"
#include "include/transducer_dataset.hpp"
#include "include/lego_guard.hpp"
#include "const_value_model.hpp"
#include "include/transducer_typed_value.hpp"
#include "include/transducer_model.hpp"
#include <sstream>
#include "dynet_computation_graph.hpp"
#include "include/wallclock_timer.hpp"
#include "transducer_variant.hpp"
#include "include/lego_list_operations.hpp"

using namespace std;
using namespace tg;

tg::transducer_instance::transducer_instance(std::shared_ptr<transducer_variant> impl) : impl(move(impl)) {

}

tg::value_t tg::transducer_instance::operator()() {
  return apply({});
}

tg::value_t tg::transducer_instance::operator()(const value_t& in0) {
  return apply({in0});
}

tg::value_t tg::transducer_instance::operator()(const value_t& in0, const value_t& in1) {
  return apply({in0, in1});
}

namespace nquSSHTXgN {
  // prints the explanatory string of an exception. If the exception is nested,
  // recurses to print the explanatory of the exception it holds
  void print_exception(const std::exception& e, int level = 0) {
    std::cerr << std::string(level, ' ') << e.what() << endl;
    try {
      std::rethrow_if_nested(e);
    } catch (const std::exception& nested_e) {
      print_exception(nested_e, level + 1);
    } catch (...) {}
  }
}


tg::value_t tg::transducer_instance::apply(const std::vector<value_t>& ins) {
  using namespace nquSSHTXgN;
  if (!impl->is_arity(ins.size())) {
    stringstream ss;
    ss << "Transducer " << impl->name() << " cannot take " << ins.size()
       << " arguments.";
    throw_with_nested(std::runtime_error(ss.str()));
  }
  try {
    dynet_computation_graph::discard();
    tg::value_t ret;
    if (show_cg_construction_time_guard::is_guarded()) {
      wallclock_timer timer;
      timer.start();

      ret = impl->_apply(ins);

      cerr << "apply time elapsed: " << timer.milliseconds_elapsed() << endl;

      timer.start();

      ret.evaluate();
      dynet_computation_graph::discard();

      cerr << "commit time elapsed: " << timer.milliseconds_elapsed() << endl;
    } else {
      ret = impl->_apply(ins);
      ret.evaluate();
      dynet_computation_graph::discard();
    }

    return ret;
  }
  catch (const nan_or_inf_exception& e) {
    if (immediate_computation_guard::is_guarded()) {
      print_exception(e);
      throw std::runtime_error("");
    } else {
      immediate_computation_guard _;
      cerr << "NaN or Inf encountered. Re-evaluating using immediate mode" << endl;
      return apply(ins);
    }
  }
  catch (const std::exception& e) {

    print_exception(e);
    throw std::runtime_error("");
  }
}

std::vector<value_t> transducer_instance::dynamic_batch_apply(
  const std::vector<tg::dynamic_transducer_application>& applications) {
  using namespace nquSSHTXgN;
  try {
    dynet_computation_graph::discard();

    std::vector<transducer_model> transducers;
    transducers.reserve(applications.size());
    for (auto&& application:applications) {
      transducers.emplace_back([&]() -> value_placeholder {
        return application();
      });
    }

    std::vector<value_t> ret;
    ret.reserve(transducers.size());
    for (auto&& transducer:transducers) {
      ret.push_back(transducer._get_impl()->transduce());
    }
    value_t(ret).evaluate();
    dynet_computation_graph::discard();
    return ret;
  }
  catch (const nan_or_inf_exception& e) {
    if (immediate_computation_guard::is_guarded()) {
      print_exception(e);
      throw std::runtime_error("");
    } else {
      immediate_computation_guard _;
      cerr << "NaN or Inf encountered. Re-evaluating using immediate mode" << endl;
      return dynamic_batch_apply(applications);
    }
  }
  catch (const std::exception& e) {
    print_exception(e);
    throw std::runtime_error("");
  }
}


std::vector<value_t> tg::transducer_instance::batch_apply(const std::shared_ptr<const transducer_dataset>& dataset) {
  using namespace nquSSHTXgN;
  if (!impl->is_arity(dataset->arity())) {
    stringstream ss;
    ss << "Transducer " << impl->name() << " cannot take " << dataset->arity() << " arguments.";
    throw_with_nested(std::runtime_error(ss.str()));
  }
  try {
    dynet_computation_graph::discard();
    vector<value_t> ret;
    ret.reserve(dataset->size());
    for (auto&& datum:(*dataset)) {
      ret.push_back(impl->_apply(datum));
    }
    value_t(ret).evaluate();
    dynet_computation_graph::discard();
    return ret;
  }
  catch (const nan_or_inf_exception& e) {
    if (immediate_computation_guard::is_guarded()) {
      print_exception(e);
      throw std::runtime_error("");
    } else {
      immediate_computation_guard _;
      cerr << "NaN or Inf encountered. Re-evaluating using immediate mode" << endl;
      return batch_apply(dataset);
    }
  }
  catch (const std::exception& e) {
    print_exception(e);
    throw std::runtime_error("");
  }
}

tg::transducer_instance::transducer_instance(value_t v) : impl(
  make_shared<transducer_variant>(const_value_model(move(v)))) {

}

tg::transducer_model tg::transducer_instance::prototype() {
  return transducer_model(impl);
}

scalar_t transducer_instance::apply_backward(const std::vector<value_t>& ins) {
  using namespace nquSSHTXgN;
  if (!impl->is_arity(ins.size())) {
    stringstream ss;
    ss << "Transducer " << impl->name() << " cannot take " << ins.size()
       << " arguments.";
    throw_with_nested(std::runtime_error(ss.str()));
  }
  try {
    dynet_computation_graph::discard();
    auto ret = impl->_apply(ins);
    if (ret.is_symbolic_tensor() && ret.as_symbolic_tensor().dim().sum_dims() == 1) {
      auto&& expr = ret.as_symbolic_tensor();
      auto loss_val = dynet::as_scalar(dynet_computation_graph::p()->incremental_forward(expr));
      block_nan_or_inf(loss_val);
      dynet_computation_graph::p()->backward(expr);
      dynet_computation_graph::discard();
      return loss_val;
    } else {
      stringstream ss;
      ss << "Cannot perform backward computation because the transducer did not return a tensor of shape {1}. Got: "<<ret;
      throw_with_nested(std::runtime_error(ss.str()));
    }

  }
  catch (const nan_or_inf_exception& e) {
    if (immediate_computation_guard::is_guarded()) {
      print_exception(e);
      throw std::runtime_error("");
    } else {
      immediate_computation_guard _;
      cerr << "NaN or Inf encountered. Re-evaluating using immediate mode" << endl;
      return apply_backward(ins);
    }
  }
  catch (const std::exception& e) {
    print_exception(e);
    throw std::runtime_error("");
  }
}


scalar_t
transducer_instance::dynamic_batch_backward(const std::vector<tg::dynamic_transducer_application>& loss_applications) {
  using namespace nquSSHTXgN;
  try {
    dynet_computation_graph::discard();
    vector<dynet::Expression> losses;

    // keep these transducers in memory during dynet evaluation, because constant values are stored in these transducers
    // If these constant values has gone out of scope by the time dynet evaluation happens, undefined behaviors will occur.
    std::vector<lambda_transducer_model> loss_application_transducers;
    loss_application_transducers.reserve(loss_applications.size());
    for(auto&& loss_application:loss_applications) {
      loss_application_transducers.push_back(lambda_transducer_model::from_lambda_fn<0>(loss_application));
    }

    for (auto&& loss_application_transducer:loss_application_transducers) {

      auto ret = loss_application_transducer.transduce();

      if (ret.is_symbolic_tensor() && ret.as_symbolic_tensor().dim().sum_dims() == 1) {

        losses.push_back(ret.as_symbolic_tensor());
      } else {
        stringstream ss;
        ss << "Cannot perform backward computation because the transducer did not return a tensor of shape {1}. Got: "<<ret;
        throw_with_nested(std::runtime_error(ss.str()));
      }
    }

    auto loss = dynet::sum(losses);

    auto loss_val = dynet::as_scalar(dynet_computation_graph::p()->incremental_forward(loss));
    block_nan_or_inf(loss_val);
    dynet_computation_graph::p()->backward(loss);
    dynet_computation_graph::discard();
    return loss_val;
  }
  catch (const nan_or_inf_exception& e) {
    if (immediate_computation_guard::is_guarded()) {
      print_exception(e);
      throw std::runtime_error("");
    } else {
      immediate_computation_guard _;
      cerr << "NaN or Inf encountered. Re-evaluating using immediate mode" << endl;
      return transducer_instance::dynamic_batch_backward(loss_applications);
    }
  }
  catch (const std::exception& e) {

    print_exception(e);
    throw std::runtime_error("");
  }
}

scalar_t transducer_instance::batch_backward(const std::shared_ptr<const transducer_dataset>& dataset) {
  using namespace nquSSHTXgN;
  if (!impl->is_arity(dataset->arity())) {
    stringstream ss;
    ss << "Transducer " << impl->name() << " cannot apply to " << dataset->arity()
       << " arguments.";
    throw_with_nested(std::runtime_error(ss.str()));
  }
  try {
    dynet_computation_graph::discard();
    vector<dynet::Expression> losses;

    for (auto&& datum:(*dataset)) {

      auto ret = impl->_apply(datum);

      if (ret.is_symbolic_tensor() && ret.as_symbolic_tensor().dim().sum_dims() == 1) {

        losses.push_back(ret.as_symbolic_tensor());
      } else {
        stringstream ss;
        ss << "Cannot perform backward computation because the transducer did not return a tensor of shape {1}. Got: "<<ret;
        throw_with_nested(std::runtime_error(ss.str()));
      }
    }

    auto loss = dynet::sum(losses);

    auto loss_val = dynet::as_scalar(dynet_computation_graph::p()->incremental_forward(loss));
    block_nan_or_inf(loss_val);
    dynet_computation_graph::p()->backward(loss);

    dynet_computation_graph::discard();
    return loss_val;
  }
  catch (const nan_or_inf_exception& e) {
    if (immediate_computation_guard::is_guarded()) {
      print_exception(e);
      throw std::runtime_error("");
    } else {
      immediate_computation_guard _;
      cerr << "NaN or Inf encountered. Re-evaluating using immediate mode" << endl;
      return batch_backward(dataset);
    }
  }
  catch (const std::exception& e) {

    print_exception(e);
    throw std::runtime_error("");
  }
}

