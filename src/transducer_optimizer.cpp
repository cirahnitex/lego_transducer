//
// Created by Dekai WU and YAN Yuchen on 20200425.
//

#include "include/transducer_optimizer.hpp"
#include "include/lego_guard.hpp"
#include "backprop_trainable_parameter.hpp"
#include "dynet_computation_graph.hpp"
#include "transducer_variant.hpp"
#include "include/lego_list_operations.hpp"
#include "include/lego_tensor_operations.hpp"
using namespace std;
using namespace tg;


void simple_sgd_optimizer::set_learning_rate_impl(float lr) {
  weights_impl.learning_rate = lr;
  bias_impl.learning_rate = lr;
}

simple_sgd_optimizer::simple_sgd_optimizer(float learning_rate):tg::optimizer_base(), weights_impl(weights_pc_m, learning_rate), bias_impl(biases_pc_m, learning_rate) {

}


std::vector<dynet::Trainer *> simple_sgd_optimizer::get_impl() {
  return {&weights_impl, &bias_impl};
}



adam_optimizer::adam_optimizer(float learning_rate, float beta_1, float beta_2, float eps):tg::optimizer_base(),
                                                                                                                         weights_impl(weights_pc_m, learning_rate, beta_1, beta_2, eps),
                                                                                                                         bias_impl(biases_pc_m, learning_rate, beta_1, beta_2, eps){}

void adam_optimizer::set_learning_rate_impl(float lr) {
  weights_impl.learning_rate = lr;
  bias_impl.learning_rate = lr;
}

std::vector<dynet::Trainer *> adagrad_optimizer::get_impl() {
  return {&weights_impl, &bias_impl};
}

std::vector<dynet::Trainer *> adam_optimizer::get_impl() {
  return {&weights_impl, &bias_impl};
}


float optimizer_base::apply_learn_from_datum(transducer_model loss_fn, const std::vector<value_t>& inputs) {
  float ret = 0;
  {
    lego_training_guard _;
    ret = loss_fn.instantiate().apply_backward(inputs);
  }

  update_params();

  return ret;
}

float optimizer_base::dynamic_learn_from_datum(const dynamic_transducer_application& compute_loss) {
  float ret = 0;
  {
    lego_training_guard _;
    transducer_model model([&]() {
      return compute_loss();
    });

    ret = model.instantiate().backward();
  }
  update_params();
  return ret;
}

float optimizer_base::learn_from_batch(transducer_model loss_fn, const std::shared_ptr<const transducer_dataset>& datum_batch) {
  float ret = 0;
  {
    lego_training_guard _;
    ret = loss_fn.instantiate().batch_backward(datum_batch);
  }
  update_params();
  return ret;
}

float optimizer_base::dynamic_learn_from_batch(
  const std::vector<dynamic_transducer_application>& compute_loss_batch) {
  float ret = 0;
  {
    lego_training_guard _;
    ret = transducer_instance::dynamic_batch_backward(compute_loss_batch);
  }
  update_params();
  return ret;
}

optimizer_base::~optimizer_base() {
  // give back the control of the backprop trainable parameters
  for (backprop_trainable_parameter_base *param:weights_to_train_m) {
    param->use_internal_pc();
  }
  for (backprop_trainable_parameter_base *param:biases_to_train_m) {
    param->use_internal_pc();
  }
}

void optimizer_base::exclude_params(const std::function<bool(const lego_param_path&)>& filter) {
  for (backprop_trainable_parameter_base *param:weights_to_train_m) {
    if(filter(param->path)) {
      param->use_internal_pc();
    }
  }
  for (backprop_trainable_parameter_base *param:biases_to_train_m) {
    if(filter(param->path)) {
      param->use_internal_pc();
    }
  }
}

void optimizer_base::set_learning_rate(float lr) {
  std::lock_guard<std::mutex> lock(mtx);
  set_learning_rate_impl(lr);
}

void optimizer_base::update_params() {
  std::lock_guard<std::mutex> lock(mtx);
  try {
    for(auto&& trainer:get_impl()) {
      trainer->update();
    }
  }
  catch(...) {
    weights_pc_m.reset_gradient();
    biases_pc_m.reset_gradient();
    throw;
  }
}

void optimizer_base::set_weight_decay(float lambda) {
  weights_pc_m.set_weight_decay_lambda(lambda);
}

optimizer_base::optimizer_base() {
  for(auto&& param_p: backprop_trainable_parameter_base::all_parameters) {

    if(dynamic_cast<backprop_trainable_bias_parameter*>(param_p)) {
      biases_to_train_m.insert(param_p);
    }
    else {
      weights_to_train_m.insert(param_p);
    }
  }
  take_ownership_of_params();
}

void optimizer_base::take_ownership_of_params() {
  for (backprop_trainable_parameter_base *param:this->weights_to_train_m) {
    param->use_external_pc(weights_pc_m);
  }
  for (backprop_trainable_parameter_base *param:this->biases_to_train_m) {
    param->use_external_pc(biases_pc_m);
  }
}

void adagrad_optimizer::set_learning_rate_impl(float lr) {
  weights_impl.learning_rate = lr;
  bias_impl.learning_rate = lr;
}

adagrad_optimizer::adagrad_optimizer(float learning_rate, float eps)
:tg::optimizer_base(),
weights_impl(weights_pc_m, learning_rate, eps),
bias_impl(biases_pc_m, learning_rate, eps) {}
