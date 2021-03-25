//
// Created by Dekai WU and YAN Yuchen on 20201119.
//
#include "../include/lego_transducer.hpp"
#include <iostream>
using namespace std;
using namespace tg;
int main() {
  lego_initialize();


  // declaring two dense layers
  transducer_model relu_dense0 = compose(relu, dense_structure.initialize(2, 4));
  transducer_model dense1 = dense_structure.initialize(4, 1);

transducer_model compute_logit([&](const value_placeholder& x0, const value_placeholder& x1){
  value_placeholder t = tensor_concat({x0, x1});
  return dense1(relu_dense0(t));
});


transducer_model performance_component([&](const value_placeholder& x0, const value_placeholder& x1) {
  return compute_logit(x0, x1) > value_placeholder::constant(0);
});

transducer_model loss_fn([&](const value_placeholder& x0, const value_placeholder& x1, const value_placeholder& oracle) {
  return pickneglogsigmoid(compute_logit(x0, x1), oracle);
});


  // prepare a training dataset
auto training_set = create_transducer_dataset(3);
training_set->emplace_back(0,0,0);
training_set->emplace_back(0,1,1);
training_set->emplace_back(1,0,1);
training_set->emplace_back(1,1,0);



// use Stochastic Gradient Descent backprop training algorithm
simple_sgd_optimizer optimizer(0.01);

training_pipeline trainer(&optimizer);
trainer.set_num_epochs(1000);

trainer.train(loss_fn, training_set);

// save the trained model into a file
transducer_model::save_to_file("model.bin", performance_component, loss_fn);

// load the trained models from file
transducer_model loaded_performance_component, loaded_loss_fn;
transducer_model::load_from_file("model.bin", loaded_performance_component, loaded_loss_fn);


// output the model prediction
cout << loaded_performance_component(0,0) << endl;
cout << loaded_performance_component(0,1) << endl;
cout << loaded_performance_component(1,0) << endl;
cout << loaded_performance_component(1,1) << endl;
}
