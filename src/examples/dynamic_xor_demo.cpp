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
  transducer_model dense0 = compose(relu, dense_structure.initialize(2, 4));
  transducer_model dense1 = dense_structure.initialize(4, 1);

  auto compute_logit = [&](bool x, bool y) {
    auto t = tensor_concat({value_placeholder::constant(x), value_placeholder::constant(y)});
    return dense1(dense0(t));
  };

  auto predict = [&](bool x, bool y)->bool {
    transducer_model tmp([&]() {
      return compute_logit(x, y);
    });
    return tmp().as_float() > 0;
  };

  auto compute_loss = [&](bool x, bool y, bool oracle)->value_placeholder {
    return pickneglogsigmoid(compute_logit(x, y),
                             value_placeholder::constant(oracle));
  };

  // prepare a training dataset
  vector<tuple<bool, bool, bool>> training_set{
    {false, false, false},
    {false, true, true},
    {true, false, true},
    {true, true, false}
  };

  vector<dynamic_transducer_application> training_set_applications;
  for(auto&& [x, y, oracle]:training_set) {
    training_set_applications.emplace_back([&, x(x), y(y), oracle(oracle)](){
      return compute_loss(x, y, oracle);
    });
  }

  // use Stochastic Gradient Descent backprop training algorithm
  simple_sgd_optimizer optimizer(0.01);
  training_pipeline trainer(&optimizer);
  trainer.set_num_epochs(1000);
  trainer.dynamic_train(training_set_applications);


  // output the model prediction
  cout << predict(false, false) << endl;
  cout << predict(true, false) << endl;
  cout << predict(false, true) << endl;
  cout << predict(true, true) << endl;
}
