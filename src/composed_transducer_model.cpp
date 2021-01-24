//
// Created by Dekai WU and YAN Yuchen on 20201203.
//

#include "composed_transducer_model.hpp"
#include "transducer_variant.hpp"

using namespace tg;
using namespace std;

value_t composed_transducer_model::transduce(const value_t& x) {
  if(pipeline.empty()) return x;
  auto itr = pipeline.begin();
  value_t y = (*itr)->transduce(x);
  for(++itr; itr != pipeline.end(); ++itr) {
    y = (*itr)->transduce(y);
  }
  return y;
}

std::string composed_transducer_model::default_name() const {
  return "composed";
}

std::vector<std::shared_ptr<transducer_variant>> composed_transducer_model::nested_transducers() {
  return std::vector<std::shared_ptr<transducer_variant>>(pipeline.begin(), pipeline.end());
}

composed_transducer_model
composed_transducer_model::compose_from(std::initializer_list<std::shared_ptr<transducer_variant>> pieces) {
  std::deque<std::shared_ptr<transducer_variant>> pipeline;
  for(auto itr = std::rbegin(pieces); itr != std::rend(pieces); ++itr) {
    (*itr)->visit([&](auto&& transducer) {
      using transducer_t = decay_t<decltype(transducer)>;
      if constexpr (std::is_same_v<transducer_t, composed_transducer_model>) {
        pipeline.insert(pipeline.end(), transducer.pipeline.begin(), transducer.pipeline.end());
      }
      else {
        pipeline.push_back(*itr);
      }
    });
  }
  return composed_transducer_model(move(pipeline));
}
