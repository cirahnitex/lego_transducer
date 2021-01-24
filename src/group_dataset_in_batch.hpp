//
// Created by Dekai WU and YAN Yuchen on 20200604.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_GROUP_DATASET_IN_BATCH_HPP
#define LEGO_GROUP_DATASET_IN_BATCH_HPP
#include <vector>

namespace tg {
  template<typename DATUM>
  std::vector<std::vector<DATUM>> group_dataset_in_batch(const std::vector<DATUM>& dataset, unsigned batch_size) {
    std::vector<std::vector<DATUM>> ret;
    for(unsigned i=0; i<dataset.size(); i+=batch_size) {
      auto begin = dataset.begin() + i;
      auto end = (i+batch_size>=dataset.size())?dataset.end():(begin+batch_size);
      ret.emplace_back(begin, end);
    }
    return ret;
  }
}

#endif //LEGO_GROUP_DATASET_IN_BATCH_HPP
