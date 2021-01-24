//
// Created by Dekai WU and YAN Yuchen on 20210119.
//

#include "include/lego_param_naming_guard.hpp"

using namespace std;
using namespace tg;

thread_local std::vector<std::string> lego_param_naming_guard::v;

namespace _private_giCOoekLkX {
  bool is_match_impl(const std::vector<std::string>& values, const std::vector<std::string>& fragment, unsigned long offset) {
    for(unsigned long i=0; i<fragment.size(); ++i) {
      if(values.at(offset + i) != fragment.at(i)) return false;
    }
    return true;
  }
};
using namespace _private_giCOoekLkX;

bool lego_param_path::contains(const std::vector<std::string>& fragment) const {
  if(fragment.empty()) return true;
  if(fragment.size() > values.size()) return false;
  unsigned long end = values.size() - fragment.size();
  for(unsigned long begin=0; begin<=end; ++begin) {
    if(is_match_impl(values, fragment, begin)) return true;
  }
  return false;
}

bool lego_param_path::starts_with(const vector<std::string>& fragment) const {
  if(fragment.empty()) return true;
  if(fragment.size() > values.size()) return false;

  return is_match_impl(values, fragment, 0);
}

bool lego_param_path::ends_with(const vector<std::string>& fragment) const {
  if(fragment.empty()) return true;
  if(fragment.size() > values.size()) return false;
  unsigned long begin = values.size() - fragment.size();
  return is_match_impl(values, fragment, begin);
}

lego_param_path::lego_param_path(vector<std::string> values):values(std::move(values)) {

}
