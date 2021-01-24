//
// Created by Dekai WU and YAN Yuchen on 20200709.
//

#include "include/transducer_dataset.hpp"
#include <fstream>

using namespace tg;
using namespace std;

tg::transducer_dataset_vec_impl::transducer_dataset_vec_impl(unsigned long arity): arity_m(arity) {

}

void tg::transducer_dataset_vec_impl::apply_emplace_back(std::vector<value_t> datum) {
  if(arity_m != datum.size()) {
    stringstream ss;
    ss << "Cannot add datum to dataset. Dataset has arity "<< arity_m<< " but datum has arity "<<datum.size();
    throw_with_nested(std::runtime_error(ss.str()));
  }
  datums.push_back(move(datum));
}

const std::vector<value_t>& transducer_dataset_vec_impl::at(unsigned long i) const& {
  if(i >= size()) {
    stringstream ss;
    ss << "Cannot access datum #" << i << " from dataset of size "<<size() << endl;
    throw_with_nested(std::runtime_error(ss.str()));
  }
  return datums.at(i);
}


unsigned long transducer_dataset_vec_impl::size() const {
  return datums.size();
}

unsigned long transducer_dataset_vec_impl::arity() const {
  return arity_m;
}

void transducer_dataset_vec_impl::save_to_stream(ostream& os) const {
  cereal::BinaryOutputArchive ar(os);
  ar << *this;
}

void transducer_dataset_vec_impl::save_to_file(const string& path) const {
  ofstream ofs(path);
  if(!ofs.is_open()) {
    throw std::runtime_error("Cannot write to file :" + path);
  }
  save_to_stream(ofs);
}


std::shared_ptr<transducer_dataset_vec_impl> tg::create_transducer_dataset(unsigned long arity) {
  return std::make_shared<transducer_dataset_vec_impl>(arity);
}

unsigned long transducer_dataset_ref_impl::arity() const {
  return parent_m->arity();
}

unsigned long transducer_dataset_ref_impl::size() const {
  return indices_m.size();
}

const vector<value_t>& transducer_dataset_ref_impl::at(unsigned long i) const& {
  if(i >= size()) {
    stringstream ss;
    ss << "Cannot access datum #" << i << " from dataset of size "<<size() << endl;
    throw_with_nested(std::runtime_error(ss.str()));
  }
  return parent_m->at(indices_m[i]);
}


transducer_dataset_ref_impl::transducer_dataset_ref_impl(shared_ptr<const transducer_dataset> parent,
                                                         vector<unsigned long> indices) : parent_m(move(parent)),
                                                                                                 indices_m(move(indices)) {}


std::shared_ptr<transducer_dataset> transducer_dataset::slice(unsigned long start, unsigned long end) const {
  if(end > size()) end = size();
  if(start > end) start = end;
  vector<unsigned long> indices(end - start);
  std::iota(indices.begin(), indices.end(), start);
  return make_shared<transducer_dataset_ref_impl>(shared_from_this(), move(indices));
}

transducer_dataset::iterator transducer_dataset::begin() const& {
  return transducer_dataset::iterator(shared_from_this(),0);
}

transducer_dataset::iterator transducer_dataset::end() const& {
  return transducer_dataset::iterator(shared_from_this(), size());
}

std::vector<std::shared_ptr<transducer_dataset>> transducer_dataset::group_to_batch(unsigned long batch_size) const {
  auto len = size();
  std::vector<std::shared_ptr<transducer_dataset>> ret;
  for(unsigned long i=0; i<len; i+=batch_size) {
    ret.push_back(slice(i, i+batch_size));
  }
  return ret;
}

bool transducer_dataset::empty() const {
  return size() == 0;
}

void transducer_dataset::save_to_stream(ostream& os) const {
  throw std::runtime_error("saving is not implemented on this type of dataset");
}

void transducer_dataset::save_to_file(const std::string& path) const {
  throw std::runtime_error("saving is not implemented on this type of dataset");
}

std::shared_ptr<transducer_dataset> transducer_dataset::load_from_stream(istream& is) {
  cereal::BinaryInputArchive ar(is);
  auto ret = std::make_shared<transducer_dataset_vec_impl>();
  ar >> *ret;
  return ret;
}

std::shared_ptr<transducer_dataset> transducer_dataset::load_from_file(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if(!ifs.is_open()) {
    throw std::runtime_error("Cannot read from file: " + path);
  }
  return load_from_stream(ifs);
}
