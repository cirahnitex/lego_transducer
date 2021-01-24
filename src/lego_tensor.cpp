//
// Created by Dekai WU and YAN Yuchen on 20200517.
//

#include "include/lego_tensor.hpp"

using namespace std;
using namespace tg;

dynet::Dim tg::to_dynet_dim(const tg::tensor_shape_t& dim) {
  return dynet::Dim(vector<long>{dim.begin(), dim.end()});
}

tensor_shape_t tg::from_dynet_dim(const dynet::Dim& dynet_dim) {
  tensor_shape_t ret;
  ret.reserve(dynet_dim.nd);
  for (unsigned long i = 0; i < dynet_dim.nd; ++i) {
    ret.push_back(dynet_dim[i]);
  }
  return ret;
}

unsigned long tg::tensor_num_values(const tg::tensor_shape_t& dim) {
  unsigned ret = 1;
  for (auto&& axis:dim) {
    ret *= axis;
  }
  return ret;
}

std::string tg::print_tensor_shape(const tensor_shape_t& sizes) {
  if (sizes.empty()) throw_with_nested(std::runtime_error("Tensor of rank 0 is not allowed."));
  std::stringstream ss;
  ss << sizes[0];
  for (unsigned long i = 1; i < sizes.size(); ++i) {
    ss << "x" << sizes[i];
  }
  return ss.str();
}

tg::tensor_t tg::tensor_t::from_dynet_tensor(const dynet::Tensor& dynet_tensor) {
  tensor_t ret;
  ret.values = dynet::as_vector(dynet_tensor);
  ret.shape = from_dynet_dim(dynet_tensor.d);
  return ret;
}

tensor_t tensor_t::zeros(const tensor_shape_t& dim) {
  tensor_t ret;
  ret.shape = dim;
  ret.values.resize(tensor_num_values(dim), 0);
  return ret;
}

tensor_t tensor_t::ones(const tensor_shape_t& dim) {
  tensor_t ret;
  ret.shape = dim;
  ret.values.resize(tensor_num_values(dim), 1);
  return ret;
}

std::ostream& tg::operator<<(std::ostream& os, const tensor_t& x) {

  os << "tensor(" << print_tensor_shape(x.shape) << ")";

  if (x.values.size() > tensor_t::MAX_TENSOR_ELEMS_TO_PRINT) return os;

  if (x.shape.empty()) {
    return os << endl << std::to_string(x.values[0]) << endl;
  }

  if (x.shape.size() == 1) {
    os << endl;
    for (unsigned long i = 0; i < x.shape[0]; ++i) {
      os << std::to_string(x.values[i]) << endl;
    }
    return os;
  }

  if (x.shape.size() == 2) {
    os << endl;
    auto n_columns = x.shape[0];
    auto n_rows = x.shape[1];
    for (unsigned long i_column = 0; i_column < n_columns; ++i_column) {
      for (unsigned long i_row = 0; i_row < n_rows; ++i_row) {
        if (i_row > 0) os << " ";
        os << std::to_string(x.values[i_row * n_columns + i_column]);
      }
      os << endl;
    }
    return os;
  }

  return os;
}

tensor_t::tensor_t(std::vector<float> values, tensor_shape_t shape):values(move(values)), shape(move(shape)) {
  auto shape_num_values = tensor_num_values(this->shape);
  if (this->values.size() != shape_num_values) {
    stringstream ss;
    ss << "Cannot create tensor. There are (" << this->values.size() << ") pre-supplied values but tensor size is {";
    ss << print_tensor_shape(this->shape);
    ss << "}";
    throw_with_nested(std::runtime_error(ss.str()));
  }
}

tensor_t::tensor_t(std::vector<float> values):tensor_t(move(values), {values.size()}) {}

