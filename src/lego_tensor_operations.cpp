//
// Created by Dekai WU and YAN Yuchen on 20200506.
//

#include "include/lego_tensor_operations.hpp"
#include "tensor_operations_impl.hpp"
#include "transducer_variant.hpp"
#include "include/transducer_model.hpp"

using namespace std;
using namespace tg;

tg::value_placeholder tg::tensor_concat(const std::vector<value_placeholder>& tensors, unsigned long axis) {
  auto model = tg::transducer_model(make_shared<transducer_variant>(tg::n_ary_concat_op_static_axis(axis)));
  return model.apply(tensors);
}

tg::value_placeholder tg::tensor_concat(const tg::value_placeholder& tensors, unsigned long axis) {
  auto model = tg::transducer_model(make_shared<transducer_variant>(tg::tensor_concat_op_static_axis(axis)));
  return model(tensors);
}



namespace tg {
  tg::transducer_model max_index_of_tensor1d(make_shared<transducer_variant>(tg::max_index_of_tensor1d_op()));

  tg::value_placeholder operator-(const tg::value_placeholder& x) {
    static auto model = tg::transducer_model(make_shared<transducer_variant>(tg::neg_op()));
    return model(x);
  }

  tg::value_placeholder operator-(const tg::value_placeholder& x, const tg::value_placeholder& y) {
    static auto model = tg::transducer_model(make_shared<transducer_variant>(minus_op()));
    return model(x, y);
  }

  tg::value_placeholder operator-(const tg::value_placeholder& x, long y) {
    auto model = tg::transducer_model(make_shared<transducer_variant>(minus_scalar_op<long>(y)));
    return model(x);
  }

  tg::value_placeholder operator-(const tg::value_placeholder& x, float y) {
    auto model = tg::transducer_model(make_shared<transducer_variant>(minus_scalar_op<float>(y)));
    return model(x);
  }

  tg::value_placeholder operator-(long x, const tg::value_placeholder& y) {
    auto model = tg::transducer_model(make_shared<transducer_variant>(scalar_minus_op<long>(x)));
    return model(y);
  }

  tg::value_placeholder operator-(float x, const tg::value_placeholder& y) {
    auto model = tg::transducer_model(make_shared<transducer_variant>(scalar_minus_op<float>(x)));
    return model(y);
  }

  tg::value_placeholder operator+(const tg::value_placeholder& x, const tg::value_placeholder& y) {
    static auto model = tg::transducer_model(make_shared<transducer_variant>(add_op()));
    return model(x, y);
  }

  tg::value_placeholder operator+(const tg::value_placeholder& x, long y) {
    auto model = tg::transducer_model(make_shared<transducer_variant>(add_scalar_op<long>(y)));
    return model(x);
  }

  tg::value_placeholder operator+(const tg::value_placeholder& x, float y){
    auto model = tg::transducer_model(make_shared<transducer_variant>(add_scalar_op<float>(y)));
    return model(x);
  }


  tg::value_placeholder operator*(const tg::value_placeholder& x, const tg::value_placeholder& y) {
    static auto model = tg::transducer_model(make_shared<transducer_variant>(cmult_op()));
    return model(x, y);
  }

  tg::value_placeholder operator*(const tg::value_placeholder& x, long y) {
    auto model = tg::transducer_model(make_shared<transducer_variant>(cmult_scalar_op<long>(y)));
    return model(x);
  }

  tg::value_placeholder operator*(const tg::value_placeholder& x, float y) {
    auto model = tg::transducer_model(make_shared<transducer_variant>(cmult_scalar_op<float>(y)));
    return model(x);
  }


  tg::value_placeholder operator/(const tg::value_placeholder& x, const tg::value_placeholder& y) {
    static auto model = tg::transducer_model(make_shared<transducer_variant>(divide_op()));
    return model(x, y);
  }

  tg::value_placeholder operator/(const tg::value_placeholder& x, long y) {
    auto model = tg::transducer_model(make_shared<transducer_variant>(divide_scalar_op<long>(y)));
    return model(x);
  }

  tg::value_placeholder operator/(const tg::value_placeholder& x, float y) {
    auto model = tg::transducer_model(make_shared<transducer_variant>(divide_scalar_op<float>(y)));
    return model(x);
  }

  transducer_model log_softmax(make_shared<transducer_variant>(tg::log_softmax_op()));

  transducer_model list_sum(make_shared<transducer_variant>(list_sum_op()));

  transducer_model tensor_sum(make_shared<transducer_variant>(tensor_sum_op()));

  transducer_model tensor_average(make_shared<transducer_variant>(tensor_average_op()));

  transducer_model tensor_std(make_shared<transducer_variant>(tensor_std_op()));

}


tg::value_placeholder tg::axis_sum(const tg::value_placeholder& x, const std::vector<unsigned long>& axes) {
  transducer_model model(make_shared<transducer_variant>(tensor_axis_sum_op(axes)));
  return model(x);
}

tg::value_placeholder tg::axis_average(const tg::value_placeholder& x, const std::vector<unsigned long>& axes) {
  transducer_model model(make_shared<transducer_variant>(tensor_axis_average_op(axes)));
  return model(x);
}

tg::value_placeholder tg::axis_std(const tg::value_placeholder& x, const std::vector<unsigned long>& axes) {
  transducer_model model(make_shared<transducer_variant>(tensor_axis_std_op(axes)));
  return model(x);
}

tg::value_placeholder tg::random_uniform(const tg::tensor_shape_t& shape, float min_val, float max_val) {

  transducer_model model(make_shared<transducer_variant>(random_uniform_op(shape, min_val, max_val)));
  return model.apply(std::vector<value_placeholder>{});
}

tg::value_placeholder tg::random_normal(const tg::tensor_shape_t& shape, float mean, float stddev) {
  transducer_model model(make_shared<transducer_variant>(random_normal_op(shape, mean, stddev)));
  return model.apply(std::vector<value_placeholder>{});
}

tg::value_placeholder tg::random_bernoulli(const tg::tensor_shape_t& shape, float p, float scale) {
  transducer_model model(make_shared<transducer_variant>(random_bernoulli_op(shape, p, scale)));
  return model.apply(std::vector<value_placeholder>{});
}


tg::value_placeholder tg::axis_max(const tg::value_placeholder& x, unsigned long axis) {
  transducer_model model(make_shared<transducer_variant>(tensor_axis_max_op(axis)));
  return model(x);
}

tg::value_placeholder tg::axis_min(const tg::value_placeholder& x, unsigned long axis) {
  transducer_model model(make_shared<transducer_variant>(tensor_axis_min_op(axis)));
  return model(x);
}

transducer_model tg::make_leaky_relu(float alpha) {
  return transducer_model(make_shared<transducer_variant>(leaky_relu_op(alpha)));
}

namespace tg {
  transducer_model tanh(make_shared<transducer_variant>(tanh_op()));
  transducer_model relu(make_shared<transducer_variant>(relu_op()));
  transducer_model sigmoid(make_shared<transducer_variant>(sigmoid_op()));
  transducer_model matmult(make_shared<transducer_variant>(matmult_op()));
  transducer_model pickneglogsoftmax(make_shared<transducer_variant>(pickneglogsoftmax_op()));
  transducer_model list_cmax(make_shared<transducer_variant>(list_max_op()));
  transducer_model list_cmin(make_shared<transducer_variant>(list_min_op()));
  transducer_model cmax(make_shared<transducer_variant>(binary_max_op()));
  transducer_model cmin(make_shared<transducer_variant>(binary_min_op()));
  transducer_model elu(make_shared<transducer_variant>(elu_op()));
  transducer_model selu(make_shared<transducer_variant>(selu_op()));
  transducer_model gelu(make_shared<transducer_variant>(gelu_op()));
  transducer_model sqrt(make_shared<transducer_variant>(sqrt_op()));
  transducer_model pow(make_shared<transducer_variant>(pow_op()));
  transducer_model log(make_shared<transducer_variant>(log_op()));
  transducer_model exp(make_shared<transducer_variant>(exp_op()));
  transducer_model pickneglogsigmoid(make_shared<transducer_variant>(pickneglogsigmoid_op()));
  transducer_model tensor_l2_norm(make_shared<transducer_variant>(tensor_l2_norm_op()));
}

value_placeholder tg::tensor_select(const value_placeholder& tensor, unsigned long idx, unsigned long axis) {
  return tensor_select(tensor, value_placeholder::constant(idx), value_placeholder::constant(axis));
}

value_placeholder
tg::tensor_select(const value_placeholder& tensor, const value_placeholder& idx, const value_placeholder& axis) {
  static auto model = tg::transducer_model(make_shared<transducer_variant>(tensor_pick_op()));
  return model(tensor, idx, axis);
}

value_placeholder
tg::tensor_slice(const value_placeholder& tensor, unsigned long start, unsigned long end, unsigned long axis) {
  return tensor_slice(tensor, value_placeholder::constant(start), value_placeholder::constant(end), value_placeholder::constant(axis));
}

value_placeholder
tg::tensor_slice(const value_placeholder& tensor, const value_placeholder& start, const value_placeholder& end,
                 const value_placeholder& axis) {
  static transducer_model model(make_shared<transducer_variant>(tensor_slice_op()));
  return model(tensor, start, end, axis);
}

value_placeholder tg::tensor_split(const value_placeholder& tensor, unsigned long axis) {
  return tensor_split(tensor, value_placeholder::constant(axis));
}

value_placeholder tg::tensor_split(const value_placeholder& tensor, const value_placeholder& axis) {
  static transducer_model model(make_shared<transducer_variant>(split_op()));
  return model(tensor, axis);
}

value_placeholder tg::tensor_reshape(const value_placeholder& tensor, tensor_shape_t shape) {
  transducer_model model(make_shared<transducer_variant>(tensor_reshape_op(std::move(shape))));
  return model(tensor);
}

value_placeholder tg::tensor_transpose(const value_placeholder& tensor, std::vector<unsigned int> axes) {
  transducer_model model(make_shared<transducer_variant>(tensor_transpose_op(std::move(axes))));
  return model(tensor);
}

value_placeholder tg::softmax(const value_placeholder& x, unsigned long axis) {
  transducer_model model(make_shared<transducer_variant>(softmax_op(axis)));
  return model(x);
}
