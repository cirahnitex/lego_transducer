//
// Created by Dekai WU and YAN Yuchen on 20200506.
//

#ifndef LEGO_LEGO_TENSOR_OPERATIONS_HPP
#define LEGO_LEGO_TENSOR_OPERATIONS_HPP

#include "value_placeholder.hpp"
#include "transducer_typed_value.hpp"
#include "transducer_model.hpp"


namespace tg {

  /**
   * \addtogroup tensor_operations
   * @{
   */


  /**
   * \brief Arithmetic negation
   * \details Compute the negative of a tensor (or scalar)
   * \param x the value to negate
   * \return The negation of x
   */
  tg::value_placeholder operator-(const tg::value_placeholder& x);

  /**
   * \brief Elementwise addition
   *
   * \details Add a tensor (or scalar) with another tensor (or scalar).
   *          If both inputs are tensors, they have to be of the same shape.
   *
   * \param x The first input tensor or scalar
   * \param y The second input tensor or scalar
   * \return The Addition result
   */
  tg::value_placeholder operator+(const tg::value_placeholder& x, const tg::value_placeholder& y);

  tg::value_placeholder operator+(const tg::value_placeholder& x, long y);

  tg::value_placeholder operator+(const tg::value_placeholder& x, float y);

  template<typename T>
  inline typename std::enable_if<std::is_integral_v<T>, tg::value_placeholder>::type operator+(const tg::value_placeholder& x, T y) {
    return operator+(x, (long)y);
  }

  template<typename T>
  inline typename std::enable_if<std::is_floating_point_v<T>, tg::value_placeholder>::type operator+(const tg::value_placeholder& x, T y) {
    return operator+(x, (float)y);
  }

  template<typename T>
  inline typename std::enable_if<std::is_arithmetic_v<T>, tg::value_placeholder>::type operator+(T x, const tg::value_placeholder& y) {
    return operator+(y, x);
  }

  /**
   * \brief Elementwise subtraction
   * \details Subtract a tensor (or scalar) from another tensor (or scalar).
   *          If both inputs are tensors, they have to be of the same shape.
   * \param x The first input tensor or scalar
   * \param y The second input tensor or scalar
   * \return The subtraction result
   */
  tg::value_placeholder operator-(const tg::value_placeholder& x, const tg::value_placeholder& y);

  tg::value_placeholder operator-(const tg::value_placeholder& x, long y);

  tg::value_placeholder operator-(const tg::value_placeholder& x, float y);

  template<typename T>
  inline typename std::enable_if<std::is_integral_v<T>, tg::value_placeholder>::type operator-(const tg::value_placeholder& x, T y) {
    return operator-(x, (long)y);
  }

  template<typename T>
  inline typename std::enable_if<std::is_floating_point_v<T>, tg::value_placeholder>::type operator-(const tg::value_placeholder& x, T y) {
    return operator-(x, (float)y);
  }

  tg::value_placeholder operator-(long x, const tg::value_placeholder& y);

  tg::value_placeholder operator-(float x, const tg::value_placeholder& y);

  template<typename T>
  inline typename std::enable_if<std::is_integral_v<T>, tg::value_placeholder>::type operator-(T x, const tg::value_placeholder& y) {
    return operator-(y, (long)x);
  }

  template<typename T>
  inline typename std::enable_if<std::is_floating_point_v<T>, tg::value_placeholder>::type operator-(T x, const tg::value_placeholder& y) {
    return operator-(y, (float)x);
  }

  /**
   * \brief Elementwise multiplication
   * \details Multiply a tensor (or scalar) with another tensor (or scalar) elementwise.
   *          If both inputs are tensors, they have to be of the same shape.
   *          If you want to perform a matrix multiplication, see tg::matmult()
   * \param x The first input tensor or scalar
   * \param y The second input tensor or scalar
   * \return The subtraction result
   */
  tg::value_placeholder operator*(const tg::value_placeholder& x, const tg::value_placeholder& y);

  tg::value_placeholder operator*(const tg::value_placeholder& x, long y);

  tg::value_placeholder operator*(const tg::value_placeholder& x, float y);

  template<typename T>
  inline typename std::enable_if<std::is_integral_v<T>, tg::value_placeholder>::type operator*(const tg::value_placeholder& x, T y) {
    return operator*(x, (long)y);
  }

  template<typename T>
  inline typename std::enable_if<std::is_floating_point_v<T>, tg::value_placeholder>::type operator*(const tg::value_placeholder& x, T y) {
    return operator*(x, (float)y);
  }

  template<typename T>
  inline typename std::enable_if<std::is_arithmetic_v<T>, tg::value_placeholder>::type operator*(T x, const tg::value_placeholder& y) {
    return operator*(y, x);
  }

  /**
   * \brief Elementwise division
   * \details Divide a tensor (or scalar) by another tensor (or scalar) elementwise.
   *          If both inputs are tensors, they have to be of the same shape.
   * \param x The first input tensor or scalar
   * \param y The second input tensor or scalar
   * \return The division result
   */
  tg::value_placeholder operator/(const tg::value_placeholder& x, const tg::value_placeholder& y);

  tg::value_placeholder operator/(const tg::value_placeholder& x, long y);

  tg::value_placeholder operator/(const tg::value_placeholder& x, float y);

  template<typename T>
  inline typename std::enable_if<std::is_integral_v<T>, tg::value_placeholder>::type operator/(const tg::value_placeholder& x, T y) {
    return operator/(x, (long)y);
  }

  template<typename T>
  inline typename std::enable_if<std::is_floating_point_v<T>, tg::value_placeholder>::type operator/(const tg::value_placeholder& x, T y) {
    return operator/(x, (float)y);
  }

  /**
   * \brief Softmax operation
   * \param x The tensor to compute softmax on
   * \param axis The axis along which softmax is computed
   * \return The result of computing softmax
   */
  value_placeholder softmax(const value_placeholder& x, unsigned long axis = 0);

  /**
   * \brief log softmax
   * \details The log softmax function normalizes each column to ensure that all
   *          values are between 0 and 1 and add to one by applying
   *          \f$\frac{e^{x_i}}{\sum_j e^{x_j}}\f$, then taking the log
   * \param x a tensor of rank 1 or 2
   * \return the tensor after calculating log softmax
   */
  extern transducer_model log_softmax;

  /**
   * \brief Natural log
   *
   * Computes elementwise natural log
   *
   * \param x The input tensor
   * \return The output tensor
   */
  extern transducer_model log;

  /**
   * \brief Natural exponent
   *
   * Computes elementwise e^x
   *
   * \param x The input tensor
   * \return The output tensor
   */
  extern transducer_model exp;

  /**
   * \brief Hyperbolic tangent
   * \details Calculate elementwise of the hyperbolic tangent
   *
   * \param x The input tensor
   *
   * \return An expression where the ith element is equal to tanh(x_i)
   */
  extern transducer_model tanh;

  /**
   * \brief ReLU
   * \details Calculate elementwise the recitifer (ReLU) function f(x) = max(x,0)
   *
   * \param x The input tensor
   *
   * \return The output tensor f(x)
   */
  extern transducer_model relu;

  /**
   * \brief Logistic sigmoid function
   * \details Calculate elementwise \f$ f(x) = 1/(1+e^{-x_i}) \f$
   *
   * \param x The input tensor
   *
   * \return The output tensor f(x)
   */
  extern transducer_model sigmoid;

  /**
   * \brief Create a Leaky ReLU transducer
   * \details Calculate elementwise \f$ f(x)=\begin{cases}x & x > 0\\ \alpha x & x \leq 0\end{cases} \f$
   * \param alpha the coefficient when x <= 0
   * \return The Leaky ReLU transducer
   */
  transducer_model make_leaky_relu(float alpha=0.1);

  /**
   * \brief Exponential Linear Unit
   *
   * Calculate elementwise the function
   *
   * \f$
   * f(x) = \left\{\begin{array}{lr}
   *            x, & \text{if } x>0\\
   *            \alpha (e^{x} - 1), & \text{if }x\leq 0\\
   *          \end{array}\right.
   * \f$
   *
   * Reference: [Clevert et al., 2015](https://arxiv.org/abs/1511.07289v5)
   *
   * \param x The input tensor
   *
   * \return The output tensor f(x)
   */
  extern transducer_model elu;

  /**
   * \brief Scaled Exponential Linear Unit (SELU)
   * \details Calculate elementwise the function
   *
   * \f$
   * f(x) = \lambda \left\{\begin{array}{lr}
   *            x, & \text{if } x>0\\
   *            \alpha (e^{x} - 1), & \text{if }x\leq 0\\
   *          \end{array}\right.
   * \f$
   *
   * With
   * \f$
   * \begin{split}
   * \lambda &= 1.05070\\
   * \alpha &= 1.67326\\
   * \end{split}
   * \f$
   *
   * Reference: [Klambaouer et al., 2017](https://arxiv.org/abs/1706.02515)
   *
   * \param x The input tensor
   *
   * \return The output tensor f(x)
   */
  extern transducer_model selu;

  /**
   * \brief Gaussian Error Linear Unit (GELU)
   * \details Calculate elementwise the function
   *
   * \f$
   * f(x) = 0.5x\left(1+\text{tanh}\left(\sqrt{2/\pi}(x+0.044715x^3)\right)\right)
   * \f$
   *
   * Reference: [Dan Hendrycks et al., 2016](https://arxiv.org/abs/1606.08415)
   *
   * \param x The input tensor
   * \return The output tensor f(x)
   */
  extern transducer_model gelu;

  /**
   * \brief Matrix multiplication
   * \details performs matrix multiplication of two matrices. Each input can be a tensor of rank 1 or 2.
   * Rank-1 tensors are considered to be column vectors.
   * \param x The first input tensor
   * \param y The second input tensor
   * \return The matrix multiplication result
   */
  extern transducer_model matmult;

  /**
   * \brief Negative softmax log likelihood
   *
   * This function performs:
   *
   * \f$
   * f(\mathbf{x}, i)=-\log(\text{softmax}(\mathbf{x}))_i
   * \f$
   *
   * Where x is the logits vector and i is the oracle label ID.
   *
   * This is the standard loss function for N choose 1 style classification
   *
   * \param x A rank-1 tensor, holding a list of scores
   * \param v The index with which to calculate the loss
   *
   * \return The negative log likelihood of element ``v`` after taking the softmax
   */
  extern transducer_model pickneglogsoftmax;

  /**
   * \brief Negative sigmoid log likelihood
   *
   * This function performs
   *
   * \f$
   * f(x,y)=\begin{cases}-\log(\text{sigmoid}(x)) & y = 1 \\ -\log(\text{sigmoid}(-x)) & y = 0\end{cases}
   * \f$
   *
   * Where x is the logit and y is the oracle.
   *
   * This is the standard loss function for binary classification.
   *
   * \param logit A scalar value. >0 means true prediction and <0 means false prediction
   * \param oracle A scalar value representing the oracle output. 0 means false and 1 means true
   * \return The negative sigmoid log likelihood.
   */
  extern transducer_model pickneglogsigmoid;

  /**
   * \brief Square root
   * \details Calculates elementwise square root
   * \param x The input tensor
   * \return the output tensor
   */
  extern transducer_model sqrt;

  /**
   * \brief Exponential function
   * \details Calculate elementwise the function \f$ f(x) = x^y \f$
   * \param base The base value x, can be a tensor.
   * \param exponent The exponent value y, must be a scalar.
   * \return The output tensor f(x)
   */
  extern transducer_model pow;

  /**
   * \brief The L2 norm of a tensor
   *
   * \f$\sqrt{\sum_i x_i^2}\f$.
   *
   * \param x The tensor to compute norm
   * \return The L2 norm of the tensor
   */
  extern transducer_model tensor_l2_norm;

  /**
   * \brief The squared distance between two tensors
   *
   * \f$\sum_i (x_i-y_i)^2\f$
   *
   * \param x The first tensor
   * \param y The second tensor
   * \return the squared distance between these two tensors
   */
  extern transducer_model tensor_squared_distance;

  /**
   * \brief Tensor concatenation
   *
   * Concatenates a list of tensors into a single tensor along an axis
   *
   * \param tensors The list of tensors to concatenate.
   *                Must be of the same shape except along the axis of concatenation
   * \param axis The axis along which to concatenate
   * \return The concatenated tensor
   */
  value_placeholder tensor_concat(const std::vector<value_placeholder>& tensors, unsigned long axis = 0);

  /**
   * \brief Tensor concatenation
   *
   * Concatenates a list of tensors into a single tensor along an axis.
   *
   * This overload takes a transducer (that returns a list of values) instead of a list of transducers directly.
   *
   * \param tensors The transducer that outputs a list of values to concatenate.
   *                Must be of the same shape except along the axis of concatenation
   * \param axis The axis along which to concatenate
   * \return The concatenated tensor
   */
  value_placeholder tensor_concat(const value_placeholder& tensors, unsigned long axis = 0);


  /**
   * \brief Tensor summation
   * \details Sum a list of tensors. All tensors must be of the same shape.
   * \param tensors The list of tensors to sum
   * \return The summed tensor
   */
  extern transducer_model list_sum;

  /**
   * \brief Elementwise max
   * \details Compute the elementwise max for a list of tensors. All tensors must be of the same shape.
   * \param tensors The list of tensors to maximize
   * \return The maximized tensor
   */
  extern transducer_model list_cmax;

  /**
   * \brief Elementwise min
   * \details Compute the elementwise min for a list of tensors. All tensors must be of the same shape.
   * \param tensors The list of tensors to minimize
   * \return The minimized tensor
   */
  extern transducer_model list_cmin;

  /**
   * \brief Elementwise max
   * \brief Compute the elementwise max for two tensors. These two tensors must be of the same shape.
   * \param x The first tensor
   * \param y The second tensor
   * \return The maximized tensor
   */
  extern transducer_model cmax;

  /**
   * \brief Elementwise min
   * \brief Compute the elementwise min for two tensors. These two tensors must be of the same shape.
   * \param x The first tensor
   * \param y The second tensor
   * \return The minimized tensor
   */
  extern transducer_model cmin;

  /**
   * \brief Find index of maximum value
   * \details Returns the index of the maximum value of a Rank-1 tensor.
   * <b>IMPORTANT</b>
   * This operation triggers an immediate computation graph evaluation.
   * Only use this near the end of your transducer.
   * \param tensor1d the model which outputs the Rank-1 tensor
   * \return a model which outputs the index of the maximum value
   */
  extern transducer_model max_index_of_tensor1d;

  /**
   * \brief Sum all elements in a tensor
   * \details See axis_sum() if you want to sum across some specific axes.
   * \param x The tensor to sum
   * \return The summed value as a scalar
   */
  extern transducer_model tensor_sum;

  /**
   * \brief Average all elements in a tensor
   * \details See axis_average() if you want to average across some specific axes
   * \param x The tensor to average
   * \return The average value as a scalar
   */
  extern transducer_model tensor_average;

  /**
   * \brief Calculate standard deviation of all elements in a tensor
   * \detaials See axis_std() if you compute across some specific axes.
   * \param x The tensor to average
   * \return The standard deviation as a scalar
   */
  extern transducer_model tensor_std;

  /**
   * \brief Sum elements in a tensor along one or more axes
   * \param x the model which outputs the tensor to sum
   * \param axes the axes along which to sum
   * \return a model which outputs the summed tensor
   */
  value_placeholder axis_sum(const value_placeholder& x, const std::vector<unsigned long>& axes);

  /**
   * \brief Average elements in a tensor along one or more axes
   * \param x The tensor to average
   * \param axes The axes along which to average
   * \return a The averaged tensor
   */
  value_placeholder axis_average(const value_placeholder& x, const std::vector<unsigned long>& axes);

  /**
   * \brief Compute standard deviation of elements in a tensor along one or more axes
   * \param x The tensor to compute
   * \param axes The axes along which to compute
   * \return a The standard deviation tensor
   */
  value_placeholder axis_std(const value_placeholder& x, const std::vector<unsigned long>& axes);

  /**
   * \brief Max out a tensor through an axis.
   * \details The returned tensor will have rank reduced by 1.
   * \param x The tensor to maximize
   * \param axis the axis along which to max through
   * \return The maximized tensor
   */
  value_placeholder axis_max(const value_placeholder& x, unsigned long axis);

  /**
   * \brief Min out a tensor through an axis.
   * \details The returned tensor will have rank reduced by 1.
   * \param x The tensor to minimize
   * \param axis The axis along which to min through
   * \return The minimized tensor
   */
  value_placeholder axis_min(const value_placeholder& x, unsigned long axis);

  /**
   * \brief Random uniform tensor generator
   * \details Generate a tensor filled with random values according to uniform distribution.
   * \param shape the shape of the generated tensor
   * \param min_val the minimum possible value
   * \param max_val the maximum possible value
   * \return The generated tensor
   */
  value_placeholder random_uniform(const tensor_shape_t& shape, float min_val = 0, float max_val = 1);


  /**
   * \brief Random normal tensor generator
   * \details Generate a tensor filled with random values according to normal distribution.
   * \param shape the shape of the generated tensor
   * \param mean the mean of the normal distribution
   * \param stddev the standard deviation of the normal distribution
   * \return The generated tensor
   */
  value_placeholder random_normal(const tensor_shape_t& shape, float mean = 0, float stddev = 1);


  /**
   * \brief Random bernoulli tensor generator
   * \details Generate a tensor filled with random values according to bernoulli distribution.
   * The value is binary. Will be either 0 (when inactive) or `scale` (when active).
   * \param shape the shape of the generated tensor
   * \param p the probability of "active"
   * \param scale the value when "active"
   * \return a model which outputs the generated tensor
   */
  value_placeholder random_bernoulli(const tensor_shape_t& shape, float p, float scale = 1);

  /**
   * \brief Selects a sub-tensor from a tensor
   *
   * Picks an index along an axis of the original tensor.
   * The resulting sub-tensor will have its rank reduced by 1.
   *
   * For example, if you select the following tensor:
   *
   * \f$
   * \begin{bmatrix}1 & 3 & 5 \\ 2 & 4 & 6 \end{bmatrix}
   * \f$
   *
   * along axis #1 at index #0, you get  \f$ \begin{bmatrix}1 \\ 2 \end{bmatrix} \f$
   *
   * \param tensor The tensor to select from
   * \param idx The index of the dimension to select
   * \param axis The axis along which the dimension index was counted at.
   * \return The selected sub-tensor
   */
  value_placeholder tensor_select(const value_placeholder& tensor, unsigned long idx, unsigned long axis);

  value_placeholder tensor_select(const value_placeholder& tensor, const value_placeholder& idx, const value_placeholder& axis);

  /**
   * \brief Selects a consecutive slice along an axis of a tensor
   *
   * For example, if you slice the following tensor:
   *
   * \f$
   * \begin{bmatrix}1 & 3 & 5 & 7 \\ 2 & 4 & 6 & 8 \end{bmatrix}
   * \f$
   *
   * along axis #1, start and index #1 and ending at index #3, you get:
   *
   * \f$
   * \begin{bmatrix}3 & 5 \\ 4 & 6 \end{bmatrix}
   * \f$
   *
   * \param tensor The tensor to slice
   * \param begin The start index (inclusive) along the axis
   * \param end The end index (exclusive) along the axis
   * \param axis The axis along which to slice
   * \return The sliced tensor
   */
  value_placeholder tensor_slice(const value_placeholder& tensor, unsigned long start, unsigned long end, unsigned long axis);

  value_placeholder tensor_slice(const value_placeholder& tensor, const value_placeholder& start, const value_placeholder& end, const value_placeholder& axis);

  /**
   * \brief Split a tensor into a list of tensors along an axis
   *
   * For example, if you split the following tensor:
   *
   * \f$
   * \begin{bmatrix}1 & 3 & 5 \\ 2 & 4 & 6 \end{bmatrix}
   * \f$
   *
   * by axis #0, you will get two tensors:
   * \f$ \begin{bmatrix}1 \\ 3 \\ 5 \end{bmatrix} \f$ and \f$ \begin{bmatrix}2 \\ 4 \\ 6 \end{bmatrix} \f$
   *
   * \param tensor The tensor to split
   * \param axis The axis to split along
   * \return The split tensors as a list.
   */
  value_placeholder tensor_split(const value_placeholder& tensor, unsigned long axis);
  value_placeholder tensor_split(const value_placeholder& tensor, const value_placeholder& axis);

  /**
   * \brief Reshape a tensor
   * \param tensor The tensor to reshape
   * \param shape The output tensor shape
   * \return The reshaped tensor
   */
  value_placeholder tensor_reshape(const value_placeholder& tensor, tensor_shape_t shape);

  /**
   * \brief Transpose a matrix, or more generally, shuffle the axes of a tensor.
   *
   * You need to provide a permutation of the original axis indices.
   *
   * For example, if you want to transpose a matrix, you need to invert the original axis.
   * The original axis indices are {0, 1}, so the shuffled indices are {1, 0}
   *
   * \param tensor The tensor to transpose
   * \param axes The shuffled axis indices
   * \return
   */
  value_placeholder tensor_transpose(const value_placeholder& tensor, std::vector<unsigned> axes = {1, 0});
  /** @} */
}

#endif //LEGO_LEGO_TENSOR_OPERATIONS_HPP
