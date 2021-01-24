
//
// Created by Dekai WU and YAN Yuchen on 20200512.
//

#ifndef LEGO_LEGO_TENSOR_HPP
#define LEGO_LEGO_TENSOR_HPP
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include "lego_serialization_helper.hpp"

namespace dynet {
  template<typename Archive>
  void serialize(Archive& ar, dynet::Dim& dim) {
    ar(dim.d, dim.nd, dim.bd);
  }
}

namespace tg {

  /**
   * \addtogroup data_values
   *
   * @{
   */

  using symbolic_tensor_t = dynet::Expression;


  /**
   * \brief Represents the shape of a tensor
   */
  using tensor_shape_t = std::vector<unsigned long>;

  /**
   * \brief Express a tensor shape into a human-readible string
   * \param shape The tensor shape to express
   * \return a human-readible string
   */
  std::string print_tensor_shape(const tensor_shape_t& shape);

  /**
   * \brief Convert from tg::tensor_shape_t to dyent::Dim
   * \param dim The tensor shape
   * \return The same tensor shape represented in dynet::Dim
   */
  dynet::Dim to_dynet_dim(const tensor_shape_t& dim);

  /**
   * \brief Convert from dynet::Dim to tg::tensor_shape_t.
   *
   * The "batch size" in dynet::Dim is discarded.
   *
   * \param dynet_dim The tensor shape
   * \return The same tensor shape represented in tg::tensor_shape_t
   */
  tensor_shape_t from_dynet_dim(const dynet::Dim& dynet_dim);

  /**
   * \breif Calculate the number of values in a tensor given the tensor shape.
   *
   * Defined as the produce of all axis length.
   *
   * \param shape the tensor shape
   * \return the number of values in the tensor
   */
  unsigned long tensor_num_values(const tensor_shape_t& shape);

  /**
   * \brief Represents a tensor.
   *
   * The value is interpreted in column major. The values {1, 2, 3, 4, 5, 6} when view in shape {2, 3} will be:
   *
   * \$f
   * \begin{bmatrix}1 & 3 & 5 \\2 & 4 & 6 \end{bmatrix}
   * \$f
   *
   */
  struct tensor_t {
    /**
     * \brief The values
     */
    std::vector<float> values;

    /**
     * \brief The tensor shape
     */
    tensor_shape_t shape;

    template<typename Archive>
    void serialize(Archive& ar) {
      ar(values, shape);
    }

    tensor_t() = default;
    tensor_t(const tensor_t&) = default;
    tensor_t(tensor_t&&) noexcept = default;
    tensor_t& operator=(const tensor_t&) = default;
    tensor_t& operator=(tensor_t&&) noexcept = default;

    /**
     * \brief Constructs a tensor given values and shape
     * \param values The values of the tensor
     * \param shape The shape of the tensor
     */
    tensor_t(std::vector<float> values, tensor_shape_t shape);

    /**
     * \brief Constructs a rank-1 tensor given values
     *
     * \param values The values of the tensor
     * \param shape The shape of the tensor
     */
    explicit tensor_t(std::vector<float> values);

    /**
     * \brief Constructs a tensor given values and shape
     *
     * This this the generic version of tensor_t(std::vector<float> values, tensor_shape_t shape)
     *
     * \param values The values of the tensor (as an iterable of numbers)
     * \param shape The shape of the tensor
     */
    template<typename T>
    tensor_t(const T& values, tensor_shape_t shape):tensor_t(std::vector<float>(values.begin(), values.end()), std::move(shape)) {
    }

    /**
    * \brief Constructs a rank-1 tensor given values
    *
    * This this the generic version of tensor_t(std::vector<float> values), where you can pass any iterable of numbers.
     *
    * \param values The values of the tensor
    * \param _ SFINAE type check helper to ensure that the template T is iterable
    */
    template<typename T>
    explicit tensor_t(const T& values, std::void_t<decltype(std::begin(std::declval<T>())),
      decltype(std::end(std::declval<T>()))>* _ = nullptr):tensor_t(std::vector<float>(values.begin(), values.end())) {
    }

    friend std::ostream& operator<<(std::ostream& os, const tensor_t& x);

    /**
     * \brief The maximum number of values for a tensor that when printed, displays all its internal values.
     *
     * Tensors with more values will not show its content when printed, just showing the shape.
     */
    static const unsigned long MAX_TENSOR_ELEMS_TO_PRINT = 32;

    /**
     * \brief Convert from a dynet::Tensor into a tg::tensor_t
     * \param dynet_tensor The tensor
     * \return The same tensor in tg::tensor_t
     */
    static tensor_t from_dynet_tensor(const dynet::Tensor& dynet_tensor);

    /**
     * \brief Create a tensor filled with zeros
     * \param shape The shape of the tensor
     * \return The tensor filled with zeros
     */
    static tensor_t zeros(const tensor_shape_t& shape);

    /**
     * \brief Create a tensor filled with ones
     * \param shape The shape of the tensor
     * \return the tensor filled with ones
     */
    static tensor_t ones(const tensor_shape_t& shape);
  };

  /**
   * \brief Prints out a tensor in human-readible form
   *
   * If the tensor is too long, its content will not be shown. Only its shape is shown.
   *
   * \param os The output stream to print to
   * \param x The tensor to print
   * \return The same output stream
   */
  std::ostream& operator<<(std::ostream& os, const tg::tensor_t& x);

  /// @}
}



#endif //LEGO_LEGO_TENSOR_HPP
