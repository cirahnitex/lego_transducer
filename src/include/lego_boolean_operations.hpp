//
// Created by Dekai WU and YAN Yuchen on 20200627.
//

#ifndef LEGO_LEGO_BOOLEAN_OPERATIONS_HPP
#define LEGO_LEGO_BOOLEAN_OPERATIONS_HPP
#include "value_placeholder.hpp"
#include "transducer_typed_value.hpp"


namespace tg {

  /**
   * \addtogroup boolean_operations
   * @{
   */

  /**
   * \brief Ensures that a scalar is a boolean scalar (holding value of either 0 or 1).
   *
   * This operation converts all non-zero values into 1.
   *
   * \param x The input scalar
   * \return The output scalar
   */
  value_placeholder to_boolean(const value_placeholder& x);

  /**
   * \brief Conditional expression
   *
   * Only one of the branches will be evaluated.
   *
   * \param cond The condition
   * \param val_if_true The value to output if the condition is satisfied
   * \param val_if_false The value to output if the condition is not satisfied
   * \return The output value
   */
  value_placeholder lazy_ifelse(const value_placeholder& cond, const value_placeholder& val_if_true, const value_placeholder& val_if_false);

  /**
   * \brief Conditional expression
   *
   * Both branches will be evaluated.
   *
   * \param cond The condition
   * \param val_if_true The value to output if the condition is satisfied
   * \param val_if_false The value to output if the condition is not satisfied
   * \return
   */
  value_placeholder eager_ifelse(const value_placeholder& cond, const value_placeholder& val_if_true, const value_placeholder& val_if_false);

  /**
   * \brief Conditional expression
   *
   * Performs what is equivalent to f(x,y,z)=x*y+(1-x)*z
   * Use this if computing your condition involves tensor computations (for example, from neural layers).
   *
   * When compared with lazy_ifelse and eager_ifelse, the evaluation of the condition is delayed, which means the computation is faster because tensor arithmetic are faster when grouped together and computed in large batches. However, this means both branches need to be expanded.
   *
   * <b>IMPORTANT!</b>
   * The two values you give must be tensors of the same shape. The condition you give must be a boolean scalar, otherwise you mess up the gradient.
   * Call to_boolean() beforehand if necessary.
   *
   * <b>Notes on backpropagation:</b>
   * The condition never receive error signal. The error signal back propagates into either the true branch or the false branch depending on the value of the condition.
   *
   * \param cond The condition
   * \param val_if_true The value to output if the condition is satisfied
   * \param val_if_false The value to output if the condition is not satisfied
   * \return The output value
   */
  value_placeholder soft_ifelse(const value_placeholder& cond, const value_placeholder& val_if_true, const value_placeholder& val_if_false);

  /**
   * \brief Negates a boolean scalar
   *
   * You must ensure that the input you give is a boolean scalar. Call to_boolean() beforehand if necessary.
   *
   * \param x The input boolean scalar
   * \return The negated boolean scalar
   */
  tg::value_placeholder operator!(const tg::value_placeholder& x);

  /**
   * \brief Logical And
   * \param x The first input boolean scalar
   * \param y The second input boolean scalar
   * \return Logical And of the two inputs
   */
  tg::value_placeholder operator&&(const tg::value_placeholder& x, const tg::value_placeholder& y);

  /**
   * \brief Logical Or
   * \param x The first input boolean scalar
   * \param y The second input boolean scalar
   * \return Logical Or of the two inputs
   */
  tg::value_placeholder operator||(const tg::value_placeholder& x, const tg::value_placeholder& y);

  /**
   * \brief Check whether two scalars are equal
   * \param x The first scalar
   * \param y The second scalar
   * \return A boolean scalar indicating whether the two inputs are equal
   */
  tg::value_placeholder operator==(const tg::value_placeholder& x, const tg::value_placeholder& y);

  /**
   * \brief Check whether two scalars are not equal
   * \param x The first scalar
   * \param y The second scalar
   * \return A boolean scalar indicating whether the two inputs are not equal
   */
  tg::value_placeholder operator!=(const tg::value_placeholder& x, const tg::value_placeholder& y);

  /**
   * \brief Compare two scalars
   * \param x the first scalar
   * \param y the second scalar
   * \return A boolean scalar indicating whether the first scalar is less than the second scalar
   */
  tg::value_placeholder operator<(const tg::value_placeholder& x, const tg::value_placeholder& y);

  /**
   * \brief Compare two scalars
   * \param x the first scalar
   * \param y the second scalar
   * \return A boolean scalar indicating whether the first scalar is greater than the second scalar
   */
  tg::value_placeholder operator>(const tg::value_placeholder& x, const tg::value_placeholder& y);

  /** @} */
}




#endif //LEGO_LEGO_BOOLEAN_OPERATIONS_HPP
