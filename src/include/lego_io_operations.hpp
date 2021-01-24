//
// Created by Dekai WU and YAN Yuchen on 20200511.
//

#ifndef LEGO_LEGO_IO_OPERATIONS_HPP
#define LEGO_LEGO_IO_OPERATIONS_HPP

#include "value_placeholder.hpp"
#include "transducer_typed_value.hpp"
#include "transducer_model.hpp"
namespace tg {
  /**
   * \addtogroup io_operations
   *
   * @{
   */

  /**
   * \brief Print the input value
   *
   * This is an identity transducer with a side effect of printing out the input value. Use this if you need to debug your transducers.
   *
   * \param x The input value to print
   * \param label Also prints out a label, so that the printing can be more meaningful.
   * \return
   */
  value_placeholder trace(const value_placeholder& x, const std::string& label = "");

  extern transducer_model identity;
  ///@}
}

#endif //LEGO_LEGO_IO_OPERATIONS_HPP
