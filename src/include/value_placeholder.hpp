//
// Created by Dekai WU and YAN Yuchen on 20201124.
//

#ifndef LEGO_VALUE_PLACEHOLDER_HPP
#define LEGO_VALUE_PLACEHOLDER_HPP
#include "transducer_typed_value.hpp"
#include "generate_type_consistent_tuple.hpp"
#include "lego_primitive_types.hpp"
#include <fstream>
#include <functional>
#include <variant>
#include <memory>
namespace tg {

  class transducer_variant;
  class lambda_transducer_model_construction_guard;
  /**
   * \brief Represents a value placeholder while constructing a transducer using lambda syntax.
   *
   * This value placeholder is implemented as a pointer to a local value in its owner transducer.
   */
  class value_placeholder {

    /**
     * \brief The nesting depth of the owner transducer
     *
     * This value can be used to lookup the owner transducer while transducing.
     */
    unsigned long owner_nesting_depth{};

    unsigned long value_idx{};

    value_placeholder(unsigned long owner_nesting_depth, unsigned long value_idx):owner_nesting_depth(owner_nesting_depth), value_idx(value_idx) {
    };

    friend lambda_transducer_model;
    friend lambda_transducer_model_construction_guard;
  public:
    
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(owner_nesting_depth, value_idx);
    }
    
    value_placeholder() = default;
    value_placeholder(const value_placeholder&) = default;
    value_placeholder(value_placeholder&&) noexcept = default;
    value_placeholder& operator=(const value_placeholder&) = default;
    value_placeholder& operator=(value_placeholder&&) noexcept = default;

    /**
     * \brief Create value placeholder holding a constant value
     * \param x The value to hold, can be any value_t constructable value
     * \return The value placeholder holding the constant value
     */
    static value_placeholder constant(value_t x);

    template<typename T>
    static value_placeholder constant(const T& x) {
      return value_placeholder::constant(value_t(x));
    }

    /**
     * \brief Create a value placeholder holding an empty list
     * \return The value placeholder holding the constant list
     */
    static value_placeholder empty_list() {
      return value_placeholder::constant(value_t::make_list());
    }

    /**
     * \brief Create a constant value placeholder holding a tensor filled with zeros
     * \param shape the shape of the tensor
     * \return the constant value placeholder
     */
    static value_placeholder zeros(const tensor_shape_t& shape);

    /**
     * \brief Create a constant value placeholder holding a tensor filled with ones
     * \param shape the shape of the tensor
     * \return the constant value placeholder
     */
    static value_placeholder ones(const tensor_shape_t& shape);


    /**
     * \brief Picks a value at given index (if this model outputs a list)
     * \param idx a transducer model that outputs the index to pick
     * \return A model that outputs the value of the list at given index
     */
    value_placeholder operator[](const value_placeholder& idx) const;

    /**
     * \brief Picks a value at given index (if this model outputs a list)
     * \param idx the index to pick. Supports negative indexing.
     * \return a model that outputs the value of the list at the given index
     */
    value_placeholder operator[](long idx) const;

    friend std::ostream& operator<<(std::ostream& os, const value_placeholder& vp);
  };
  std::ostream& operator<<(std::ostream& os, const value_placeholder& vp);
}



#endif //LEGO_VALUE_PLACEHOLDER_HPP
