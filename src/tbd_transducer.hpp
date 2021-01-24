//
// Created by Dekai WU and YAN Yuchen on 20200707.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_TBD_TRANSDUCER_HPP
#define LEGO_TBD_TRANSDUCER_HPP
#include "include/transducer_typed_value.hpp"
namespace tg {

  /**
   * \brief A transducer that may not be determined yet.
   *
   * A TBD transducer is a transducer that will become another transducer later
   *
   * This transducer is used for supporting recursion.
   *
   * When an empty tg::transducer_model composes with other transducers, it immediately holds an empty TBD transducer and composes the TBD transducer with others. Later on when the transducer model is assignment with a concrete transducer, it replaces the concrete transducer with its TBD inplace (so that all pointers points that were pointing to this TBD will instead point to a concrete transducer).
   *
   */
  class tbd_transducer {
  public:

    template<typename Archive>
    void serialize(Archive& ar) {

    }

    template<typename ...Args>
    value_t transduce(Args...) {
      throw std::runtime_error("Cannot apply a TBD transducer");
    }

    std::string default_name() const;

  };

}

#endif
