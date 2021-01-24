//
// Created by Dekai WU and YAN Yuchen on 20201203.
//

#ifndef LEGO_COMPOSED_TRANSDUCER_MODEL_HPP
#define LEGO_COMPOSED_TRANSDUCER_MODEL_HPP

#include <deque>
#include <memory>
#include "include/transducer_typed_value.hpp"
namespace tg {
  class transducer_variant;

  /**
   * \brief Represents a transducer that is a composition of many unary transducers.
   *
   * The compose operation will result in such transducer.
   */
  class composed_transducer_model {
    // value will pass through the pipeline from begin to end
    std::deque<std::shared_ptr<transducer_variant>> pipeline;
    explicit composed_transducer_model(std::deque<std::shared_ptr<transducer_variant>> pipeline): pipeline(std::move(pipeline)) {}
  public:

    template<typename Archive>
    void serialize(Archive& ar) {
      ar(pipeline);
    }

    composed_transducer_model() = default;
    composed_transducer_model(const composed_transducer_model&) = default;
    composed_transducer_model(composed_transducer_model&&) noexcept = default;
    composed_transducer_model& operator=(const composed_transducer_model&) = default;
    composed_transducer_model& operator=(composed_transducer_model&&) noexcept = default;

    /**
     * \brief Compose transducers
     *
     * Creates a composed transducer y=f(g(h(x))) given f,g,h
     *
     * \param pieces The list of transducers to compose
     * \return The composed transducer
     */
    static composed_transducer_model compose_from(std::initializer_list<std::shared_ptr<transducer_variant>> pieces);

    value_t transduce(const value_t& x);

    [[nodiscard]] std::string default_name() const;

    std::vector<std::shared_ptr<transducer_variant>> nested_transducers();
  };
}

#endif
