//
// Created by Dekai WU and YAN Yuchen on 20200525.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_GENERIC_RNN_MODEL_HPP
#define LEGO_GENERIC_RNN_MODEL_HPP

#include "dropout_model.hpp"
#include "rnn_cells.hpp"

namespace tg {
  /**
   * \beirf Represents an unidirectional, single stack RNN.
   *
   * Can be of any RNN cell.
   */
  class generic_rnn_model {
    std::shared_ptr<rnn_cell_base> rnn_cell_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(rnn_cell_m);
    }

    generic_rnn_model() = default;

    generic_rnn_model(const generic_rnn_model&) = default;

    generic_rnn_model(generic_rnn_model&&) noexcept = default;

    generic_rnn_model& operator=(const generic_rnn_model&) = default;

    generic_rnn_model& operator=(generic_rnn_model&&) noexcept = default;

    explicit generic_rnn_model(std::shared_ptr<rnn_cell_base> rnn_cell);

    /**
     * Apply the RNN model on a sequence of inputs
     * \param init_state the initial state, pass NULL for default state.
     * \param xs the sequence of inputs
     * \return (1) the sequence of outputs
     *         (2) the final state
     */
    std::pair<std::vector<value_t>, value_t> transduce_impl(const value_t& init_state, const std::vector<value_t>& xs);

    value_t transduce(const value_t& init_state, const value_t& xs);

    std::string default_name() const;

  };

  /**
   * \brief Represents an unidirectional, multistack RNN.
   *
   * Can be of any RNN cell.
   */
  class generic_stacked_rnn_model {
    std::vector<generic_rnn_model> rnns_m;
    std::shared_ptr<dropout_model> dropout_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(rnns_m, dropout_m);
    }

    generic_stacked_rnn_model() = default;

    generic_stacked_rnn_model(const generic_stacked_rnn_model&) = default;

    generic_stacked_rnn_model(generic_stacked_rnn_model&&) noexcept = default;

    generic_stacked_rnn_model& operator=(const generic_stacked_rnn_model&) = default;

    generic_stacked_rnn_model& operator=(generic_stacked_rnn_model&&) noexcept = default;

    /**
     * Constructs a stacked RNN from a list of RNN cells
     * \param rnn_cells the list of RNN cells
     * \param dropout_rate if > 0, applies dropout between stacks
     */
    explicit generic_stacked_rnn_model(const std::vector<std::shared_ptr<rnn_cell_base>>& rnn_cells,
                                       float dropout_rate = 0);

    /**
     * Apply the RNN model on a sequence of inputs
     * \param init_state the list of init states (from bottom to top), pass NULL[] for default init state.
     * \param xs the sequence of inputs
     * \return (1) the list of outputs
     *         (2) the final states (from bottom to top)
     */
    std::pair<std::vector<value_t>, std::vector<value_t>>
    transduce_impl(const std::vector<value_t>& init_state, const std::vector<value_t>& xs);

    // same as transduce_impl. The only difference is that for the init_state param, pass NULL (instead of NULL[]) for default init state.
    value_t transduce(const value_t& init_state, const value_t& xs);

    unsigned long num_stacks() const;

    std::string default_name() const;

  };

  /**
   * \beirf Represents a bidirectional, single stack RNN.
   *
   * Can be of any RNN cell.
   */
  class generic_bidirectional_rnn_model {
    std::shared_ptr<rnn_cell_base> forward_cell_m;
    std::shared_ptr<rnn_cell_base> backward_cell_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(forward_cell_m, backward_cell_m);
    }

    generic_bidirectional_rnn_model() = default;

    generic_bidirectional_rnn_model(const generic_bidirectional_rnn_model&) = default;

    generic_bidirectional_rnn_model(generic_bidirectional_rnn_model&&) noexcept = default;

    generic_bidirectional_rnn_model& operator=(const generic_bidirectional_rnn_model&) = default;

    generic_bidirectional_rnn_model& operator=(generic_bidirectional_rnn_model&&) noexcept = default;

    generic_bidirectional_rnn_model(std::shared_ptr<rnn_cell_base> forward_cell,
                                    std::shared_ptr<rnn_cell_base> backward_cell);


    /**
     * Apply the bidirectional RNN on a sequence of inputs
     * \param init_state the initial state for (1) forward RNN and (2) backward RNN.
     *                   Pass a pair of NULLs for default initial states.
     * \param xs the sequence of inputs
     * \return (1) the sequence of outputs
     *         (2) the final state of (a) forward RNN and (b) backward RNN
     */
    std::pair<std::vector<value_t>, std::pair<value_t, value_t>>
    transduce_impl(const std::pair<value_t, value_t>& init_state, const std::vector<value_t>& xs);

    value_t transduce(const value_t& init_state, const value_t& xs);


    std::string default_name() const;

  };

  /**
   * \brief Represents a bidirectional, multistack RNN.
   *
   * Can be of any RNN cell.
   */
  class generic_stacked_bidirectional_rnn_model {
    std::vector<generic_bidirectional_rnn_model> birnns_m;
    std::shared_ptr<dropout_model> dropout_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(birnns_m, dropout_m);
    }

    generic_stacked_bidirectional_rnn_model() = default;

    generic_stacked_bidirectional_rnn_model(const generic_stacked_bidirectional_rnn_model&) = default;

    generic_stacked_bidirectional_rnn_model(generic_stacked_bidirectional_rnn_model&&) noexcept = default;

    generic_stacked_bidirectional_rnn_model& operator=(const generic_stacked_bidirectional_rnn_model&) = default;

    generic_stacked_bidirectional_rnn_model& operator=(generic_stacked_bidirectional_rnn_model&&) noexcept = default;

    /**
     * Construct a stacked bidirectional RNN from a list of [forward_cell, backward_cell] pair
     * \param rnn_cell_pairs [forward_cell, backward_cell] pairs.
     *                       Every forward_cell and backward_cell should have the same input size
     *                       and output size equals to half the input size.
     * \param dropout_rate if > 0, applies dropout between stacks
     */
    explicit generic_stacked_bidirectional_rnn_model(
      const std::vector<std::pair<std::shared_ptr<rnn_cell_base>, std::shared_ptr<rnn_cell_base>>>& rnn_cell_pairs,
      float dropout_rate = 0);

    /**
    * Apply this stacked bidirectional RNN on a sequence of inputs
    * \param init_state the initial states for (1) forward RNN and (2) backward RNN.
    *                   From bottom to top stack.
    *                   Pass a list of pair of NULLs for default initial states.
    * \param xs the sequence of inputs
    * \return (1) the sequence of outputs
    *         (2) the final state of (a) forward RNN and (b) backward RNN, from bottom to top stack
    */
    std::pair<std::vector<value_t>, std::vector<std::pair<value_t, value_t>>>
    transduce_impl(const std::vector<std::pair<value_t, value_t>>& init_state, const std::vector<value_t>& xs);

    value_t transduce(const value_t& init_state, const value_t& xs);

    unsigned long num_stacks() const;

    std::string default_name() const;

  };
}

#endif
