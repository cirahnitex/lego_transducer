//
// Created by Dekai WU and YAN Yuchen on 20200526.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_RNN_CELLS_HPP
#define LEGO_RNN_CELLS_HPP
#include "backprop_trainable_parameter.hpp"
#include "dense_model.hpp"
namespace tg {
  class rnn_cell_base {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }
    /**
     * Apply this RNN cell for one timestep
     * \param prev_state state from previous timestep. Passing NULL for default state.
     * \param x the current input
     * \return a pair that contains (1) the current output and (2) state for next timestep.
     */
    value_t transduce(const value_t& prev_state, const value_t& x);

    /**
     * \brief Convert a null value into default cell state
     *
     * If the supplied value is not null, this function will echo back the input
     *
     * \param state The value to convert to cell state
     * \return The converted cell state
     */
    value_t null_state_to_default_state(const value_t& state);

    /**
     * Apply this RNN cell for one timestep
     * \param prev_state state from previous timestep.
     *                   IMPORTANT! Passing NULL will give an error.
     * \param x the current input
     * \return (1) the current output
     *         (2) state for next timestep
     */
    virtual std::pair<value_t, value_t> transduce_impl(const value_t& prev_state, const value_t& x) = 0;

    /**
     * Get the default initial state
     * \return the default initial state
     */
    virtual value_t default_initial_state() const = 0;

  };

  class naive_rnn_cell :public rnn_cell_base {
    n_ary_dense_model dense_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(cereal::base_class<rnn_cell_base>(this), dense_m);
    }
    naive_rnn_cell() = default;
    naive_rnn_cell(const naive_rnn_cell&) = default;
    naive_rnn_cell(naive_rnn_cell&&) noexcept = default;
    naive_rnn_cell& operator=(const naive_rnn_cell&) = default;
    naive_rnn_cell& operator=(naive_rnn_cell&&) noexcept = default;
    naive_rnn_cell(unsigned long input_size, unsigned long output_size);

    std::pair<value_t, value_t> transduce_impl(const value_t& prev_state, const value_t& x) override;

    value_t default_initial_state() const override;

    std::string default_name() const;

  };

  class vanilla_lstm_cell : public rnn_cell_base {
    unsigned long input_size_m{};
    unsigned long output_size_m{};
    dense_model forget_gate, input_gate, output_gate, input_layer;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(cereal::base_class<rnn_cell_base>(this), input_size_m, output_size_m, forget_gate, input_gate, output_gate, input_layer);
    }
    vanilla_lstm_cell() = default;
    vanilla_lstm_cell(const vanilla_lstm_cell&) = default;
    vanilla_lstm_cell(vanilla_lstm_cell&&) noexcept = default;
    vanilla_lstm_cell& operator=(const vanilla_lstm_cell&) = default;
    vanilla_lstm_cell& operator=(vanilla_lstm_cell&&) noexcept = default;
    vanilla_lstm_cell(unsigned long input_size, unsigned long output_size);

    std::pair<value_t, value_t> transduce_impl(const value_t& prev_state, const value_t& x) override;

    value_t default_initial_state() const override;

    std::string default_name() const;
  };

  class coupled_lstm_cell :public rnn_cell_base {
    unsigned long input_size_m{};
    unsigned long output_size_m{};
    dense_model forget_gate, output_gate, input_layer;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(cereal::base_class<rnn_cell_base>(this), input_size_m, output_size_m, forget_gate, output_gate, input_layer);
    }
    coupled_lstm_cell() = default;
    coupled_lstm_cell(const coupled_lstm_cell&) = default;
    coupled_lstm_cell(coupled_lstm_cell&&) noexcept = default;
    coupled_lstm_cell& operator=(const coupled_lstm_cell&) = default;
    coupled_lstm_cell& operator=(coupled_lstm_cell&&) noexcept = default;
    coupled_lstm_cell(unsigned long input_size, unsigned long output_size);

    std::pair<value_t, value_t> transduce_impl(const value_t& prev_state, const value_t& x) override;

    value_t default_initial_state() const override;

    std::string default_name() const;
  };

  class gru_cell :public rnn_cell_base {
    unsigned long input_size_m{};
    unsigned long output_size_m{};
    dense_model pre_input_gate_m, input_fc_m, output_gate_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(cereal::base_class<rnn_cell_base>(this), input_size_m, output_size_m, pre_input_gate_m, input_fc_m, output_gate_m);
    }
    gru_cell() = default;
    gru_cell(const gru_cell&) = default;
    gru_cell(gru_cell&&) noexcept = default;
    gru_cell& operator=(const gru_cell&) = default;
    gru_cell& operator=(gru_cell&&) noexcept = default;
    gru_cell(unsigned long input_size, unsigned long output_size);

    std::pair<value_t, value_t> transduce_impl(const value_t& prev_state, const value_t& x) override;

    value_t default_initial_state() const override;

    std::string default_name() const;
  };
}

CEREAL_REGISTER_TYPE(tg::naive_rnn_cell)
CEREAL_REGISTER_TYPE(tg::vanilla_lstm_cell)
CEREAL_REGISTER_TYPE(tg::coupled_lstm_cell)
CEREAL_REGISTER_TYPE(tg::gru_cell)

#endif //LEGO_RNN_CELLS_HPP
