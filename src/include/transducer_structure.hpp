//
// Created by Dekai WU and YAN Yuchen on 20200428.
//

#ifndef LEGO_TRANSDUCER_STRUCTURE_HPP
#define LEGO_TRANSDUCER_STRUCTURE_HPP
#include "transducer_model.hpp"
#include "lego_primitive_types.hpp"

namespace tg {
  /**
   * \addtogroup transducer
   * @{
   */

  /**
   * \brief Basic dense layer.
   * \details A dense layer performs
   * \f$ \mathbf{W}\mathbf{x}+\mathbf{b} \f$
   * where \f$ \mathbf{W} \f$ and \f$ \mathbf{b} \f$ are parameters
   */
  class dense_structure_t{
  public:

    /**
     * \brief Create a dense layer
     *
     * It returns a transducer model with the same signature as model_signature()
     *
     * \param input_dimension Number of input units
     * \param output_dimension Number of output units
     * \param use_bias whether the dense layer have the bias parameter.
     * \return The created dense layer transducer model
     */
    transducer_model initialize(unsigned long input_dimension, unsigned long output_dimension, bool use_bias = true);

    /**
     * \brief Apply this dense layer
     * \details This function has the same signature as the model returned by initialize()
     * \param x The input tensor, of shape either {input_dimension}
     * \return The output tensor of shape {output_dimension}.
     */
    transducer_model model_signature(const transducer_model& x) const;
  };
  extern dense_structure_t dense_structure;

  /**
   * \brief Multi-input dense layer
   *
   * Similar to a dense layer except that it takes multiple inputs. It computes
   *
   * \f$
   * \mathbf{W}\begin{bmatrix}\mathbf{x_0} \\ \mathbf{x_1}\\ \vdots \\ \mathbf{x_{N-1}}} \end{bmatrix}+\mathbf{b}
   * \f$
   *
   * Where N is the number of inputs
   */
  class n_ary_dense_structure_t {
  public:
    /**
     * \brief Create a multi-input dense layer
     *
     * It returns a transducer model with the same signature as model_signature()
     *
     * \param input_dimensions The dimensions for x_0, x_1, ..., x_{n-1}
     * \param output_dimension The dimensions for output tensor
     * \param use_bias Whether to use the bias \f$ \mathbf{b} \f$ or not
     * \return The multi-input dense layer transducer model
     */
    transducer_model initialize(std::vector<unsigned long> input_dimensions, unsigned long output_dimension, bool use_bias = true);

    /**
     * \brief Apply this dense layer
     *
     * This function has the same signature as the model returned by initialize()
     *
     * \tparam T = tg::transducer_model
     * \param xs N input tensors (as variadic argument) of shape {input_dimension}
     * \return The output tensor of shape {output_dimension}.
     */
    template<typename ...T>
    transducer_model model_signature(const T& ... xs) const { return transducer_model(); }
  };
  extern n_ary_dense_structure_t n_ary_dense_structure;

  /**
   * \brief Embedding lookup table.
   *
   * An embedding lookup table stores a mapping table
   *   * from an ID, ranging from 0 to (capacity - 1)
   *   * to a rank-1 tensor, whose dimension is often called <b>embedding size</b>
   */
  class embedding_table_structure_t {
  public:

    /**
     * \brief Initialize an embedding table with random values
     *
     * \param embedding_size The embedding size
     * \param capacity The number of entries of this embedding table
     * \return The created embedding table transducer model
     */
    transducer_model initialize(unsigned long embedding_size, unsigned long capacity);

    /**
     * \brief Initialize an embedding table with pre-trained values.
     *
     * You need to supply a list of pre-trained value table in std::vector<std::vector<float>>
     * The size of your list is the capacity of this embedding table. Each item in your list can be
     *   * either of length <b>embedding size</b>, in which case the corresponding entry will initialize to this values
     *   * or of length zero, in which case the corresponding entry will fallback to random initialization
     *
     * It returns a transducer model with the same signature as model_signature()
     *
     * \param embedding_size The embedding size
     * \param pretrained_embeddings The list of pre-trained embeddings.
     * \return The created embedding table model
     */
    transducer_model initialize(unsigned long embedding_size, const std::vector<std::vector<float>>& pretrained_embeddings);

    /**
     * \brief Lookup the embedding of an ID
     *
     * This function has the same signature as the model returned by initialize()
     *
     * \param id the ID to lookup
     * \return The embedding
     */
    transducer_model model_signature(const transducer_model& id);
  };
  extern embedding_table_structure_t embedding_table_structure;

  /**
   * \brief The final classification operation
   *
   * It takes logits (which is a tensor of shape {num_labels}) and returns the predicted label ID.
   *
   */
  class classify_from_logits_structure_t {
  public:

    /**
     * \brief Initialize a classification operation
     *
     * It returns two transducers: performance component and loss function
     *   * The performance component can be used to build classifier.
     *     It have the same signature as performance_component_signature()
     *   * The loss function can be used to compute the loss of a classification.
     *     It have the same signature as loss_function_signature()
     *
     * \param num_labels number of labels to classify from
     * \return (1) The created readout transducer model and (2) The transducer model that computes readout loss
     */
    std::pair<transducer_model, transducer_model> initialize(unsigned long num_labels);

    /**
     * \brief Get the classification result
     * \param logits A tensor of shape {num_labels}
     * \return The index of the maximum value of the logits
     */
    transducer_model performance_component_signature(const transducer_model& logits);

    /**
     * \brief Compute the readout loss
     *
     * \param logits A tensor of shape {num_labels}
     * \param oracle_id The ID of the oracle label
     * \return the cross-entropy loss, which is defined as -log(softmax(logits))[oracle_id]
     */
    transducer_model loss_function_signature(const transducer_model& logits, const transducer_model& oracle_id);
  };
  extern classify_from_logits_structure_t classify_from_logits_structure;

  /**
   * \brief A dictionary that converts between symbol and ID.
   */
  class dict_structure_t {
  public:
    /**
     * \brief Initialize a dictionary from a vocabulary
     *
     * The returned transducer model have the same signature as model_signature()
     *
     * \param vocab The list of tokens (should not contain duplicates).
     *              The index of each token in the list will be its ID.
     * \return The created dictionary transducer
     */
    transducer_model initialize(const std::vector<std::string>& vocab);

    /**
     * \brief Convert between symbol and ID
     *
     * <b>Notes on out-of-vocabulary symbols</b>
     *
     * When converting from symbol to ID, all of-of-vocabulary symbols are assigned with ID equal to the vocab-size
     *
     * When converting from ID to symbol, all IDs >= vocab-size will return the <unk> symbol (including the angle brackets)
     *
     * \param x A symbol or an ID
     * \return The corresponding ID or symbol
     */
    transducer_model model_signature(const transducer_model& x);
  };
  extern dict_structure_t dict_structure;

  /**
   * \brief Similar to tg::embedding_lookup_table, except that it looks up from a symbol instead of an ID.
   *
   * Equivalent to composing an embedding lookup table with a dictionary.
   */
  class symbolic_embedding_table_structure_t {
  public:

    /**
     * \brief Initialize an embedding table with random values
     *
     * The created transducer model have the same signature as model_signature()
     *
     * \param embedding_size The embedding size
     * \param vocab The vocabulary (should not contain duplicates)
     * \return The created embedding table transducer model
     */
    transducer_model initialize(unsigned long embedding_size, const std::vector<symbol_t>& vocab);

    /**
     * \brief The generic version of initialize(unsigned long embedding_size, const std::vector<symbol_t>& vocab)
     *
     * \param embedding_size The embedding size
     * \param vocab The vocabulary
     * \return The created embedding table transducer model
     */
    template<typename T>
    transducer_model initialize(unsigned long embedding_size, const T& vocab) {
      return initialize(embedding_size, std::vector<symbol_t>(vocab.begin(), vocab.end()));
    }

    /**
     * \brief Initialize an embedding table with pre-trained embeddings.
     *
     * You need to supply a token-to-pretrained-value hashmap in std::unordered_map<symbol_t, td::vector<float>>
     * The keys of your hashmap is the vocabulary the constructed embedding table. Each value in your hashmap can be
     *   * either of length <b>embedding size</b>, in which case the corresponding entry will initialize to this values
     *   * or of length zero, in which case the corresponding entry will fallback to random initialization
     *
     * The created transducer model have the same signature as model_signature()
     *
     * \param embedding_size The embedding size
     * \param pretrained_embeddings Your token-to-pretrained-value hashmap
     * \return The created embedding table transducer model
     */
    transducer_model initialize(unsigned long embedding_size, const std::unordered_map<symbol_t,
      std::vector<float>>& pretrained_embeddings);

    /**
     * \brief Lookup the embedding of a symbol
     *
     * If the symbol is out of vocabulary, the special embedding value for out-of-vocabulary token is returned.
     * This embedding value will also be trained alongside other embedding values.
     *
     * \param symbol The symbol to lookup
     * \return The embedding value
     */
    transducer_model model_signature(const transducer_model& symbol);
  };
  extern symbolic_embedding_table_structure_t symbolic_embedding_table_structure;

  /**
   * \brief Similar to tg::classify_from_logits_structure_t, expect that it outputs the label as a symbol instead of an ID.
   *
   * Equivalent to composing a dictionary with tg::classify_from_logits_structure_t
   */
  class symbolic_classify_from_logits_structure_t {
  public:

    /**
     * \brief Construct a classification operation from a set of labels
     *
     * It returns two transducers: performance component and loss function
     *   * The performance component can be used to build classifier.
     *     It have the same signature as performance_component_signature()
     *   * The loss function can be used to compute the loss of a classification.
     *     It have the same signature as loss_function_signature()
     *
     * \param labels The list of labels to classify (should not contain duplicates)
     * \return The created readout layer
     */
    std::pair<transducer_model, transducer_model> initialize(const std::vector<symbol_t>& labels);

    /**
     * \brief The generic version of initialize(const std::vector<symbol_t>& labels)
     *
     * \param labels The set of labels to classify
     * \return The created readout layer
     */
    template<typename T>
    std::pair<transducer_model, transducer_model> initialize(const T& labels) {
      return initialize(std::vector<symbol_t>(labels.begin(), labels.end()));
    }

    /**
     * \brief Get the classification result
     *
     * \param logits A tensor of shape {num_labels}
     * \return The output label (of type symbol)
     */
    transducer_model performance_component_signature(const transducer_model& logits);

    /**
     * \brief Compute the readout loss
     *
     * \param logits A tensor of shape {num_labels}
     * \param oracle The oracle label (of type symbol)
     * \return the cross-entropy loss
     */
    transducer_model loss_function_signature(const transducer_model& logits, const transducer_model& oracle);
  };
  extern symbolic_classify_from_logits_structure_t symbolic_classify_from_logits_structure;

  /**
   * \brief A bilinear layer.
   *
   * A bilinear layer computes
   *
   * \f$
   * f(\mathbf{x_0},\mathbf{x_1})=\mathbf{x_0^T}\mathbf{M}\mathbf{x_1}+\mathbf{b}
   * \f$
   *
   */
  class bilinear_structure_t {
  public:
    /**
     * \brief Initialize a bilinear layer
     *
     * The returned transducer model have the same signature as model_signature()
     *
     * \param input_0_size The dimension of \f$ \mathbf{x_0} \f$
     * \param input_1_size The dimension of \f$ \mathbf{x_1} \f$
     * \param output_size The dimension of \f$ f(\mathbf{x_0},\mathbf{x_1}) \f$
     * \param with_bias Whether to have bias term or not
     * \return The created bilinear transducer model
     */
    transducer_model initialize(unsigned long input_0_size, unsigned long input_1_size, unsigned long output_size, bool with_bias = true);

    /**
     * \brief Apply this bilinear layer
     * \param x0 first input tensor, of shape {input_0_size}
     * \param x1 second input tensor, of shape {input_1_size}
     * \return The computed result, of shape {output_size}
     */
    transducer_model model_signature(const transducer_model& x0, const transducer_model& x1);
  };
  extern bilinear_structure_t bilinear_structure;

  /**
   * \brief A biaffine layer.
   *
   * Biaffine layer contains both bilinear term, linear term and bias term. It's the combination of a bilinear layer and a dense layer.
   *
   * A biaffine layer computes
   * \f$
   * f(\mathbf{x_0},\mathbf{x_1})=\mathbf{x_0^T}\mathbf{M}\mathbf{x_1}+\mathbf{W_0}\mathbf{x_0}+\mathbf{W_1}\mathbf{x_1}+\mathbf{b}
   * \f$
   */
  class biaffine_structure_t {
  public:
    /**
     * \brief Initialize a biaffine layer.
     *
     * The return transducer model has the same signature as model_signature()
     *
     * \param input_0_size The dimension of \f$ \mathbf{x_0} \f$
     * \param input_1_size The dimension of \f$ \mathbf{x_1} \f$
     * \param output_size The dimension of \f$ f(\mathbf{x_0},\mathbf{x_1}) \f$
     * \param with_bias Whether to have bias term or not
     * \return The created biaffine transducer model
     */
    transducer_model initialize(unsigned long input_0_size, unsigned long input_1_size, unsigned long output_size, bool with_bias = true);

    /**
     * \brief Apply this biaffine layer.
     *
     * \param x0 first input tensor, of shape {input_0_size}
     * \param x1 second input tensor, of shape {input_1_size}
     * \return The computed result, of shape {output_size}
     */
    transducer_model model_signature(const transducer_model& x0, const transducer_model& x1);
  };

  extern biaffine_structure_t biaffine_structure;

  enum RNN_CELL_TYPE {
    NAIVE_RNN, VANILLA_LSTM, COUPLED_LSTM, GRU
  };

  /**
   * \brief A unidirectional RNN.
   */
  class rnn_structure_t {
  public:
    /**
     * \brief Initialize the unidirectional RNN
     *
     * The return transducer model have the same signature as model_signature()
     *
     * \param input_size the size of input tensor
     * \param output_size the size of output tensor
     * \param cell_type the type of RNN cell. See enum tg::RNN_CELL_TYPE for a list of supported RNN cell types
     * \param num_stacks the number of stacks if you want a multi-stack RNN
     * \param dropout_rate the dropout rate to apply in between stacks (only applies to multi-stack RNN)
     * \return the RNN model
     */
    transducer_model initialize(unsigned long input_size, unsigned long output_size, RNN_CELL_TYPE cell_type, unsigned long num_stacks=1, float dropout_rate=0);

    /**
     * \brief Apply this RNN on a list of inputs (in timestep order)
     *
     * It returns two values (wrapped in a list):
     *   1. the outputs (as a list of rank-1 tensors. list item #N is the output at timestep #N)
     *   2. the state for next timestep. Use this if you want this state to chain into other RNNs of the same structure.
     *
     * \param prev_state The state from previous timestep. Normally you would simply give NULL for default state.
     * \param ins The inputs (as a list of rank-1 tensors. list item #N is the input at timestep #N)
     * \return The outputs (in timestep order) and the final state.
     */
    transducer_model model_signature(const transducer_model& prev_state, const transducer_model& ins);
  };
  extern rnn_structure_t rnn_structure;


  /**
   * \brief A bidirectional RNN.
   *
   * The output at a timestep is a concatenation of the forward RNN output and the backward RNN output.
   */
  class bidirectional_rnn_structure_t {
  public:
    /**
     * \brief Initialize the bidirectional RNN
     *
     * The return transducer model have the same signature as model_signature()
     *
     * \param input_size the size of input tensor
     * \param output_size the size of output tensor. Must be a multiple of 2,
     *                    because the forward and backward RNN each contributes to half of it.
     * \param cell_type the type of RNN cell. See enum `RNN_CELL_TYPE` for a list of supported RNN cell types
     * \param num_stacks the number of stacks if you want a multi-stack RNN
     * \param dropout_rate the dropout rate to apply in between stacks (only applies to multi-stack RNN)
     * \return the bidirectional RNN model
     */
    transducer_model initialize(unsigned long input_size, unsigned long output_size, RNN_CELL_TYPE cell_type, unsigned long num_stacks=1, float dropout_rate=0);

    /**
     * \brief Apply this bidirectional RNN on a list of inputs (in timestep order)
     *
     * It returns two values (wrapped in a list):
     *   1. the outputs (as a list of rank-1 tensors. list item #N is the output at timestep #N)
     *   2. the final state (combining the final state from both forward and backward directions).
     *      Use this if you want this state to chain into other bidirectional RNNs of the same structure.
     *
     * \param prev_state The state from previous timestep. Normally you would simply give NULL for default state.
     * \param ins The inputs (as a list of rank-1 tensors. list item #N is the input at timestep #N)
     * \return The outputs (in timestep order) and the final state.
     */
    transducer_model model_signature(const transducer_model& prev_state, const transducer_model& ins);
  };
  extern bidirectional_rnn_structure_t bidirectional_rnn_structure;


  /**
   * \brief Dropout layer
   *
   * A dropout layer randomly drops out (set to zero) values from the input tensor with probability p,
   * and scales the un-dropped values by a factor of 1/p
   *
   * This transducer have no effect when not training.
   * However, you can force this dropout to apply by turning on the tg::lego_training_guard,
   * if you want to debug this dropout layer.
   */
  class dropout_structure_t {
  public:

    /**
     * \brief Initialize a dropout layer
     *
     * The return transducer model have the same signature as model_signature()
     *
     * \param dropout_rate The dropout rate p
     * \return The created dropout layer transducer model
     */
    transducer_model initialize(float dropout_rate);

    /**
     * \brief Apply this dropout layer
     *
     * This operation has no effect while not training.
     *
     * \param x The tensor to dropout
     * \return The tensor after dropout
     */
    transducer_model model_signature(const transducer_model& x);
  };
  extern dropout_structure_t dropout_structure;

  /**
   * \brief Similar to dropout except the dropout mask is the same across one or more axes.
   *
   * Use this if you want to drop columns within a matrix, or drop matrices within a higher-rank tensor.
   *
   * This transducer have no effect when not training.
   * However, you can force this dropout to apply by turning on the tg::lego_training_guard,
   * if you want to debug this dropout layer.
   *
   */
  class axis_synchronized_dropout_structure_t {
  public:
    /**
     * \brief Initialize this dropout layer
     *
     * The return transducer model have the same signature as model_signature()
     *
     * \param dropout_rate The dropout rate p
     * \param synchronized_axes the set of axes to share dropout masks
     * \return The created dropout layer transducer model
     */
    transducer_model initialize(float dropout_rate, std::unordered_set<unsigned long> synchronized_axes);

    /**
     * \brief Apply this dropout layer
     *
     * This operation has no effect while not training.
     *
     * \param x The tensor to dropout
     * \return The tensor after dropout
     */
    transducer_model model_signature(const transducer_model& x);
  };
  extern axis_synchronized_dropout_structure_t axis_synchronized_dropout_structure;

  /// @}
}
#endif //LEGO_TRANSDUCER_STRUCTURE_HPP
