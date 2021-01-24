//
// Created by Dekai WU and YAN Yuchen on 20200503.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_BACKPROP_TRAINABLE_PARAMETER_HPP
#define LEGO_BACKPROP_TRAINABLE_PARAMETER_HPP
#include "include/transducer_typed_value.hpp"
#include "include/lego_serialization_helper.hpp"
#include "include/lego_param_naming_guard.hpp"
namespace tg {
  class optimizer_base;
  /**
   * A dynet::Parameter must lives in a dynet::ParameterCollection.
   * However, this design conflicts with the human intuition that
   * a model should hold it's own parameters
   * instead of relying on some external parameter collection management system.
   *
   * In our library, a model owns its parameters.
   * A backprop trainer can ask a model to temporarily move its parameters into the trainer's parameter collection,
   * so that it can be trained.
   * When the backprop training is complete, the model reclaims back the ownership of its parameters.
   *
   * Implementational-wise, the dynet::ParameterCollection it lives in can be one of:
   * (1) a unique internal parameter collection owned by the parameter itself.
   * (2) an external parameter collection (from backprop trainer) when managed by a backprop trainer.
   *
   * This class provides interfaces to switch between the underlying implementations.
   */
  class backprop_trainable_parameter_base {
  public:
    /**
     * \brief A parameter can optionally have a path, that can be used to identify itself.
     *
     * Sometimes it is useful to selectively not train certain parameters.
     * For example, if you don't want to train the word embeddings.
     *
     * In this case, you can use the path of this parameter to identify which parameters you don't want to train.
     */
    lego_param_path path;


    template<typename Archive>
    void save(Archive& ar) const {
      ar(path);
    }

    template<typename Archive>
    void load(Archive& ar) {
      ar(path);
    }

    [[nodiscard]] virtual bool is_using_internal_pc() const = 0;
    [[nodiscard]] virtual bool is_using_external_pc() const = 0;
    virtual void use_internal_pc() = 0;
    virtual void use_external_pc(dynet::ParameterCollection& pc) = 0;

    /**
     * \brief Keeps a record of all parameters, so that a backprop trainer knows what to train.
     *
     */
    static thread_local std::unordered_set<backprop_trainable_parameter_base*> all_parameters;

    /**
     * \brief Register this into the global parameters container.
     */
    backprop_trainable_parameter_base();
    backprop_trainable_parameter_base(const backprop_trainable_parameter_base&);
    backprop_trainable_parameter_base(backprop_trainable_parameter_base&&) noexcept;
    backprop_trainable_parameter_base& operator=(const backprop_trainable_parameter_base&) = default;
    backprop_trainable_parameter_base& operator=(backprop_trainable_parameter_base&&) noexcept = default;

    /**
     * \brief Unregister this from the global parameters container.
     */
    virtual ~backprop_trainable_parameter_base();
  };

  /**
   * Represents a basic backprop trainable parameter.
   */
  class backprop_trainable_parameter :public backprop_trainable_parameter_base  {
    // Its own unique PC.
    // When managed by a backprop trainer, this value is nullptr.
    std::unique_ptr<dynet::ParameterCollection> internal_pc_m;

    // this is mutable because copying the values of this parameter (which should be a const operation)
    // requires an intrusive pointer to this dynet::Parameter (which is not const operation)
    mutable dynet::Parameter v;


    // Putting a parameter into dynet::ComputationGraph can be memory-consuming
    // So we cache the symbolic tensor so that the next time this parameter would be converted into symbolic tensor,
    // instead of putting this parameter into dynet::ComputationGraph again,
    // we instead return the cached symbolic tensor.
    static thread_local std::unordered_map<const backprop_trainable_parameter*, dynet::Expression> symbolic_tensor_cache;
  public:

    template<typename Archive>
    void save(Archive& ar) const {
      ar(cereal::base_class<backprop_trainable_parameter_base>(this));
      bool valid = operator bool();
      ar(valid);
      if(!valid) return;
      auto&& values = v.values();
      ar(values->d);
      ar(dynet::as_vector(*values));
    }

    template<typename Archive>
    void load(Archive& ar) {
      ar(cereal::base_class<backprop_trainable_parameter_base>(this));

      bool valid = true;
      ar(valid);
      if(!valid) return;
      internal_pc_m = std::make_unique<dynet::ParameterCollection>();

      dynet::Dim dim;
      ar(dim);
      v = internal_pc_m->add_parameters(dim);

      std::vector<float> values;
      ar(values);
      v.set_value(values);
    }

    backprop_trainable_parameter() = default;
    backprop_trainable_parameter(const backprop_trainable_parameter& x);
    backprop_trainable_parameter(backprop_trainable_parameter&&) noexcept = default;
    backprop_trainable_parameter& operator=(const backprop_trainable_parameter& x);
    backprop_trainable_parameter& operator=(backprop_trainable_parameter&&) noexcept = default;

    explicit backprop_trainable_parameter(const tensor_shape_t& dim);

    explicit operator bool() const;

    bool is_using_internal_pc() const override ;
    bool is_using_external_pc() const override ;
    void use_internal_pc() override ;
    void use_external_pc(dynet::ParameterCollection& pc) override ;

    /**
     * Get a symbolic tensor from this parameter.
     * \return the symbolic tensor
     */
    dynet::Expression as_symbolic_tensor() const;

    /**
     * Copies the underlying parameter values into a numeric tensor.
     * \return the numeric tensor
     */
    tensor_t as_tensor() const;

    virtual ~backprop_trainable_parameter();
  };

  /**
   * \brief Same as backprop trainable parameter, but is considered as bias parameter thus does not participate in weight decay.
   */
  class backprop_trainable_bias_parameter :public backprop_trainable_parameter {
  public:
    using backprop_trainable_parameter::backprop_trainable_parameter;
    template<typename Archive>
    void save(Archive& ar) const {
      ar(cereal::base_class<backprop_trainable_parameter>(this));
    }

    template<typename Archive>
    void load(Archive& ar) {
      ar(cereal::base_class<backprop_trainable_parameter>(this));
    }
  };

  /**
   * Represents a backprop trainable lookup parameter (optimized for embedding table usage).
   * Conceptually, it's a list of parameters.
   */
  class backprop_trainable_lookup_parameter :public backprop_trainable_parameter_base {
    std::unique_ptr<dynet::ParameterCollection> internal_pc_m;
    mutable dynet::LookupParameter v;
    static thread_local std::unordered_map<const backprop_trainable_lookup_parameter*, std::vector<dynet::Expression>> symbolic_tensors_cache;
  public:
    template<typename Archive>
    void save(Archive& ar) const {
      ar(cereal::base_class<backprop_trainable_parameter_base>(this));

      bool valid = operator bool();
      ar(valid);
      if(!valid) return;
      unsigned long n = num_entries();
      tensor_shape_t dim = embedding_dim();
      ar(n, dim);
      for(unsigned long i=0; i<n; ++i) {
        ar(dynet::as_vector(v.values()->at(i)));
      }
    }

    template<typename Archive>
    void load(Archive& ar) {
      ar(cereal::base_class<backprop_trainable_parameter_base>(this));

      bool valid = true;
      ar(valid);
      if(!valid) return;

      unsigned long n{};
      tensor_shape_t dim;
      ar(n, dim);
      internal_pc_m = std::make_unique<dynet::ParameterCollection>();
      v = internal_pc_m->add_lookup_parameters(n, to_dynet_dim(dim));
      for(unsigned long i=0; i<n; ++i) {
        std::vector<float> val;
        ar(val);
        v.initialize(i, val);
      }
    }


    backprop_trainable_lookup_parameter() = default;
    backprop_trainable_lookup_parameter(const backprop_trainable_lookup_parameter& x);
    backprop_trainable_lookup_parameter(backprop_trainable_lookup_parameter&&) noexcept = default;
    backprop_trainable_lookup_parameter& operator=(const backprop_trainable_lookup_parameter& x);
    backprop_trainable_lookup_parameter& operator=(backprop_trainable_lookup_parameter&&) noexcept = default;

    backprop_trainable_lookup_parameter(unsigned num_entries, const tensor_shape_t& embedding_dim);

    explicit operator bool() const;

    bool is_using_internal_pc() const override;
    bool is_using_external_pc() const override;
    void use_internal_pc() override;
    void use_external_pc(dynet::ParameterCollection& pc) override;

    unsigned long num_entries() const;

    tensor_shape_t embedding_dim() const;

    /**
     * Lookup the embedding at given index.
     * \param index the index of the embedding table entry
     * \return the embedding as a symbolic tensor
     */
    dynet::Expression lookup_as_symbolic_tensor(unsigned long index) const;

    /**
     * Lookup the embedding at given index.
     * \param index the index of the embedding table entry
     * \return a copy of the embedding as a tensor
     */
    tensor_t lookup_as_tensor(unsigned long index) const;

    /**
     * Set the value at give index
     * \param index the index of the embedding table entry
     * \param value the embedding value
     */
    void set_value(unsigned long index, const std::vector<float>& value);

    virtual ~backprop_trainable_lookup_parameter();
  };
}

CEREAL_REGISTER_TYPE(tg::backprop_trainable_parameter)
CEREAL_REGISTER_TYPE(tg::backprop_trainable_bias_parameter)
CEREAL_REGISTER_TYPE(tg::backprop_trainable_lookup_parameter)
#endif //LEGO_BACKPROP_TRAINABLE_PARAMETER_HPP
