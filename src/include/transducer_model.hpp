//
// Created by Dekai WU and YAN Yuchen on 20200423.
//

#ifndef LEGO_TRANSDUCER_MODEL_HPP
#define LEGO_TRANSDUCER_MODEL_HPP

#include "transducer_typed_value.hpp"
#include "generate_type_consistent_tuple.hpp"
#include "lego_primitive_types.hpp"
#include "value_placeholder.hpp"
#include <fstream>
#include <functional>

namespace tg {

  class transducer_instance;

  class transducer_dataset;

  class training_pipeline;

  class backprop_trainable_parameter_base;

  class transducer_variant;

  /**
   * \ingroup transducer
   * \brief A transducer with internal trainable parameters
   */
  class transducer_model {
    mutable std::shared_ptr<transducer_variant> impl;
    friend tg::training_pipeline;
  private:
    static void save_to_stream_impl(std::ostream& os, const std::vector<transducer_model>& models);
    static void load_from_stream_impl(std::istream& is, std::vector<transducer_model>& models);

    /**
     * \brief A constant value transducer model.
     *
     * A constant value transducer model takes no inputs and returns a constant value.
     *
     * \param v the constant value
     */
    explicit transducer_model(value_t v);

  public:
    template<typename Archive>
    void serialize(Archive& ar);

    transducer_model() = default;

    /**
     * \brief Create a shallow copy
     *
     * It doesn't give you a deep clone.
     */
    transducer_model(const transducer_model&) = default;

    transducer_model(transducer_model&&) noexcept = default;

    transducer_model& operator=(const transducer_model&);

    transducer_model& operator=(transducer_model&&) noexcept;

    explicit transducer_model(std::shared_ptr<transducer_variant> impl);


    /**
     * \brief Create a unary transducer by designing your custom transducer graph.
     *
     * \details You supply a unary function that describes the transducer graph.
     * Your supplied function should take a value placeholder
     * and returns a value placeholder by applying transducers on the input value placeholder.
     *
     * For example, here is how you can create a transducer that squares its input:
     * \code
     * transducer_model my_square([](const value_placeholder& x){
     *   return x * x;
     * });
     * \endcode
     *
     * \param fn Your unary function that describes the transducer network
     */
    explicit transducer_model(const std::function<value_placeholder(const value_placeholder&)>& fn);

    /**
     * \brief Create a binary transducer by designing your custom transducer graph.
     *
     * \details You supply a binary function that describes the transducer graph.
     * Your supplied function should take 2 value placeholders
     * and returns a value placeholder by applying transducers on the input value placeholders.
     *
     * For example, here is how you can create a transducer that computes the Geometric mean of two numbers:
     * \code
     * transducer_model my_geometric_mean([](const value_placeholder& x, const value_placeholder& y) {
     *   return sqrt(x * y);
     * });
     * \endcode
     *
     * \param fn Your binary function that describes the transducer network
     */
    explicit transducer_model(const std::function<value_placeholder(const value_placeholder&, const value_placeholder&)>& fn);

    /**
     * \brief Create a ternary transducer by designing your custom transducer graph.
     * \details Similar to how a unary and binary transducer can be created.
     * \param fn Your ternary function that describes the transducer network
     */
    explicit transducer_model(const std::function<value_placeholder(const value_placeholder&, const value_placeholder&, const value_placeholder&)>& fn);

    /**
     * \brief Create a quaternary transducer from a quaternary function by designing your custom transducer graph.
     * \details Similar to how a unary and binary transducer can be created.
     * \param fn Your quaternary function that describes the transducer network
     */
    explicit transducer_model(const std::function<value_placeholder(
      const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&)>& fn);

    /**
     * \brief Create a 5-ary transducer from a 5-ary function by designing your custom transducer graph.
     * \details Similar to how a unary and binary transducer can be created.
     * \param fn Your 5-ary function that describes the transducer network
     */
    explicit transducer_model(const std::function<value_placeholder(
      const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&)>& fn);

    /**
     * \brief Create a 6-ary transducer from a 6-ary function by designing your custom transducer graph.
     * \details Similar to how a unary and binary transducer can be created.
     * \param fn Your 6-ary function that describes the transducer network
     */
    explicit transducer_model(const std::function<value_placeholder(
      const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&)>& fn);

    /**
     * \brief Create a 7-ary transducer from a 7-ary function by designing your custom transducer graph.
     * \details Similar to how a unary and binary transducer can be created.
     * \param fn Your 7-ary function that describes the transducer network
     */
    explicit transducer_model(const std::function<value_placeholder(
      const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&)>& fn);

    /**
     * \brief Create a 8-ary transducer from a 8-ary function by designing your custom transducer graph.
     * \details Similar to how a unary and binary transducer can be created.
     * \param fn Your 8-ary function that describes the transducer network
     */
    explicit transducer_model(const std::function<value_placeholder(
      const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&)>& fn);

    /**
     * \brief Create a nullary transducer by designing your custom transducer graph.
     *
     * \param fn Your unary function that describes the transducer network
     */
    explicit transducer_model(const std::function<value_placeholder()>& fn);

    /**
     * \brief Checks whether this transducer model is valid.
     *
     * If you construct a transducer model using the default constructor,
     * then you get an invalid transducer model.
     *
     * \return true if this transducer model is valid.
     */
    [[nodiscard]] bool valid() const;

    /**
     * \brief Instantiate this transduccer model into a transducer instance.
     *
     * See tg::transducer_instance for more information.
     *
     * \return the transducer instance
     */
    [[nodiscard]] transducer_instance instantiate() const;

    /**
     * \brief Apply this transducer model to some value placeholders.
     *
     *
     * \tparam T = tg::value_placeholder
     * \param in0 The first value placeholder
     * \param ins The rest value placeholder
     * \return the value placeholder after applying this transducer
     */
    template<typename... T>
    value_placeholder operator()(const value_placeholder& in0, const T& ... rest) const;

    [[nodiscard]] value_placeholder apply(const std::vector<value_placeholder>& input_placeholders) const;

    /**
     * \brief Apply this transducer, transducing a given input
     *
     * You need to pass N arguments (where N=arity). Each Argument must be tg::value_t constructable.
     *
     * \tparam T = tg::typed_value
     * \param ins The input values
     * \return the output of this transducer model
     */
    template<typename... T>
    value_t operator()(const T& ... ins) const {
      return apply(std::vector<value_t>{value_t(ins)...});
    }

    /**
     * \brief The non-variadic version of transduce()
     *
     * \param datum A list of input values whose length must match the arity of this transducer.
     * \return The output value
     */
    [[nodiscard]] value_t apply(const std::vector<value_t>& datum) const;

    /**
     * \brief Transduce multiple datums
     *
     * This is generally faster than calling transduce() one by one,
     * because tensor arithmetic is faster when executed in batch.
     *
     * \param batch The dataset containing the datums to transduce
     * \return The transduction result for each datum
     */
    std::vector<value_t> batch_transduce(const std::shared_ptr<const transducer_dataset>& batch) const;

    /**
     * \brief Checks whether a transducer supports certain arity.
     *
     * Note that it is possible for a transducer to have multiple arities, like tg::make_list
     *
     * \return The number of inputs this transducer expects.
     */
    bool is_arity(unsigned long arity) const;

    /**
     * \brief Get the underlying transducer implementation.
     *
     * If the implementation is unavailable, will create a TBD transducer immediately.
     *
     * \internal
     * \return the underlying transducer implementation
     */
    [[nodiscard]] const std::shared_ptr<transducer_variant>& _get_impl() const;

    /**
     * \brief Save one or more transducer models into an output stream
     *
     * \tparam T = tg::transducer_model
     * \param os The output stream to save to
     * \param models One or more transducer models to save
     */
    template <typename ...T>
    static void save_to_stream(std::ostream& os, const T& ... models) {
      save_to_stream_impl(os, std::vector<transducer_model>{models...});
    }

    /**
     * \brief Load one of more transducer models from an input stream.
     *
     * \tparam T = tg::transducer_model
     * \param is The input stream to load from
     * \param models One or more transducer models to load into
     */
    template <typename ...T>
    static void load_from_stream(std::istream& is, T& ... models) {
      std::vector<transducer_model> ret;
      load_from_stream_impl(is, ret);
      std::vector<transducer_model*> out_refs{&models...};

      if(ret.size() != out_refs.size()) {
        std::stringstream ss;
        ss << "Error when loading models: Expect "<< out_refs.size() << " model" << (out_refs.size()>1?"s":"") << " to be loaded, but got " << ret.size() << " model" << (ret.size()>1?"s":"") << " in the input stream.";
        throw std::runtime_error(ss.str());
      }

      for(unsigned long i=0; i<out_refs.size(); ++i) {
        *out_refs[i] = std::move(ret[i]);
      }
    }

    /**
     * \brief Within this transducer, find a (possibly nested) transducer by name.
     *
     * Throws an exception if the transducer with given name is not found.
     *
     * When there are multiple matches, only the first match will be returned.
     *
     * \param name The name of the transducer to find.
     * \return The found transducer.
     */
    transducer_model find_transducer_by_name(const std::string& name);

    /**
     * \brief Save one of more transducer models into an output file.
     *
     * \tparam T = tg::transducer_model
     * \param path Path to the output file
     * \param models One or more transducer models to save
     */
    template <typename ...T>
    static void save_to_file(const std::string& path, T& ... models) {
      std::ofstream ofs(path, std::ios::binary);
      if(!ofs) throw_with_nested(std::runtime_error("Cannot write to file at: " + path));
      save_to_stream(ofs, models...);
    }

    /**
     * \brief Load one of more transducer models from an input file.
     *
     * \tparam T = tg::transducer_model
     * \param path Path to the input file
     * \param models One or more transducer models to load into
     */
    template <typename ...T>
    static void load_from_file(const std::string& path, T&... models) {
      std::ifstream ifs(path, std::ios::binary);
      if(!ifs) throw_with_nested(std::runtime_error("Cannot read from file at: " + path));
      load_from_stream(ifs, models...);
    }

    /**
     * \brief Get the display name of this transducer model, mainly for debugging purposes.
     * \return the display name of this transducer model.
     */
    [[nodiscard]] std::string name() const;

    /**
     * \brief Change the display name of this transducer inplace
     *
     * The name does not affect the model's transduction behavior, but
     * giving your models meaningful names can help you debugging.
     *
     * \param name The new display name.
     * \return this transducer
     */
    transducer_model& rename(const std::string& name);
  };

  transducer_model compose_impl(std::initializer_list<std::shared_ptr<transducer_variant>> pieces);

  /**
   * \brief Compose multiple transducers
   *
   * for function x -> F(x)
   * and function x -> G(x)
   * and function x -> H(x)
   * the function composition: F ○ G ○ H
   * will result in the function: x -> F(G(H(x)))
   *
   * \param args The list of transducers to compose
   * \return the composed transducer
   */
  template<typename ...transducer_model_t>
  transducer_model compose(const transducer_model_t& ...args) {
    return compose_impl({args._get_impl()...});
  }

  /**
   * \brief Represents a dynamic transducer applied on a specific input
   *
   * Even though a static transducer model already provides extensive flexibility,
   * there are cases where you would like to construct the transducer model dynamically for each input.
   *
   * A dynamic transducer application is a function that returns a nullary transducer model. This transducer model, when transduced, returns the application result.
   *
   */
  using dynamic_transducer_application = std::function<value_placeholder()>;
}


#endif //LEGO_TRANSDUCER_MODEL_HPP

