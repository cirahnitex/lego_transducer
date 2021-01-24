//
// Created by Dekai WU and YAN Yuchen on 20200421.
//

#ifndef LEGO_TRANSDUCER_TYPED_VALUE_HPP
#define LEGO_TRANSDUCER_TYPED_VALUE_HPP

#include <variant>
#include "lego_tensor.hpp"
#include "generate_type_consistent_tuple.hpp"
#include "lego_primitive_types.hpp"
#include <memory>
#include <list>

namespace dynet {
  template<typename Archive>
  void serialize(Archive& ar, Expression& x) {
    throw std::runtime_error("Cannot serialize a symbolic tensor");
  }
}

namespace tg {
  class value_t;

  class dynet_computation_graph;

  class transducer_graph_node;

  class lambda_transducer_model;

  /**
   * \defgroup data_values
   * @{
   */

  using list_t = std::vector<value_t>;

  /**
   * \brief Write a tg::typed_value in a human-readible form
   * \param os The output stream to write to
   * \param x The value to write
   * \return The same output stream
   */
  std::ostream& operator<<(std::ostream& os, const tg::value_t& x);

  /**
   * \brief This exception indicates that NaN or Inf value is encountered.
   */
  class nan_or_inf_exception :public std::runtime_error {
  public:
    nan_or_inf_exception(const nan_or_inf_exception&) = default;
    nan_or_inf_exception(nan_or_inf_exception&&) noexcept = default;
    nan_or_inf_exception& operator=(const nan_or_inf_exception&) = default;
    nan_or_inf_exception& operator=(nan_or_inf_exception&&) noexcept = default;
    nan_or_inf_exception(const std::string& msg): std::runtime_error(msg) {}
  };

  /**
   * \brief Holds a data value (that gets passed between transducers).
   *
   * This data value can be one of the following types:
   *   * null
   *   * integer: an integer, usually representing an index
   *   * float: a single float value
   *   * tensor: a float-valued tensor of any rank, in which
   *     * a scalar is represented as a tensor of shape {1}
   *     * a boolean is represented as a scalar with value either 0 or 1
   *   * symbol: a string
   *   * list: an ordered list of values
   *   * symbolic tensor: for internal use only.
   */
  class value_t {
    using varient_t = std::variant<long, scalar_t, tensor_t, symbol_t, list_t, symbolic_tensor_t>;
    std::shared_ptr<varient_t> v;


    friend dynet_computation_graph;
    friend transducer_graph_node;
    friend lambda_transducer_model;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(v);
    }

    /**
     * \brief Construct a null value
     */
    value_t();

    value_t(const value_t& x) = default;

    value_t(value_t&&) noexcept = default;

    value_t& operator=(const value_t&) = default;

    value_t& operator=(value_t&&) noexcept = default;

    /**
     * \brief Construct a null value
     *
     * \param x The boolean
     */
    explicit value_t(nullptr_t);

    /**
     * \brief Construct a boolean value
     *
     * Internally stored as a float tensor of shape {1},
     * in which 0 means false and 1 means true.
     *
     * \param x The boolean
     */
    explicit value_t(bool x);

    /**
     * \brief Construct an integer value
     *
     * Internally stores as a 64-bit signed integer
     *
     * \param x The integer
     */
    explicit value_t(int x);

    /**
     * \brief Construct an integer value
     *
     * Internally stores as a 64-bit signed integer
     *
     * \param x The integer
     */
    explicit value_t(unsigned int x);

    /**
     * \brief Construct an integer value
     *
     * Internally stores as a 64-bit signed integer
     *
     * \param x The integer
     */
    explicit value_t(long x);

    /**
     * \brief Construct an integer value
     *
     * Internally stores as a 64-bit signed integer
     *
     * \param x The integer
     */
    explicit value_t(unsigned long x);

    /**
     * \brief Construct a scalar value
     *
     * Internally stores as a float tensor of shape {1}
     *
     * \param x The scalar value
     */
    explicit value_t(float x);

    /**
     * \brief Construct a scalar value
     *
     * Internally stores as a float tensor of shape {1}
     *
     * \param x The scalar value
     */
    explicit value_t(double x);

    /**
     * \brief Construct a symbol
     *
     * Internally stores as a string
     *
     * \param x The symbol
     */
    explicit value_t(const symbol_t& x);

    /**
     * \brief Construct a scalar value
     *
     * Internally stores as a string
     *
     * \param x The symbol
     */
    explicit value_t(const char *x);

    /**
     * \brief Construct a symbolic tensor
     *
     * When tg::immediate_computation_guard is guarded, this constructor will immediately evaluate the symbolic tensor
     * and throws error if Inf or NaN is encountered.
     *
     * \param x The symbolic tensor
     */
    explicit value_t(const symbolic_tensor_t& x);

    /**
     * \brief Construct a tensor
     *
     * \param x The tensor
     */
    explicit value_t(const tensor_t& x);

    /**
     * \brief Construct a tensor
     *
     * \param x The tensor
     */
    explicit value_t(tensor_t&& x);

    /**
     * \brief Construct a list
     *
     * \param x The list of tg::typed_value
     */
    explicit value_t(const std::vector<value_t>& xs);

    /**
     * \brief Construct a list
     *
     * \param x The list of tg::typed_value
     */
    explicit value_t(std::vector<value_t>&& xs);

    /**
     * \brief Construct a list
     *
     * \tparam T A type that could be directly constructed into a tg::typed_value
     * \param x The list of values
     */
    template<typename T>
    explicit
    value_t(const std::vector<T>& xs, typename std::enable_if<!std::is_same<T, value_t>::value>::type * = nullptr) {
      std::vector<value_t> ret;
      ret.reserve(xs.size());
      for (auto&& x:xs) {
        ret.emplace_back(x);
      }
      v = std::make_shared<varient_t>(std::move(ret));
    }

    /**
     * \brief Construct a tg::value_t holding a list
     *
     * \param args One or more inputs that should goes into the list.
     * \return The constructed tg::value_t
     */
    template<typename ...T>
    static value_t make_list(const T& ...args) {
      return value_t(std::vector<value_t>({value_t(args)...}));
    }

    virtual ~value_t();

    /**
     * \brief Get the value as an integer
     *
     * Applicable when the value is of type integer, float, tensor and symbolic tensor.
     *
     * \return The integer value
     */
    long as_integer() const;

    /**
     * \brief Get the value as a float
     *
     * Applicable when the value is of type integer, float, tensor and symbolic tensor.
     * \return
     */
    float as_float() const;

    /**
     * \brief Get the value as a symbol
     *
     * Applicable when the value is a symbol.
     *
     * \return The symbol value
     */
    const symbol_t& as_symbol() const;

    /**
     * \brief Get the value as a symbolic tensor
     *
     * Applicable when the value is of type integer, float, tensor and symbolic tensor.
     *
     * \return The symbolic tensor value
     */
    symbolic_tensor_t as_symbolic_tensor() const;

    /**
     * \brief Get the value as a tensor
     *
     * Applicable when the value is of type integer, float, tensor.
     *
     * If this is a symbolic tensor, evaluate() need to be called first.
     *
     * \return The tensor value
     */
    tensor_t as_tensor() const;

    /**
     * \brief Get the value as a list
     *
     * Applicable when the value is of type list
     *
     * \return The list value
     */
    const list_t& as_list() const&;

    list_t as_list() const&&;

    /**
     * \brief Get the value as a tuple
     *
     * Applicable when the value is of type list
     *
     * \tparam N The size of the tuple, must be no greater than the list size.
     * \return The tuple value
     */
    template<size_t N>
    auto as_tuple() const {
      const list_t& list = as_list();

      if (list.size() < N) {
        std::stringstream ss;
        ss << "Cannot convert list of size " << list.size() << " into a tuple of size " << N;
        std::throw_with_nested(std::runtime_error(ss.str()));
      }

      return generate_type_consistent_tuple<N>([&](size_t i) -> const value_t& {
        return list.at(i);
      });
    }

    /**
     * \brief Get the value as a pair
     *
     * Applicable when the value is of type list with length >= 2
     *
     * \return The pair value
     */
    std::pair<value_t, value_t> as_pair() const;

    /**
     * \brief Get the shape of this tensor
     * \return the size of each axis
     */
    tensor_shape_t tensor_shape() const;

    /**
     * \brief Get the rank of this tensor
     * \return The rank of this tensor
     */
    unsigned long tensor_rank() const;

    /**
     * \brief Get a string representing the tensor shape, in the form of AxBxC.
     *
     * This is mainly for debugging purposes.
     *
     * \return the string representation.
     */
    std::string print_tensor_shape() const;


    /**
     * \brief Get the value type as a human-readable name
     *
     * This is only for debugging purposes.
     * There is no guarantee that the name of a type stays unchanged through future versions.
     * If you want to programmatically check the value type,
     * please use is_null(), is_integer(), etc. instead.
     *
     * \return The name of the value type
     */
    std::string type_name() const;

    /**
     * \brief Calculate the number of elements in this tensor
     * by multiplying the size of all axis.
     *
     * \return the total number of dimensions
     */
    unsigned long tensor_num_elements() const;

    /**
     * \brief Check if this value is of type null
     * \return If this value is of type null
     */
    bool is_null() const;

    /**
     * \brief Check if this value is of type integer
     * \return If this value is of type integer
     */
    bool is_integer() const;

    /**
     * \brief Check if this value is of type float
     * \return If this value is of type float
     */
    bool is_float() const;

    /**
     * \brief Check if this value is either integer or float
     * \return If this value is either integer or float
     */
    bool is_any_scalar() const;

    /**
     * \brief Check if this value is of type symbol
     * \return If this value is of type symbol
     */
    bool is_symbol() const;

    /**
     * \brief Check if this value is of type tensor
     * \return If this value is of type tensor
     */
    bool is_tensor() const;

    /**
     * \brief Check if this value is of type symbolic tensor
     * \return If this value is of type symbolic tensor
     */
    bool is_symbolic_tensor() const;

    /**
     * \brief Check is this value is either tensor or symbolic tensor
     * \return If this value is either tensor or symbolic tensor
     */
    bool is_any_tensor() const;

    /**
     * \brief Check if this value is of type list
     * \return If this value is of type list
     */
    bool is_list() const;

    /**
     * \brief Select an item from a list
     * \param i The index of the item (supports negative indexing, just like python)
     * \return Item #i in the list
     */
    value_t select(long i) const;


    /**
     * \brief Select an item from a list
     *
     * Identical to select(long i) const
     *
     * \param i The index of the item (supports negative indexing, just like python)
     * \return Item #i in the list
     */
    value_t operator[](long i) const;

    /**
     * \brief Select a sub-tensor from a tensor of higher rank.
     *
     * The resulting tensor will have rank reduced by 1.
     *
     * \param i the index of the sub-tensor within the higher rank tensor
     * \param axis the axis along which the sub-tensors are chosen from
     * \return the selected sub-tensor
     */
    value_t select(unsigned long i, unsigned long axis) const;

    /**
     * \brief Select multiple items from a list
     *
     * \param indices the indices of the items
     * \return the selected items
     */
    value_t select_many(const std::vector<unsigned long>& indices) const;

    /**
     * \brief Select multiple items from a list
     *
     * \param indices the indices of the items (supports negative indexing, just like python)
     * \return the selected items
     */
    value_t select_many(const std::vector<long>& indices) const;

    /**
     * \brief Select multiple sub-tensors from a tensor.
     *
     * \param indices the indices of the sub-tensors
     * \param axis the axis along which the sub-tensors are chosen from
     * \return the selected sub-tensors
     */
    value_t select_many(const std::vector<unsigned long>& indices, unsigned long axis) const;

    /**
     * \brief Select consecutive items in a list
     *
     * \param start the starting index
     * \param end the ending index (exclusive)
     * \return the selected consecutive items (as a list)
     */
    value_t slice(unsigned long start, unsigned long end) const;

    /**
     * \brief Select consecutive sub-tensors from a tensor
     *
     * \param start the starting index of the consecutive sub-tensors
     * \param end the ending index (exclusive)
     * \param axis the axis along which the sub-tensors are chosen from
     * \return the consecutive sub-tensors (as a tensor)
     */
    value_t slice(unsigned long start, unsigned long end, unsigned long axis) const;


    friend std::ostream& operator<<(std::ostream& os, const value_t& x);

    /**
     * \brief Access the value using std::visit style
     *
     * Use this if you want to write some function that handle cases of most/all types.
     * This is faster when compared with a bunch of if-else block.
     *
     * \param visitor The visitor function
     * \return What visitor function returns
     */
    template<typename _Visitor>
    constexpr decltype(auto) visit(_Visitor&& visitor) const {
      if (!v) std::throw_with_nested(std::runtime_error("Cannot get value from null"));
      return std::visit(visitor, *v);
    }

    struct type_info {
        bool is_integer;
        bool is_float;
        bool is_tensor;
        bool is_symbol;
        bool is_list;
        bool is_symbolic_tensor;
        bool is_any_scalar;
        bool is_any_tensor;

      constexpr type_info(bool isInteger, bool isFloat, bool isTensor, bool isSymbol, bool isList, bool isSymbolicTensor)
        : is_integer(isInteger), is_float(isFloat), is_tensor(isTensor), is_symbol(isSymbol), is_list(isList),
          is_symbolic_tensor(isSymbolicTensor), is_any_scalar(isInteger || isFloat), is_any_tensor(isTensor || isSymbolicTensor) {}
    };

    /**
     * \brief Query all possible types that may be held by a value_t::variant_t at compile value.
     *
     * Useful when writing the visitor of visit() or visit_many()
     *
     * \tparam T one of the possible types held by value_t::variant_t
     * \return the type information
     */
    template<typename T>
    static constexpr type_info static_type_info() {
      using V = std::decay_t<T>;
      return type_info(
        std::is_same_v<V, long>, std::is_same_v<V, scalar_t>, std::is_same_v<V, tensor_t>,
        std::is_same_v<V, symbol_t>, std::is_same_v<V, list_t>, std::is_same_v<V, symbolic_tensor_t>
      );
    }

  private:
    static bool is_any_null(const value_t& x) {
      return x.is_null();
    }

    template<typename... Values>
    static bool is_any_null(const value_t& x, Values&& ... others) {
      return x.is_null() || is_any_null(others...);
    }

  public:

    /**
     * \brief Access multiple values using std::visit style
     *
     * Use this if you want to write some function that handle cases of most/all types.
     * This is faster when compared with a bunch of if-else block.
     *
     * \param visitor The visitor function
     * \param values The values to visit
     * \return What visitor function returns
     */
    template<typename _Visitor, typename... Values>
    static constexpr decltype(auto) visit_many(_Visitor&& visitor, Values&& ... values) {
      if (is_any_null(values...)) std::throw_with_nested(std::runtime_error("Cannot get value from null"));
      return std::visit(visitor, (*values.v)...);
    }

    /**
     * \brief Evaluate this tensor inplace, converting from symbolic tensor to numeric tensor.
     *
     * If this value is a list, all nested tensors will be evaluated.
     *
     */
    void evaluate();

  private:

    void collect_nested_symbolic_tensors(std::unordered_set<varient_t *>& ret);

  };

  /**
   * \brief Ensure that a tensor is not NaN or Inf
   *
   * Throws an exception if it does.
   *
   * \param tensor The tensor to ensure
   */
  void block_nan_or_inf(const dynet::Tensor& tensor);

  /**
   * \brief Ensure that a float value is not NaN or Inf
   *
   * Throws an exception if it does.
   *
   * \param value The value to ensure
   */
  void block_nan_or_inf(float value);

  /// @}
}


#endif //LEGO_TRANSDUCER_TYPED_VALUE_HPP
