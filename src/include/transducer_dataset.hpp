//
// Created by Dekai WU and YAN Yuchen on 20200709.
//

#ifndef LEGO_TRANSDUCER_DATASET_HPP
#define LEGO_TRANSDUCER_DATASET_HPP

#include "transducer_typed_value.hpp"

namespace tg {


  /**
   * \brief Contains a list of datums for a transducer to transduce on
   *
   * A dataset has an arity, which must match the arity of the transducer it is supposed to feed into.
   *
   * An N-ary dataset contains many N-ary datums, in which an N-ary datum can feed into an N-ary transducer.
   *
   * To construct a dataset, please use create_transducer_dataset()
   *
   */
  class transducer_dataset : public std::enable_shared_from_this<transducer_dataset> {
  public:

    class iterator {
      std::shared_ptr<const transducer_dataset> dataset_m;
      unsigned long i;
    public:
      explicit iterator(std::shared_ptr<const transducer_dataset> dataset, unsigned long _num = 0) : dataset_m(std::move(dataset)), i(_num) {}
      inline iterator& operator++() {++i; return *this;}
      inline iterator operator++(int) {iterator retval = *this; ++(*this); return retval;}
      inline bool operator==(const iterator& other) const {return i == other.i;}
      inline bool operator!=(const iterator& other) const {return i != other.i;}
      inline const std::vector<value_t>& operator*() {return dataset_m->at(i);}
      // iterator traits
      using difference_type = long;
      using value_type = std::vector<value_t>;
      using pointer = std::vector<value_t>*;
      using reference = const std::vector<value_t>&;
      using iterator_category = std::forward_iterator_tag;
    };

    transducer_dataset() = default;
    transducer_dataset(const transducer_dataset&) = default;
    transducer_dataset(transducer_dataset&&) noexcept = default;
    transducer_dataset& operator=(const transducer_dataset&) = default;
    transducer_dataset& operator=(transducer_dataset&&) noexcept = default;

    /**
     * \brief Get the arity of this dataset
     * \return The arity of this dataset
     */
    virtual unsigned long arity() const = 0;

    /**
     * \brief Get the number of data in this dataset.
     * \return The size of this dataset.
     */
    virtual unsigned long size() const = 0;

    /**
     * \brief Check if this dataset does not contain any data.
     * \return True if this dataset is empty.
     */
    virtual bool empty() const;

    /**
     * \brief Get a datum at given index
     * \param i The index of the datum
     * \return The datum
     */
    virtual const std::vector<value_t>& at(unsigned long i) const& = 0;

    /**
     * \brief Take a consecutive slice of this dataset
     *
     * This function will return a new dataset containing the slice. The original dataset is untouched.
     *
     * \param begin The starting index (inclusive)
     * \param end The ending index (exclusive)
     * \return The sliced dataset
     */
    virtual std::shared_ptr<transducer_dataset> slice(unsigned long begin, unsigned long end) const;

    /**
     * \brief Group the dataset in batches
     *
     * The last batch may not have enough items if the number of datums is not a multiple of the batch size.
     *
     * \param batch_size The batch size
     * \return The batches
     */
    virtual std::vector<std::shared_ptr<transducer_dataset>> group_to_batch(unsigned long batch_size) const;

    virtual iterator begin() const &;

    virtual iterator end() const &;

    virtual void save_to_stream(std::ostream& os) const;

    virtual void save_to_file(const std::string& path) const;

    static std::shared_ptr<transducer_dataset> load_from_stream(std::istream& is);

    static std::shared_ptr<transducer_dataset> load_from_file(const std::string& path);
  };


  /**
   * \brief A dataset that owns its underlying data.
   *
   * Because it owns it's underlying data, you can add new datums into it.
   */
  class transducer_dataset_vec_impl : public transducer_dataset {
    unsigned long arity_m{};
    std::vector<std::vector<value_t>> datums;

  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(arity_m, datums);
    }
    transducer_dataset_vec_impl() = default;
    transducer_dataset_vec_impl(const transducer_dataset_vec_impl&) = default;
    transducer_dataset_vec_impl(transducer_dataset_vec_impl&&) noexcept = default;
    transducer_dataset_vec_impl& operator=(const transducer_dataset_vec_impl&) = default;
    transducer_dataset_vec_impl& operator=(transducer_dataset_vec_impl&&) noexcept = default;
    explicit transducer_dataset_vec_impl(unsigned long arity);
    /**
     * \brief Get the arity of this dataset
     * \return The arity of this dataset
     */
    unsigned long arity() const override;

    /**
     * \brief Get the number of datums contained in this dataset
     * \return The number of datums contained in this dataset
     */
    unsigned long size() const override;

    /**
     * \brief Get a datum at given index
     * \param i The index of the datum
     * \return The datum
     */
    const std::vector<value_t>& at(unsigned long i) const& override;

    /**
      * \brief Insert an N-ary datum to this dataset
      *
      * An N-ary datum contains N values of type tg::value_t. So you must pass N arguments to this function, where each argument is tg::value_t constructable.
      *
      * N must match the arity of this dataset.
      *
      * \param datum_contents Contents of the datum you wish to insert
      */
    template<typename ...T>
    void emplace_back(T ...datum_contents) {
      apply_emplace_back(std::vector<value_t>{value_t(std::move(datum_contents))...});
    }

    /**
     * \brief The non-variadic version of emplace_back()
     * \param datum The datum you wish to insert
     */
    void apply_emplace_back(std::vector<value_t> datum);

    void save_to_stream(std::ostream& os) const override;

    void save_to_file(const std::string& path) const override;
  };

  /**
   * \brief Create a blank transducer dataset, in which you can insert datums.
   *
   * Insert datums by calling transducer_dataset_vec_impl::emplace_back()
   *
   * \param arity The arity of the dataset
   * \return The created blank dataset
   */
  std::shared_ptr<transducer_dataset_vec_impl> create_transducer_dataset(unsigned long arity);

  /**
   * \brief A reference to a slice/reordering of another dataset. It does not own its underlying datums.
   *
   */
  class transducer_dataset_ref_impl : public transducer_dataset {
    std::shared_ptr<const transducer_dataset> parent_m;
    std::vector<unsigned long> indices_m;
  public:
    transducer_dataset_ref_impl(std::shared_ptr<const transducer_dataset> parent,
                                std::vector<unsigned long> indices);

    unsigned long arity() const override;

    unsigned long size() const override;

    const std::vector<value_t>& at(unsigned long i) const& override;

  };
}

#endif //LEGO_TRANSDUCER_DATASET_HPP
