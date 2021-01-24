//
// Created by Dekai WU and YAN Yuchen on 20200709.
//

#ifndef LEGO_LEGO_TRAINING_PIPELINE_HPP
#define LEGO_LEGO_TRAINING_PIPELINE_HPP
#include "transducer_optimizer.hpp"
#include "transducer_model.hpp"

#include <atomic>

namespace tg {

  /**
   * \addtogroup training
   * @{
   */

  /**
   * \brief A simple training pipeline for training a given model on a given dataset.
   *
   * It provides simple interface that performs the following routines:
   *   - [optional] Group the dataset in batches
   *   - [optional] Shuffle the (batched) dataset // todo: add shuffle functionality
   *   - Spawn one or more training worker threads
   *   - Train on the training set for some epochs
   *   - [optional] After each epoch, validate the model against a given validation set
   *   - [optional] Report progress during training
   */
  class training_pipeline {
    std::unordered_set<backprop_trainable_parameter_base*> params_to_train;
    unsigned long num_epochs_m{1};
    std::chrono::steady_clock::duration report_interval{std::chrono::seconds(10)};
    unsigned long num_workers_m{1};
    unsigned long batch_size_m{1};
    event_emitter<> epoch_completion_listener_m;
    event_emitter<> new_best_listener_m;
    event_emitter<> before_training_datum_listener_m;

    optimizer_base* optimizer_m;


    /**
     * \brief Stores the number of training datum that have been trained across all epochs (failed datum does not count)
     */
    std::atomic<unsigned long> num_training_datums_completed_m;

    /**
     * \brief Stores the number of epochs completed.
     *
     */
    unsigned long num_epochs_completed_m{0};

    float best_validation_loss_m{std::numeric_limits<float>::infinity()};


  public:

    /**
     * \brief Create a training pipeline
     *
     * If you need to train the model, you should supply an optimizer (SGD or Adam, etc).
     *
     * If you don't need to train the model and just want to use this training pipeline for validation and testing purposes, you can omit the optimizer.
     *
     * \param optimizer A pointer to the optimizer.
     */
    explicit training_pipeline(optimizer_base* optimizer=nullptr);


    /**
     * \brief Get the number of training examples completed.
     *
     * Use this if you want to manually schedule the learning rate depending on the number of examples trained.
     *
     * \return the number of training examples completed currently
     */
    unsigned long num_training_datums_completed() const;

    /**
     * \brief Get the number of epochs completed.
     *
     * Use this if you want to manually schedule the learning rate depending on the number of epochs completed.
     *
     * \return the number of epochs completed currently
     */
    unsigned long num_epochs_completed() const;


    /**
     * \brief Set how much time to wait before reporting the training progress
     *
     * Setting to zero will disable the progress reporting
     *
     * \param report_interval the time to wait between reports, in milliseconds
     */
    void set_report_interval(std::chrono::steady_clock::duration report_interval);


    /**
     * \brief Set the number of worker threads to spawn.
     *
     * Having N threads training concurrently will increase the memory usage roughly by a factor of N.
     *
     * Note that when using GPU, the training speed might be bottlenecked by GPU so increasing number of training threads might not be beneficial depending on the hardware and your model.
     *
     * \param workers number of worker threads to spawn
     */
    void set_num_workers(unsigned long workers);

    /**
     * \brief Set the number of epochs to train.
     * \param num_epochs
     */
    void set_num_epochs(unsigned long num_epochs);

    /**
     * \brief Set the size of a batch.
     *
     * \param batch_size the new batch size
     */
    void set_batch_size(unsigned long batch_size);

    /**
     * \brief Train a model on a training set.
     * \param loss_model the transducer model that returns the loss to minimize.
     * \param training_set The training set as a list of training examples (with oracle),
     *                     where each training example is a list of values that will be feed to your loss model.
     */
    void train(transducer_model loss_model, const std::shared_ptr<const transducer_dataset>& training_set);

    /**
     * \brief Train a dynamic transducer on a training set
     *
     * You need to provide a list of transducer applications, each of which is applied on a datum in the training set.
     *
     * \param transducer_applications_for_training a list of dynamic transducer applications
     */
    void dynamic_train(const std::vector<dynamic_transducer_application>& transducer_applications_for_training);

    /**
     * \brief Train a model on a training set while validating it on a validation set
     * \param loss_model the transducer model that returns the loss to minimize.
     * \param training_set the training set as a list of training examples (with oracle).
     *                     where each training example is a list of values that will be feed to your loss model.
     * \param validation_set The validation set as a list of validation examples
     *                       where each validation example is a list of values that will be feed to your loss model.
     */
    void train_and_validate(transducer_model loss_model, const std::shared_ptr<const transducer_dataset>& training_set, const std::shared_ptr<const transducer_dataset>& validation_set);

    /**
     * \brief Train a dynamic transducer on a training set while validating it on a validation set
     *
     * <b>For the training set:</b>
     * You need to provide a list of transducer applications, each of which is applied on a datum in the training set.
     *
     * <b>For the validation set:</b>
     * You need to provide a list of transducer applications, each of which is applied on a datum in the validation set.
     *
     * \param transducer_applications_for_training
     * \param transducer_applications_for_validation
     */
    void dynamic_train_and_validate(const std::vector<dynamic_transducer_application>& transducer_applications_for_training, const std::vector<dynamic_transducer_application>& transducer_applications_for_validation);

    /**
     * \brief Validate a model on a validation set
     * \param loss_model The transducer model that returns the loss to minimize.
     * \param validation_set The validation set as a list of validation examples
     *                       where each validation example is a list of values that will be feed to your loss model.
     * \return The average validation loss per datum
     */
    [[nodiscard]] float validate(transducer_model loss_model, const std::shared_ptr<const transducer_dataset>& validation_set) const;


    /**
     * \brief Validate a dynamic transducer on a validation set
     *
     * You need to provide a list of transducer applications, each of which is applied on a datum in the validation set.
     *
     * \param transducer_applications_for_validation the list of transducer validations
     * \return The average validation loss per datum
     */
    [[nodiscard]] float dynamic_validate(std::vector<dynamic_transducer_application>& transducer_applications_for_validation) const;

    /**
     * \brief Apply a model on a dataset.
     *
     * Apply the model on every datum of the dataset.
     *
     * When batch_size is specified, the data will be processed in batches.
     * When num_workers is specified, the data will be processed in multiple threads in parallel.
     *
     * \param perf The model to apply
     * \param dataset The dataset to be applied
     * \return A list containing the output from each datum.
     */
    std::vector<value_t> transduce_many(transducer_model perf, const std::shared_ptr<const transducer_dataset>& dataset);

    /**
     * \brief Apply a dynamic transducer on a dataset
     *
     * You need to provide a list of transducer applications, each of which is applied on a datum.
     *
     * When batch_size is specified, the data will be processed in batches.
     * When num_workers is specified, the data will be processed in multiple threads in parallel.
     *
     * \param applications The dynamic transducer applications
     * \return A list containing the output of each transducer application.
     */
    std::vector<value_t> dynamic_transduce_many(const std::vector<dynamic_transducer_application>& applications);

    /**
     * \brief Register a listener that will be triggered
     * before a training example is about to feed into the loss function.
     *
     * The major usecase for this listener is to adjust the learning rate on the fly
     * (thus applying a custom learning rate scheduler).
     *
     * When batch size > 1, this event will trigger before every batch.
     * For example, for batch size = 16, this event will trigger once every 16 examples.
     *
     * <b>ENSURE THREAD SAFETY</b>
     * Your listener will be executed in the worker thread, which means it may be called concurrently when multiple worker threads are executing in parallel.
     *
     * \param listener the callback function to invoke
     * \return A handle to your listener.
     *         Please keep this handle if you want to unregister this listener later on.
     */
    event_emitter<>::listener_handle_t add_before_training_example_listener(const event_emitter<>::listener_t& listener);

    /**
     * \brief Unregister a before-training-example listener.
     * \param listener
     */
    void remove_before_training_example_listener(const event_emitter<>::listener_handle_t& listener);

    /**
     * \brief Register a listener that will be triggered
     * when a new best validation score has been achieved.
     *
     * A good opportunity for you to save your model.
     *
     * Sidenote: This only triggers while train_and_validate() but not train()
     *
     * \param listener The callback function to invoke
     * \return A handle to your listener.
     *         Please keep this handle if you want to unregister this listener later on.
     */
    event_emitter<>::listener_handle_t add_new_best_listener(const event_emitter<>::listener_t& listener);

    /**
     * \brief Unregister a new best event listener
     * \param listener the listener handle
     */
    void remove_new_best_listener(const event_emitter<>::listener_handle_t& listener);

    /**
      * \brief Register a listener that will be triggered
      * when an epoch is completed.
      *
      * the event sequence during the lifetime of an epoch is as follows:
      *   (1) Training for an epoch
      *   (2) epoch completion event
      *   (3) validating against the validation set
      *   (4) (possibly) new best event
      * \param listener
      * \return A handle to your listener.
      */
    event_emitter<>::listener_handle_t add_epoch_completion_listener(const event_emitter<>::listener_t& listener);

    /**
     * \brief Unregister an epoch completion event listener
     * \param listener the listener handle
     */
    void remove_epoch_completion_listener(const event_emitter<>::listener_handle_t& listener);
    
  private:
    void train_and_validate_impl(const std::shared_ptr<const transducer_dataset>& training_set, const std::shared_ptr<const transducer_dataset>& validation_set, const std::function<float(const std::shared_ptr<transducer_dataset>&)>& learn_from_training_set_batch, const std::function<float(const std::shared_ptr<transducer_dataset>&)>& compute_loss_from_validation_set_batch);

    float validate_impl(const std::shared_ptr<const transducer_dataset>& validation_set, const std::function<float(const std::shared_ptr<transducer_dataset>&)>& compute_loss_from_validation_set_batch) const;
  };

}
#endif //LEGO_LEGO_TRAINING_PIPELINE_HPP
