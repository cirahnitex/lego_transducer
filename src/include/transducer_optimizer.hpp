//
// Created by Dekai WU and YAN Yuchen on 20200425.
//

#ifndef LEGO_TRANSDUCER_OPTIMIZER_HPP
#define LEGO_TRANSDUCER_OPTIMIZER_HPP

#include "transducer_instance.hpp"
#include "transducer_model.hpp"
#include "event_emitter.hpp"
#include <vector>
#include "transducer_typed_value.hpp"
#include <dynet/training.h>
#include <mutex>
#include "lego_param_naming_guard.hpp"

namespace tg {
  class backprop_trainable_parameter_base;

  /**
   * \addtogroup training
   * @{
   */

  /**
   * \brief An optimizer can update your model's backprop trainable parameters by running your model's loss function
   *
   * This class provides the bare minimum interface for an optmizer: You give it some datums and it updates your model parameters. All other stuffs related to the training pipeline are up to you to manage, here I list a few:
   *   - Group the dataset in minibatches
   *   - Spawn some worker threads to train it concurrently
   *   - Keep track of how many epochs have passed
   *   - Periodically validate against the validation set when appropriate
   *   - Periodically save your model when appropriate
   *   - Print out the training progress
   *
   * If you don't want to bother with the training pipeline and just want something simple, see tg::training_pipeline.
   *
   * An optimizer is thread safe. You have the same optimizer training on the same model concurrently.
   */
  class optimizer_base {
  protected:

    /**
     * \brief The list of weight parameters.
     *
     */
    std::unordered_set<backprop_trainable_parameter_base *> weights_to_train_m;

    /**
     * \brief The list of bias parameters.
     */
    std::unordered_set<backprop_trainable_parameter_base *> biases_to_train_m;

    /**
     * \brief Contains the weight parameters
     */
    dynet::ParameterCollection weights_pc_m;

    /**
     * \brief Contains the bias parameters
     */
    dynet::ParameterCollection biases_pc_m;

    std::mutex mtx; // locks when performing update

    virtual std::vector<dynet::Trainer*> get_impl() = 0;

    virtual void set_learning_rate_impl(float lr) = 0;

  public:
    /**
     * \brief Initialize the optimizer
     *
     * This optimizer will train on all existing parameters.
     */
    optimizer_base();

    virtual ~optimizer_base();

    /**
     * \brief Learn from one datum
     *
     * Run your loss function model on one datum. After that, update the model parameter.
     *
     * \param loss_fn The transducer model that returns the loss.
     * \param ins The content of the datum.
     * \return The loss value returned by your transducer.
     */
    template<typename ...T>
    float learn_from_datum(transducer_model loss_fn, T ...ins) {
      return apply_learn_from_datum(std::move(loss_fn), std::vector<value_t>({value_t(std::move(ins))...}));
    }

    /**
     * \brief Learn from one datum, using a dynamic transducer.
     *
     * \param compute_loss The dynamic transducer application that returns a loss
     * \return The loss value computed by the dynamic transducer
     */
    float dynamic_learn_from_datum(const dynamic_transducer_application& compute_loss);

    /**
     * \brief The non-variadic version of learn_from_datum()
     * \param loss_fn The transducer model that returns the loss.
     * \param inputs The datum.
     * \return The loss value returned by your transducer.
     */
    float apply_learn_from_datum(transducer_model loss_fn, const std::vector<value_t>& inputs);

    /**
     * \brief Learn from a minibatch of datums
     *
     * Run your loss function on multiple datums. After that, update the model parameter.
     *
     * This is generally faster than calling learn_from_datum() one by one, because tensor arithmetic is faster when executed in batches.
     *
     * \param loss_fn The transducer model that returns the loss
     * \param datum_batch The batch of datum
     * \return The summed loss returned by your transducer on the datums.
     */
    float learn_from_batch(transducer_model loss_fn, const std::shared_ptr<const transducer_dataset>& datum_batch);

    /**
     * \brief Learn from a minibatch of dynamic transducers
     * \param compute_loss_batch A list of dynamic transducers that each returns a loss
     * \return The summed loss returned by your dynamic transducers
     */
    float dynamic_learn_from_batch(const std::vector<dynamic_transducer_application>& compute_loss_batch);

    /**
     * \brief Adjust the learning rate
     * \param lr The new learning rate
     */
    void set_learning_rate(float lr);

    /**
     * \brief Enable weight decay.
     *
     * Weight decay is a regularization technique.
     * It is similar to L2 regularization, but computationally more efficient.
     *
     * You need to provide a decay coefficient,
     * which is similar to how L2 regularization's lambda works.
     *
     * Weight decay only applies to weights, not the biases.
     *
     * \param lambda The decay coefficient.
     */
    void set_weight_decay(float lambda);

    /**
     * \brief Exclude some parameters from training
     *
     * You need to provide a filter function that selects which parameters to exclude according to their param path.
     *
     * Param path can be assigned using tg::lego_param_naming_guard
     *
     * \param filter Returns true for parameters that needs to be excluded.
     */
    void exclude_params(const std::function<bool(const lego_param_path& param_path)>& filter);

  private:
    /**
     * \brief Update the parameters according to gradient.
     *
     * This update operation is atomic. Only one thread will be able to perform update at a time.
     * The gradient on all parameters will reset after updating, even if the update is not successful.
     * This is usually due to the magnitude of gradients have NaN or Inf.
     */
    void update_params();

    void take_ownership_of_params();
  };

  /**
   * \brief Stochastic Gradient Descend optimizer
   */
  class simple_sgd_optimizer : public tg::optimizer_base {
    dynet::SimpleSGDTrainer weights_impl, bias_impl;
  protected:
    std::vector<dynet::Trainer*> get_impl() override;

    void set_learning_rate_impl(float lr) override;

  public:

    /**
     * \brief Construct the optimizer
     * \param learning_rate
     */
    explicit simple_sgd_optimizer(float learning_rate = 0.01);


  };

  class adagrad_optimizer : public tg::optimizer_base {
    dynet::AdagradTrainer weights_impl, bias_impl;
  protected:
    std::vector<dynet::Trainer*> get_impl() override;
    void set_learning_rate_impl(float lr) override;
  public:

    explicit adagrad_optimizer(float learning_rate = 0.1, float eps = 1e-20);
  };

  /**
   * \brief Adam optimizer
   */
  class adam_optimizer : public tg::optimizer_base {
    dynet::AdamTrainer weights_impl, bias_impl;
  protected:
    std::vector<dynet::Trainer*> get_impl() override;

    void set_learning_rate_impl(float lr) override;

  public:

    /**
     * \brief Construct the optimizer
     * \param learning_rate
     * \param beta_1
     * \param beta_2
     * \param eps
     */
    explicit adam_optimizer(float learning_rate = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8);

  };

  /// @}
}

#endif //LEGO_TRANSDUCER_OPTIMIZER_HPP
