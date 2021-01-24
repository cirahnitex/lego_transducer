//
// Created by Dekai WU and YAN Yuchen on 20200709.
//

#include "include/lego_training_pipeline.hpp"
#include "include/parallel_array_map.hpp"
#include "include/wallclock_timer.hpp"
#include "include/transducer_dataset.hpp"
#include "include/lego_initialize.hpp"
#include "dynet_computation_graph.hpp"
#include "include/transducer_dataset.hpp"
#include "include/lego_transducer.hpp"
#include "lambda_transducer_value_cache.hpp"
#include <iomanip>

using namespace tg;
using namespace std;

void tg::training_pipeline::set_num_epochs(unsigned long num_epochs) {
  this->num_epochs_m = num_epochs;
}

void tg::training_pipeline::set_batch_size(unsigned long batch_size) {
  this->batch_size_m = batch_size;
}

tg::event_emitter<>::listener_handle_t
tg::training_pipeline::add_before_training_example_listener(const tg::event_emitter<>::listener_t& listener) {
  return before_training_datum_listener_m.add_listener(listener);
}

void tg::training_pipeline::remove_before_training_example_listener(const tg::event_emitter<>::listener_handle_t& listener) {
  before_training_datum_listener_m.remove_listener(listener);
}

tg::event_emitter<>::listener_handle_t
tg::training_pipeline::add_new_best_listener(const tg::event_emitter<>::listener_t& listener) {
  return new_best_listener_m.add_listener(listener);
}

void tg::training_pipeline::remove_new_best_listener(const tg::event_emitter<>::listener_handle_t& listener) {
  new_best_listener_m.remove_listener(listener);
}

tg::event_emitter<>::listener_handle_t
tg::training_pipeline::add_epoch_completion_listener(const tg::event_emitter<>::listener_t& listener) {
  return epoch_completion_listener_m.add_listener(listener);
}

void tg::training_pipeline::remove_epoch_completion_listener(const tg::event_emitter<>::listener_handle_t& listener) {
  epoch_completion_listener_m.remove_listener(listener);
}


void tg::training_pipeline::set_report_interval(std::chrono::steady_clock::duration report_interval) {
  this->report_interval = report_interval;
}

void tg::training_pipeline::set_num_workers(unsigned long val) {
  this->num_workers_m = val;
}

unsigned long training_pipeline::num_epochs_completed() const {
  return num_epochs_completed_m;
}

namespace __private_VPHKVRydpm {
  std::string formatted_now()
  {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
  }
}
using namespace __private_VPHKVRydpm;

void training_pipeline::train_and_validate(transducer_model loss_model,
                                           const std::shared_ptr<const transducer_dataset>& training_set,
                                           const std::shared_ptr<const transducer_dataset>& validation_set) {

  return train_and_validate_impl(training_set, validation_set, [&](const std::shared_ptr<transducer_dataset>& batch) {
    return optimizer_m->learn_from_batch(loss_model, batch);
  }, [&](const std::shared_ptr<transducer_dataset>& batch) {
    float batch_loss = 0;
    for(auto&& loss:loss_model.batch_transduce(batch)) {
      batch_loss += loss.as_float();
    }
    return batch_loss;
  });
}

void training_pipeline::dynamic_train(
  const std::vector<dynamic_transducer_application>& transducer_applications_for_training) {

  auto training_set_ids = create_transducer_dataset(1);
  for(unsigned long i=0; i<transducer_applications_for_training.size(); ++i) {
    training_set_ids->emplace_back(i);
  }

  auto validation_set_ids = create_transducer_dataset(1);

  return train_and_validate_impl(training_set_ids, validation_set_ids, [&](const std::shared_ptr<transducer_dataset>& batch) {
    vector<dynamic_transducer_application> applications_batch;
    applications_batch.reserve(batch->size());
    for(auto&& datum : *batch) {
      applications_batch.push_back(transducer_applications_for_training.at(datum.at(0).as_integer()));
    }
    return optimizer_m->dynamic_learn_from_batch(applications_batch);
  }, [&](const std::shared_ptr<transducer_dataset>& batch) {
    return 0;
  });
}

void training_pipeline::dynamic_train_and_validate(
  const std::vector<dynamic_transducer_application>& transducer_applications_for_training,
  const std::vector<dynamic_transducer_application>& transducer_applications_for_validation) {

  auto training_set_ids = create_transducer_dataset(1);
  for(unsigned long i=0; i<transducer_applications_for_training.size(); ++i) {
    training_set_ids->emplace_back(i);
  }

  auto validation_set_ids = create_transducer_dataset(1);
  for(unsigned long i=0; i<transducer_applications_for_validation.size(); ++i) {
    validation_set_ids->emplace_back(i);
  }

  return train_and_validate_impl(training_set_ids, validation_set_ids, [&](const std::shared_ptr<transducer_dataset>& batch) {
    vector<dynamic_transducer_application> applications_batch;
    applications_batch.reserve(batch->size());
    for(auto&& datum : *batch) {
      applications_batch.push_back(transducer_applications_for_training.at(datum.at(0).as_integer()));
    }
    return optimizer_m->dynamic_learn_from_batch(applications_batch);
  }, [&](const std::shared_ptr<transducer_dataset>& batch)->float {

    vector<dynamic_transducer_application> applications_batch;
    applications_batch.reserve(batch->size());
    for(auto&& datum : *batch) {
      applications_batch.push_back(transducer_applications_for_validation.at(datum.at(0).as_integer()));
    }

    auto losses = transducer_instance::dynamic_batch_apply(applications_batch);

    float ret = 0;
    for(auto&& loss:losses) {
      ret += loss.as_float();
    }

    return ret;
  });
}

void training_pipeline::train(transducer_model loss_model, const std::shared_ptr<const transducer_dataset>& training_set) {
  train_and_validate(loss_model, training_set, create_transducer_dataset(0));
}

unsigned long training_pipeline::num_training_datums_completed() const {
  return num_training_datums_completed_m;
}

training_pipeline::training_pipeline(optimizer_base *optimizer): optimizer_m(optimizer) {

}

float training_pipeline::validate_impl(const std::shared_ptr<const transducer_dataset>& validation_set,
                                       const std::function<float(
                                         const std::shared_ptr<transducer_dataset>&)>& compute_loss_from_validation_set_batch) const {
  parallel_map_thread_pool workers(num_workers_m);

  auto batched_validation_set = validation_set->group_to_batch(batch_size_m);

  workers.for_each_worker([&]() {
    preallocate_thread_local_mempool();
  });

  std::mutex loss_mtx;
  float sum_validation_loss = 0;

  workers.for_each<std::shared_ptr<transducer_dataset>>(batched_validation_set, [&](const std::shared_ptr<transducer_dataset>& batch) {

    // compute the current batch loss
    float batch_loss = compute_loss_from_validation_set_batch(batch);

    // accumulate the loss
    {
      loss_mtx.lock();
      sum_validation_loss += batch_loss;
      loss_mtx.unlock();
    }

  });

  return sum_validation_loss / (float)validation_set->size();
}

float
training_pipeline::validate(transducer_model loss_model, const shared_ptr<const transducer_dataset>& validation_set) const {
  return validate_impl(validation_set, [&](const std::shared_ptr<transducer_dataset>& batch) {
    float batch_loss = 0;
    for(auto&& loss:loss_model.batch_transduce(batch)) {
      batch_loss += loss.as_float();
    }
    return batch_loss;
  });
}

float training_pipeline::dynamic_validate(
  std::vector<dynamic_transducer_application>& transducer_applications_for_validation) const {
  auto validation_set_ids = create_transducer_dataset(1);
  for(unsigned long i=0; i<transducer_applications_for_validation.size(); ++i) {
    validation_set_ids->emplace_back(i);
  }
  return validate_impl(validation_set_ids, [&](const std::shared_ptr<transducer_dataset>& batch)->float {

    transducer_model tmp_transducer([&]()->value_placeholder{
      vector<value_placeholder> loss_transducers;
      loss_transducers.reserve(batch->size());
      for(auto&& datum:*batch) {
        loss_transducers.push_back(transducer_applications_for_validation.at(datum.at(0).as_integer())());
      }
      return list_sum(make_list.apply(loss_transducers));
    });


    return tmp_transducer().as_float();
  });
}

std::vector<value_t> training_pipeline::dynamic_transduce_many(
  const std::vector<dynamic_transducer_application>& applications) {
  auto ids = create_transducer_dataset(1);
  for(unsigned long i=0; i<applications.size(); ++i) {
    ids->emplace_back(i);
  }
  parallel_map_thread_pool workers(num_workers_m);
  auto batched_ids = ids->group_to_batch(batch_size_m);

  {
    workers.for_each_worker([&]() {
      preallocate_thread_local_mempool();
    });
  }

  std::vector<value_t> ret(ids->size());
  workers.for_each<std::shared_ptr<transducer_dataset>>(batched_ids, [&](const std::shared_ptr<transducer_dataset>& batch, unsigned long batch_index){
    std::vector<dynamic_transducer_application> batch_applications;
    for(auto&& _id:*batch) {
      batch_applications.push_back(applications.at(_id[0].as_integer()));
    }
    auto batch_result = transducer_instance::dynamic_batch_apply(batch_applications);
    unsigned long offset = batch_index*batch_size_m;
    for(unsigned long i=0; i<batch_result.size(); ++i) {
      ret[offset + i] = batch_result[i];
    }
  });

  return ret;
}

std::vector<value_t>
training_pipeline::transduce_many(transducer_model perf, const shared_ptr<const transducer_dataset>& dataset) {

  if(!perf.is_arity(dataset->arity())) {
    stringstream ss;
    ss << "Cannot transduce a dataset of arity "<< dataset->arity();
    throw std::runtime_error(ss.str());
  }

  parallel_map_thread_pool workers(num_workers_m);
  auto batched_dataset = dataset->group_to_batch(batch_size_m);

  {
    workers.for_each_worker([&]() {
      preallocate_thread_local_mempool();
    });
  }

  std::vector<value_t> ret(dataset->size());
  workers.for_each<std::shared_ptr<transducer_dataset>>(batched_dataset, [&](const std::shared_ptr<transducer_dataset>& batch, unsigned long batch_index){
    auto batch_result = perf.batch_transduce(batch);
    unsigned long offset = batch_index*batch_size_m;
    for(unsigned long i=0; i<batch_result.size(); ++i) {
      ret[offset + i] = batch_result[i];
    }
  });

  return ret;
}

namespace __private_lego_training_pipeline {
  void print_duration(ostream& os, const std::chrono::steady_clock::duration& duration) {
    if( duration < std::chrono::milliseconds(10)) {
      os << std::chrono::duration_cast<std::chrono::microseconds>(duration).count()/1000.0 << "ms";
    }
    else if(duration < std::chrono::seconds(10)) {
      os << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()/1000.0 << "s";
    }
    else {
      os << std::chrono::duration_cast<std::chrono::seconds>(duration).count() << "s";
    }
  }
}
using namespace __private_lego_training_pipeline;
void training_pipeline::train_and_validate_impl(const std::shared_ptr<const transducer_dataset>& training_set, const std::shared_ptr<const transducer_dataset>& validation_set, const std::function<float(const std::shared_ptr<transducer_dataset>&)>& learn_from_training_set_batch, const std::function<float(const std::shared_ptr<transducer_dataset>&)>& compute_loss_from_validation_set_batch) {
  if(!optimizer_m) {
    throw std::runtime_error("Cannot perform training because no optimizer is specified. Please provide an optimizer when constructing a training_pipeline.");
  }

  if(training_set->empty()) {
    cerr << "WARN: Training on empty dataset. Doing nothing." << endl;
    return;
  }

  parallel_map_thread_pool workers(num_workers_m);


  auto batched_training_set = training_set->group_to_batch(batch_size_m);
  auto batched_validation_set = validation_set->group_to_batch(batch_size_m);

  {
    wallclock_timer t;
    t.start();
    workers.for_each_worker([&]() {
      preallocate_thread_local_mempool();
    });
    cerr << "Training worker threads initialized in "<<t.milliseconds_elapsed()/(float)1000<<"s"<<endl;
  }


  for(unsigned long i_epoch = 0; i_epoch < num_epochs_m; ++i_epoch) {


    std::mutex loss_mtx;
    float sum_training_loss{};
    std::atomic<unsigned long> num_datums_trained{}; // # of datums trained within this epoch (excluding those failed)
    std::atomic<unsigned long> num_datums_failed{}; // # of datums failed to train with this epoch

    std::mutex report_mtx; // ensures that reporting is atomic
    wallclock_timer report_timer; // measures time spent for a report period
    report_timer.start();
    float sum_loss_since_last_report{};
    unsigned long report_starting_datum{};

    wallclock_timer epoch_timer; // measures the time spent for the entire epoch
    epoch_timer.start();

    // train on training set
    workers.for_each<std::shared_ptr<transducer_dataset>>(batched_training_set, [&](const std::shared_ptr<transducer_dataset>& batch) {

      // fire the before training datum event
      before_training_datum_listener_m.fire();

      float loss{};

      try {
        // compute the current batch loss and update the model
        loss = learn_from_training_set_batch(batch);
      }
      catch(std::exception& e) {
        num_datums_failed += batch->size();
        dynet_computation_graph::discard();
        throw;
      }


      // accumulate the current batch loss
      {
        loss_mtx.lock();
        sum_training_loss += loss;
        sum_loss_since_last_report += loss;
        loss_mtx.unlock();
      }


      // accumulate the # of datums trained
      {
        num_training_datums_completed_m += batch->size();
        num_datums_trained += batch->size();
      }


      // report if time elapsed since last report have reached the threshold
      if (report_interval != chrono::steady_clock::duration::zero()){
        report_mtx.lock();

        if(report_timer.time_elapsed() >= report_interval) {

          cerr << "datum#[" << report_starting_datum << "-" << (num_datums_trained - 1) <<"]";

          cerr << "\tloss: " << sum_loss_since_last_report << endl;
          report_starting_datum = num_datums_trained;
          sum_loss_since_last_report = 0;
          report_timer.start();
        }

        report_mtx.unlock();
      }
    });


    // validate on validation set
    bool is_new_best = false;
    float sum_validation_loss = 0;

    if(!batched_validation_set.empty()) {

      // calculate the validation set loss
      workers.for_each<std::shared_ptr<transducer_dataset>>(batched_validation_set, [&](const std::shared_ptr<transducer_dataset>& batch) {

        // compute the current batch loss

        float batch_loss = compute_loss_from_validation_set_batch(batch);

        // accumulate the loss
        {
          loss_mtx.lock();
          sum_validation_loss += batch_loss;
          loss_mtx.unlock();
        }

      });


      // update new best record if possible

      if(sum_validation_loss < best_validation_loss_m) {
        is_new_best = true;
        best_validation_loss_m = sum_validation_loss;
      }

    }

    cerr << "[" << formatted_now() << "] epoch#" << num_epochs_completed_m << " ";
    cerr << "spent: ";
    print_duration(cerr, epoch_timer.time_elapsed());
    cerr << ",\t";
    cerr << "training loss: " << sum_training_loss;
    if(!batched_validation_set.empty()) {
      cerr << ",\tvalidation loss: "<<  sum_validation_loss;
      if(is_new_best) {
        cerr << " (new best)";
      }
      else {
        cerr << " (prev best: "<< best_validation_loss_m <<")";
      }
      cerr << endl;
      // trigger event if a new best is encountered
      if(is_new_best) new_best_listener_m.fire();
    }
    else {
      cerr << endl;
    }

    num_epochs_completed_m++;
    epoch_completion_listener_m.fire();
  }
}
