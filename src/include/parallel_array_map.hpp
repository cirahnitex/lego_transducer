//
// Created by Dekai WU and YAN Yuchen on 20200622.
//

#ifndef TG_PARALLEL_ARRAY_MAP_HPP
#define TG_PARALLEL_ARRAY_MAP_HPP
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace tg {

  /**
   * \addtogroup parallel_computing
   *
   * @{
   */

  /**
   * \brief This is a polyfill for std::latch (in c++20)
   *
   * //todo: We need to upgrade to C++20 and remove this!
   */
  class latch
  {

    bool count_down(std::unique_lock<std::mutex> &)
    /// pre_condition (count_ > 0)
    {
      if(count_ == 0) throw std::runtime_error("Failed to count down a latch with counter = 0");
      if (--count_ == 0)
      {
        ++generation_;
        //lk.unlock();
        cond_.notify_all();
        return true;
      }
      return false;
    }

    bool try_count_down(std::unique_lock<std::mutex> &lk)
    {
      if (count_ > 0)
      {
        return count_down(lk);
      }
      return true;
    }
  public:

    latch(const latch&) = delete;
    latch& operator=(const latch&) = delete;

    /// Constructs a latch with a given count.
    latch(std::size_t count) :
      count_(count), generation_(0)
    {
    }

    ~latch()
    {

    }

    /// Blocks until the latch has counted down to zero.
    void wait()
    {
      std::unique_lock<std::mutex> lk(mutex_);
      if (count_ == 0) return;
      std::size_t generation(generation_);
      cond_.wait(lk, [&](){return generation != generation_;});
    }


    bool try_wait()
    {
      std::unique_lock<std::mutex> lk(mutex_);
      return (count_ == 0);
    }


    void count_down()
    {
      std::unique_lock<std::mutex> lk(mutex_);
      count_down(lk);
    }

    bool try_count_down()
    {
      std::unique_lock<std::mutex> lk(mutex_);
      return try_count_down(lk);
    }
    void signal()
    {
      count_down();
    }


    void count_down_and_wait()
    {
      std::unique_lock<std::mutex> lk(mutex_);
      std::size_t generation(generation_);
      if (count_down(lk))
      {
        return;
      }
      cond_.wait(lk, [&](){return generation != generation_;});
    }
    void sync()
    {
      count_down_and_wait();
    }


    void reset(std::size_t count)
    {
      std::lock_guard<std::mutex> lk(mutex_);
      count_ = count;
    }

  private:
    std::mutex mutex_;
    std::condition_variable cond_;
    std::size_t count_;
    std::size_t generation_;
  };


  /**
   * Performs a list_map operation using multithread,
   * transforming a list of input values into a list of output values.
   *
   * The transformation function should take:
   * * the input value
   * * [optional] the index of the input value
   * returns:
   * * the output value
   *
   * For efficiency, the current thread will also participate in the computation.
   * It is only when num_workers > 1 will any additional worker threads be spawn.
   *
   * \tparam from_T the type of each input value to map from
   * \tparam to_T the type of each output value to map to. Must have a default constructor.
   * \param ins the list of input value
   * \param fn the transformation function that turns an input value into an output value
   * \param num_workers number of worker threads to spawn
   * \return the list of output value
   */
  template<typename from_T, typename to_T>
  std::vector<to_T> parallel_array_map(const std::vector<from_T>& ins, const std::function<to_T(const from_T&, unsigned long i)>& fn, unsigned long num_workers) {
    std::vector<to_T> ret(ins.size());

    std::atomic<unsigned long> next_i(0);

    auto task = [&]() {
      for(unsigned long i = next_i++; i<ins.size(); i = next_i++) {
        ret[i] = fn(ins[i], i);
      }
    };

    if(num_workers <= 1) {
      task();
    }
    else {
      std::vector<std::thread> threads;
      for(unsigned long i=0; i<num_workers; ++i) {
        threads.emplace_back(task);
      }
      for(auto& thread:threads) {
        thread.join();
      }
    }


    return ret;
  }

  /**
   * \brief Similar to parallel_array_map(), but doesn't return any value. The tasks are performed for their side-effects only.
   * \tparam from_T the type of each input value
   * \param ins ns the list of input value
   * \param fn the task function that will be executes on every input value
   * \param num_workers number of worker threads to spawn
   */
  template<typename from_T>
  void parallel_for_each(const std::vector<from_T>& ins, const std::function<void(const from_T&, unsigned long i)>& fn, unsigned long num_workers) {

    std::atomic<unsigned long> next_i(0);

    auto task = [&]() {
      for(unsigned long i = next_i++; i<ins.size(); i = next_i++) {
        fn(ins[i], i);
      }
    };

    if(num_workers <= 1) {
      task();
    }
    else {
      std::vector<std::thread> threads;
      for(unsigned long i=0; i<num_workers; ++i) {
        threads.emplace_back(task);
      }
      for(auto& thread:threads) {
        thread.join();
      }
    }

  }

  template<typename from_T, typename to_T>
  std::vector<to_T> parallel_array_map(const std::vector<from_T>& ins, const std::function<to_T(const from_T&)>& fn, unsigned long num_workers) {
    return parallel_array_map<from_T, to_T>(ins, [&](const from_T& in, unsigned long){return fn(in);}, num_workers);
  }

  template<typename from_T>
  void parallel_for_each(const std::vector<from_T>& ins, const std::function<void(const from_T&)>& fn, unsigned long num_workers) {
    parallel_for_each<from_T>(ins, [&](const from_T& in, unsigned long){return fn(in);}, num_workers);
  }


  /**
   * \brief Spawns a thread pool to perform parallel_array_map() or parallel_for_each()
   *
   * It provides the same functionalities as parallel_array_map() or parallel_for_each(),
   * But the threads are created only once and reused.
   *
   * Also it has progress reporting built-in.
   *
   * This will eliminate overhead of initializing thread_local variables each time when creating new threads, if you believe this overhead is non-trivial in your task.
   *
   */
  class parallel_map_thread_pool {
    unsigned long num_workers_m;
    latch idle_latch;
    latch completion_latch;
    bool should_terminate_m{false};
    std::function<void()> task_m;
    std::vector<std::thread> threads;

    // The prefix of the progress report
    std::string report_prefix_m{};

    // The interval between which progress are reported
    // Zero means no report
    std::chrono::steady_clock::duration report_interval_m{std::chrono::steady_clock::duration::zero()};

  public:
    explicit parallel_map_thread_pool(unsigned num_workers):num_workers_m(num_workers), idle_latch(num_workers + 1), completion_latch(num_workers + 1) {
      if(num_workers > 1) {
        for(unsigned long i=0; i<num_workers; ++i) {
          threads.emplace_back([&](){
            while (true) {
              idle_latch.count_down_and_wait();
              if(should_terminate_m) break;
              task_m();
              completion_latch.count_down_and_wait();
            }
          });
        }
      }
    }


    ~parallel_map_thread_pool() {
      should_terminate_m = true;
      idle_latch.count_down();
      for(auto&& thread:threads) {
        thread.join();
      }
    }

    /**
     * \brief Enabling progress reporting
     *
     * The report is formatted as follows:
     * My awesome task: 1%
     *
     * Where the prefix "My awesome task" is customizable.
     *
     * A report is generated when both the following criteria are true:
     *   - Time since last report has reached a certain threshold. (default: 10 sec)
     *   - At least 1% of the tasks has been completed since last report.
     *
     * \param report_prefix The prefix of the progress report
     * \param report_interval The time interval between each report.
     */
    void enable_progress_reporting(std::string report_prefix, std::chrono::steady_clock::duration report_interval = std::chrono::seconds(10)) {
      report_prefix_m = std::move(report_prefix);
      report_interval_m = report_interval;
    }

    /**
     * \brief Disable progress reporting
     */
    void disable_progress_reporting() {
      report_prefix_m = "";
      report_interval_m = std::chrono::steady_clock::duration::zero();
    }

    void for_each_worker(const std::function<void()>& task) {
      if(num_workers_m > 1) {
        task_m = task;
        completion_latch.reset(num_workers_m + 1);
        idle_latch.count_down_and_wait();
        idle_latch.reset(num_workers_m + 1);
        completion_latch.count_down_and_wait();
      }
      else {
        task();
      }
    }

    template<typename from_T>
    void for_each(const std::vector<from_T>& ins, const std::function<void(const from_T&, unsigned long i)>& fn) {
      using steady_clock = std::chrono::steady_clock;

      std::atomic<unsigned long> next_i(0);

      if(report_interval_m == steady_clock::duration::zero()) {
        for_each_worker([&]() {
          for(unsigned long i = next_i++; i<ins.size(); i = next_i++) {
            fn(ins[i], i);
          }
        });
        return;
      }

      steady_clock::time_point last_report_time = steady_clock::now();
      unsigned long last_report_check_i = 0;
      std::mutex report_mtx;
      const unsigned long minimum_items_between_report_checks = std::ceil(ins.size()/100);

      for_each_worker([&]() {
        for(unsigned long i = next_i++; i<ins.size(); i = next_i++) {
          fn(ins[i], i);
          if(i - last_report_check_i >= minimum_items_between_report_checks) {
            std::lock_guard<std::mutex> lock(report_mtx);
            last_report_check_i = i;
            auto now = steady_clock::now();
            if(now - last_report_time >= report_interval_m) {
              last_report_time = now;
              if(!report_prefix_m.empty()) {
                std::cerr << report_prefix_m << ": ";
              }
              std::cerr << i*100/ins.size() << "%" << std::endl;
            }
          }
        }
      });
    }

    template<typename from_T>
    void for_each(const std::vector<from_T>& ins, const std::function<void(const from_T&)>& fn) {
      for_each<from_T>(ins, [&](const from_T& x, unsigned long i) {
        fn(x);
      });
    }

    template<typename from_T, typename to_T>
    std::vector<to_T> array_map(const std::vector<from_T>& ins, const std::function<to_T(const from_T&, unsigned long i)>& fn) {
      std::vector<to_T> ret(ins.size());

      for_each<from_T>(ins, [&](const from_T& x, unsigned long i) {
        ret[i] = fn(x, i);
      });

      return ret;
    }

    template<typename from_T, typename to_T>
    std::vector<to_T> array_map(const std::vector<from_T>& ins, const std::function<to_T(const from_T&)>& fn) {
      std::vector<to_T> ret(ins.size());

      for_each<from_T>(ins, [&](const from_T& x, unsigned long i) {
        ret[i] = fn(x);
      });

      return ret;
    }
  };

  /// @}
}

#endif //TG_PARALLEL_ARRAY_MAP_HPP
