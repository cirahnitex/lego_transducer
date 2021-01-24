//
// Created by Dekai WU and YAN Yuchen on 20200615.
//

#ifndef TG_WALLCLOCK_TIMER_HPP
#define TG_WALLCLOCK_TIMER_HPP
#include <chrono>

namespace tg {
  /**
   * \addtogroup utilities
   * @{
   */


  /**
   * \brief A timer that measures time elapsed using wallclock time.
   *
   * Can be paused and resumed, useful when
   * measuring the cumulative time spent for some code segment that gets called repeatedly.
   *
   * The time resolution depends on the implementation of std::chrono::steady_clock. Usually it's a nanosecond.
   */
  class wallclock_timer {
    std::chrono::steady_clock::duration duration_m{};
    std::chrono::steady_clock::time_point start_m{};
    bool is_active_m{false};
    std::string autoreport_prefix_m{};
  public:
    /**
     * \brief Construct a timer.
     *
     * You can optionally provide an autoreport prefix. When provided, this timer will automatically report the elapsed time when it is destroyed, in the format of ${autoreport_prefix}${milliseconds_elapsed}
     *
     * \param label The prefix of the report
     */
    explicit wallclock_timer(std::string autoreport_prefix = ""):autoreport_prefix_m(std::move(autoreport_prefix)) {

    }

    ~wallclock_timer() {
      if(!autoreport_prefix_m.empty()) {
        std::cout << autoreport_prefix_m << milliseconds_elapsed() << std::endl;
      }
    }

    /**
     * \brief Start the timer, resetting the elapsed time to zero and starts measuring.
     */
    void start() {
      is_active_m = true;
      start_m = std::chrono::steady_clock::now();
      duration_m = std::chrono::steady_clock::duration();
    }

    /**
     * \brief Restart the timer
     *
     * Identical to start()
     *
     */
    inline void restart() {
      start();
    }

    /**
     * \brief Pause the timer, preventing the elapsed time from increasing.
     */
    void pause() {
      if(!is_active_m) return;
      duration_m += std::chrono::steady_clock::now() - start_m;
      is_active_m = false;
    }


    /**
     * \brief Resume the timer.
     */
    void resume() {
      if(is_active_m) return;
      is_active_m = true;
      start_m = std::chrono::steady_clock::now();
    }

    /**
     * \brief Query the elapsed time.
     * \return elapsed time in milliseconds
     */
    unsigned long milliseconds_elapsed() {
      return std::chrono::duration_cast<std::chrono::milliseconds>(time_elapsed()).count();
    }

    /**
     * \brief Query the elapsed time.
     * \return elapsed time in microseconds
     */
    unsigned long microseconds_elapsed() {
      return std::chrono::duration_cast<std::chrono::microseconds>(time_elapsed()).count();
    }

    std::chrono::steady_clock::duration time_elapsed() {
      if(is_active_m) {
        return duration_m + (std::chrono::steady_clock::now() - start_m);
      }
      else {
        return duration_m;
      }
    }

  };

  /// @}
}


#endif //TG_WALLCLOCK_TIMER_HPP
