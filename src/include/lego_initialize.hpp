//
// Created by Dekai WU and YAN Yuchen on 20200512.
//

#ifndef LEGO_LEGO_INITIALIZE_HPP
#define LEGO_LEGO_INITIALIZE_HPP

namespace tg {
  enum ProfilingVerbosity {NONE=0, ROUGH, VERBOSE};

  /**
   * \addtogroup global_configurations
   * @{
   */

  /**
   * \brief Initialize the library.
   *
   * Invoke this before you invoke any other functions of this library.
   *
   * \param memory The memory (in MB) budget for internal computation graph
   * \param disable_gpu Turn this flag on if you want to force using CPU even if GPU is available.
   * \param profiling For internal use only
   */
  void lego_initialize(unsigned memory = 512, bool disable_gpu = false, ProfilingVerbosity profiling= NONE);

  /**
   * \brief Preallocate the memory pool for the current thread
   *
   * When the first time a thread performs tensor arithmetic, it needs to allocate its tensor arithmetic memory pool, which can take around a second depending the amount of memory requested and the hardware devices.
   *
   * Calling this function will force the current thread to preallocate this memory pool immediately. This is useful if you want to measure the transduction time without accidentally taking into account the memory allocation time.
   *
   * This function is currently used in tg::lego_training_pipeline, to correctly measure the epoch execution time.
   *
   * If the current thread have already had its memory pool allocated, calling this function has no effect.
   */
  void preallocate_thread_local_mempool();


  /// @}

}

#endif //LEGO_LEGO_INITIALIZE_HPP
