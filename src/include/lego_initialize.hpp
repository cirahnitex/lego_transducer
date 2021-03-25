//
// Created by Dekai WU and YAN Yuchen on 20200512.
//

#ifndef LEGO_LEGO_INITIALIZE_HPP
#define LEGO_LEGO_INITIALIZE_HPP
#include <vector>

namespace tg {
  enum ProfilingVerbosity {NONE=0, ROUGH, VERBOSE};

  enum Device_ID {CPU=0,
    GPU_0, GPU_1, GPU_2, GPU_3, GPU_4, GPU_5, GPU_6, GPU_7, GPU_8, GPU_9, GPU_10, GPU_11, GPU_12, GPU_13, GPU_14, GPU_15, GPU_16, GPU_17, GPU_18, GPU_19, GPU_20, GPU_21, GPU_22, GPU_23, GPU_24, GPU_25, GPU_26, GPU_27, GPU_28, GPU_29, GPU_30, GPU_31, GPU_32, GPU_33, GPU_34, GPU_35, GPU_36, GPU_37, GPU_38, GPU_39, GPU_40, GPU_41, GPU_42, GPU_43, GPU_44, GPU_45, GPU_46, GPU_47, GPU_48, GPU_49, GPU_50, GPU_51, GPU_52, GPU_53, GPU_54, GPU_55, GPU_56, GPU_57, GPU_58, GPU_59, GPU_60, GPU_61, GPU_62, GPU_63, GPU_64, GPU_65, GPU_66, GPU_67, GPU_68, GPU_69, GPU_70, GPU_71, GPU_72, GPU_73, GPU_74, GPU_75, GPU_76, GPU_77, GPU_78, GPU_79, GPU_80, GPU_81, GPU_82, GPU_83, GPU_84, GPU_85, GPU_86, GPU_87, GPU_88, GPU_89, GPU_90, GPU_91, GPU_92, GPU_93, GPU_94, GPU_95, GPU_96, GPU_97, GPU_98, GPU_99};

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
   * \param devices The list of devices to use. Leaving it empty to auto select.
   * \param profiling For internal use only
   */
  void lego_initialize(unsigned memory = 512, const std::vector<tg::Device_ID>& devices={}, ProfilingVerbosity profiling= NONE);

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
