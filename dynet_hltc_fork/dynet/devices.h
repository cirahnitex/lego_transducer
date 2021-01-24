#ifndef DYNET_DEVICES_H
#define DYNET_DEVICES_H

#include <unordered_map>
#include <string>
#include <exception>
#include <thread>
#include <atomic>
#if HAVE_CUDA
#include <curand.h>
#endif
#include "dynet/aligned-mem-pool.h"
#include "dynet/cuda.h"
#include "dynet/globals.h"
#include "dynet/device-structs.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace Eigen {
  struct DefaultDevice;
  class CudaStreamDevice;
  struct GpuDevice;
}

namespace dynet {


// changed the memory allocation algorithm so that each thread uses its own memory pool (expect that parameter pool is still shared)
class Device {
 protected:
  Device(int i, DeviceType t, MemAllocator* m, DeviceMempoolSizes sizes_mb) : device_id(i), type(t), mem(m), pools(this), sizes_mb(std::move(sizes_mb)){}
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  virtual ~Device();
 public:
  void reset_rng(unsigned seed) {};
  int device_id;
  DeviceType type;
  MemAllocator* mem;
  float* kSCALAR_MINUSONE;
  float* kSCALAR_ONE;
  float* kSCALAR_ZERO;
  std::string name;
  virtual DeviceMempoolSizes mark(ComputationGraph *cg);
  virtual void revert(const DeviceMempoolSizes & cp);
  void allocate_tensor(DeviceMempool mem_pool, Tensor & tensor);

  class device_pools_getter{
  public:
    Device* dev;
    explicit device_pools_getter(Device* dev):dev(dev){}
    inline AlignedMemoryPool* operator[](unsigned i) {
        return &*dev->get_pool(i);
    }
    inline AlignedMemoryPool* operator[](unsigned i) const {
      return &*dev->get_pool(i);
    }
  };

  device_pools_getter pools;

  /**
   * \brief Get the memory pool
   *
   * Will create the pools if necessary. When created, pool #0, #1 and #3 are thread_local, while pool #2 is global.
   *
   * \param pool
   * \return
   */
  std::shared_ptr<AlignedMemoryPool> get_pool(unsigned int pool);

  void ensure_init_thread_local_pools();

 private:

  static thread_local std::unordered_map<const Device*, std::vector<std::shared_ptr<AlignedMemoryPool>>> thread_local_pools;
  DeviceMempoolSizes sizes_mb;

 protected:
  virtual std::shared_ptr<AlignedMemoryPool> create_pool(size_t initial_cap) = 0;
  std::shared_ptr<AlignedMemoryPool> shared_pool{};
};

#if HAVE_CUDA
class Device_GPU : public Device {
 public:
  typedef Eigen::CudaStreamDevice EigenDevice;
  explicit Device_GPU(int my_id, const DeviceMempoolSizes & mb, int device_id, unsigned seed);
  ~Device_GPU();
  void reset_rng(unsigned seed);
  int cuda_device_id;
  cublasHandle_t cublas_handle;
#if HAVE_CUDNN
  cudnnHandle_t cudnnHandle;
#endif
  Eigen::GpuDevice* edevice;
  Eigen::CudaStreamDevice* estream;
  GPUAllocator gpu_mem;
  curandGenerator_t curandeng;
 protected:
  std::shared_ptr<AlignedMemoryPool> create_pool(size_t initial_cap) override;
};
#endif

class Device_CPU : public Device {
 public:
  typedef Eigen::DefaultDevice EigenDevice;
  explicit Device_CPU(int my_id, const DeviceMempoolSizes & mb, bool shared);
  ~Device_CPU();
  CPUAllocator cpu_mem;
  Eigen::DefaultDevice* edevice;
  MemAllocator* shmem;
protected:
  std::shared_ptr<AlignedMemoryPool> create_pool(size_t initial_cap) override;
};

class DeviceManager final {
 public:
  DeviceManager();
  ~DeviceManager();

  void clear();

  void add(Device* d);

  Device* get(size_t i) { return devices[i]; }

  size_t num_devices() const { return devices.size(); }

  const std::vector<Device*>& get_devices() const { return devices; }

  Device* get_global_device(const std::string & name);

  // no copying allowed
  DeviceManager(const DeviceManager &) = delete;
  void operator=(const DeviceManager &) = delete;

 private:
  std::vector<Device*> devices;
  std::unordered_map<std::string, Device*> devices_map;
};

DeviceManager* get_device_manager();

inline void show_pool_mem_info() {
  DeviceManager* device_manager = get_device_manager();
  auto devs = device_manager->get_devices();
  if (devs.size() == 0) return;
  std::cerr << "\nMemory pool info for each devices:\n";
  for (Device* dev : devs) {
    std::cerr << " Device " << dev->name << " - FOR Memory " << (dev->get_pool(0)->get_cap() >> 20)
        << "MB, BACK Memory " << (dev->get_pool(1)->get_cap() >> 20)
        << "MB, PARAM Memory " << (dev->get_pool(2)->get_cap() >> 20)
        << "MB, SCRATCH Memory " << (dev->get_pool(3)->get_cap() >> 20) << "MB." << std::endl;
  }
}

} // namespace dynet

#endif
