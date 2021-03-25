//
// Created by Dekai WU and YAN Yuchen on 20200512.
//

#include "include/lego_initialize.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <dynet/dynet.h>
using namespace tg;
using namespace std;

std::string get_dynet_device_name(tg::Device_ID device_id) {
  if(device_id == CPU) return "CPU";
  int gpu_idx = (int)device_id - 1;
  return "GPU:"+to_string(gpu_idx);
}

std::string get_dynet_devices_cmdarg(const std::vector<tg::Device_ID>& devices) {
  stringstream ss;
  ss << get_dynet_device_name(devices.front());
  for (long i = 1; i < devices.size(); ++i) {
    ss << "," << get_dynet_device_name(devices[i]);
  }
  return ss.str();
}

void tg::lego_initialize(unsigned int memory, const std::vector<tg::Device_ID>& devices, tg::ProfilingVerbosity profiling) {
  static bool is_initialized{false};
  if(is_initialized) throw std::runtime_error("lego_initialize() cannot be called twice");
  is_initialized = true;

  // silence the dynet output by temporarily disabling cerr
  std::cerr.setstate(std::ios_base::failbit);

  std::string quater_mem = std::to_string(memory / 4);
  std::vector<std::string> arguments = {
    "",
    "--dynet-mem=" + quater_mem + "," + quater_mem + "," + quater_mem + "," + quater_mem,
    "--dynet-autobatch=1",
    "--dynet-profiling=" + std::to_string((unsigned)profiling)
  };
  if(!devices.empty()) arguments.emplace_back("--dynet-devices=" + get_dynet_devices_cmdarg(devices));

  std::vector<char *> argv;
  for (const auto& arg : arguments)
    argv.push_back((char *) arg.data());
  argv.push_back(nullptr);

  int argc = (int) argv.size() - 1;
  char **argv2 = argv.data();
  auto dynet_params = dynet::extract_dynet_params(argc, argv2, true);
  dynet::initialize(dynet_params);

  // dynet internally uses the mt1997 RNG, but with one exception.
  // in interprocess, when generating queue names, it uses rand() instead of mt1997 RNG
  // so we also need to randomize this
  // otherwise you cannot have multiple dynet program running on the same machine! queue name clash!
  srand(dynet_params.random_seed);


  std::cerr.clear();
}

//bool tg::is_gpu_used() {
//  return dynet::default_device->type == dynet::DeviceType::GPU;
//}

void tg::preallocate_thread_local_mempool() {
  dynet::preallocate_thread_local_mempool();
}
