#include "dynet/globals.h"
#include "dynet/devices.h"
#include "dynet/timing.h"
#include <memory>
#ifdef HAVE_CUDA
#include "dynet/cuda.h"
#endif

namespace dynet {

thread_local std::shared_ptr<std::mt19937> rndeng(std::make_shared<std::mt19937>(std::random_device()()));
Device* default_device = nullptr;
float default_weight_decay_lambda;
int autobatch_flag; 
int profiling_flag = 0;
thread_local NamedTimer timer;

}
