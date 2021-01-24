#ifndef DYNET_GLOBALS_H
#define DYNET_GLOBALS_H

#include <random>
#include <memory>

namespace dynet {

class Device;
class NamedTimer;

extern thread_local std::shared_ptr<std::mt19937> rndeng;
extern Device* default_device;
extern thread_local NamedTimer timer; // debug timing in executors.

} // namespace dynet

#endif
