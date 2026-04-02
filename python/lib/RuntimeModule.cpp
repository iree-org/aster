//===- RuntimeModule.cpp -------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Python bindings for the HIP runtime module.
// HIP is loaded dynamically via dlopen so there is no compile-time link
// dependency on the HIP SDK. If libamdhip64.so is absent at runtime every
// binding throws std::runtime_error.
//
//===----------------------------------------------------------------------===//

#include "hip.h"
#include <cstdio>
#include <dlfcn.h>
#include <mutex>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

//===----------------------------------------------------------------------===//
// Dynamic library loading
//===----------------------------------------------------------------------===//

HipApi::~HipApi() {
  if (lib)
    ::dlclose(lib);
}

const HipApi *HipApi::load() {
  void *handle = ::dlopen("libamdhip64.so", RTLD_LAZY | RTLD_LOCAL);
  if (!handle)
    return nullptr;

  static HipApi api;
  bool ok = true;

  // Resolves one symbol into a typed function pointer; prints to stderr and
  // sets ok=false if the symbol is absent.
  auto loadSym = [&](auto &fp, const char *sym) {
    void *p = ::dlsym(handle, sym);
    if (!p) {
      std::fprintf(stderr,
                   "HIP runtime: symbol '%s' not found in libamdhip64.so\n",
                   sym);
      ok = false;
      return;
    }
    static_assert(sizeof(p) == sizeof(fp), "function pointer size mismatch");
    *reinterpret_cast<void **>(&fp) = p;
  };

  loadSym(api.init, "hipInit");
  loadSym(api.setDevice, "hipSetDevice");
  loadSym(api.getDevice, "hipGetDevice");
  loadSym(api.getDeviceCount, "hipGetDeviceCount");
  loadSym(api.getDeviceProperties, "hipGetDevicePropertiesR0600");
  loadSym(api.deviceReset, "hipDeviceReset");
  loadSym(api.moduleLoadData, "hipModuleLoadData");
  loadSym(api.moduleGetFunction, "hipModuleGetFunction");
  loadSym(api.moduleUnload, "hipModuleUnload");
  loadSym(api.moduleLaunchKernel, "hipModuleLaunchKernel");
  loadSym(api.malloc, "hipMalloc");
  loadSym(api.free, "hipFree");
  loadSym(api.memcpy, "hipMemcpy");
  loadSym(api.deviceSynchronize, "hipDeviceSynchronize");
  loadSym(api.peekAtLastError, "hipPeekAtLastError");
  loadSym(api.getLastError, "hipGetLastError");
  loadSym(api.getErrorString, "hipGetErrorString");
  loadSym(api.eventCreate, "hipEventCreate");
  loadSym(api.eventDestroy, "hipEventDestroy");
  loadSym(api.eventRecord, "hipEventRecord");
  loadSym(api.eventSynchronize, "hipEventSynchronize");
  loadSym(api.eventElapsedTime, "hipEventElapsedTime");
  loadSym(api.streamCreate, "hipStreamCreate");
  loadSym(api.streamDestroy, "hipStreamDestroy");
  loadSym(api.streamSynchronize, "hipStreamSynchronize");

  if (!ok) {
    ::dlclose(handle);
    return nullptr;
  }

  api.lib = handle;
  return &api;
}

// Returns the loaded API table, or nullptr if libamdhip64.so was not found or
// any required symbol was missing. Initialised once on first call.
static const HipApi *hip() {
  static const HipApi *api = HipApi::load();
  return api;
}

//===----------------------------------------------------------------------===//
// Error checking helper
//===----------------------------------------------------------------------===//

// Returns the HIP API table, throwing if libamdhip64.so was not loaded.
static const HipApi &requireHip() {
  const HipApi *api = hip();
  if (!api)
    throw std::runtime_error(
        "HIP runtime not available: could not load libamdhip64.so");
  return *api;
}

static void hipCheckImpl(hipError_t err, const char *file, int line) {
  if (err == hipSuccess)
    return;
  throw std::runtime_error(std::string("HIP error at ") + file + ":" +
                           std::to_string(line) + " - " +
                           hip()->getErrorString(err));
}

#define hipCheck(call) hipCheckImpl((call), __FILE__, __LINE__)

//===----------------------------------------------------------------------===//
// Python module
//===----------------------------------------------------------------------===//

NB_MODULE(_runtime_module, m) {
  m.doc() = "Python bindings for HIP runtime module";

  m.def("hip_init", []() {
    const HipApi &h = requireHip();
    static std::once_flag flag;
    std::call_once(flag, [&h]() {
      hipCheck(h.init(0));
      hipCheck(h.setDevice(0));
    });
  });

  // Clear any sticky HIP error from a previous failed call. Without this,
  // a failed hipModuleLoadData leaves a deferred error that the next
  // hipCheck (e.g. on hipMalloc) picks up, cascading failures across
  // configs in the same subprocess pool.
  m.def("hip_peek_at_last_error", []() -> std::string {
    const HipApi &h = requireHip();
    hipError_t err = h.peekAtLastError();
    if (err != hipSuccess)
      return std::string(h.getErrorString(err));
    return "";
  });

  m.def("hip_clear_last_error", []() { (void)requireHip().getLastError(); });

  // Query all device properties needed for occupancy/resource checks.
  // Returns a dict with the hardware constants that target.py hardcodes.
  // Source: clr/rocclr/device/rocm/rocdevice.cpp (lines 1593-1610).
  m.def("hip_get_device_props", [](int device_id) -> nb::dict {
    const HipApi &h = requireHip();
    hipDeviceProp_t props;
    hipCheck(h.getDeviceProperties(&props, device_id));
    nb::dict d;
    d["name"] = std::string(props.name);
    d["gcn_arch_name"] = std::string(props.gcnArchName);
    d["warp_size"] = props.warpSize;
    // LDS per CU (bytes).
    d["lds_per_cu"] = static_cast<int>(props.sharedMemPerMultiprocessor);
    // Register file: regsPerMultiprocessor = vgprsPerSimd * simdPerCU *
    // warpSize e.g. 512 * 4 * 64 = 131072 on gfx942.
    d["regs_per_multiprocessor"] = props.regsPerMultiprocessor;
    // CU count.
    d["multiprocessor_count"] = props.multiProcessorCount;
    // Max threads per block.
    d["max_threads_per_block"] = props.maxThreadsPerBlock;
    // Max threads per multiprocessor (= max waves per CU * warpSize).
    d["max_threads_per_multiprocessor"] = props.maxThreadsPerMultiProcessor;
    return d;
  });

  m.def("hip_module_load_data", [](const nb::bytes &binary) -> void * {
    const HipApi &h = requireHip();
    hipModule_t *mod = new hipModule_t();
    hipCheck(h.moduleLoadData(
        mod,
        const_cast<void *>(reinterpret_cast<const void *>(binary.data()))));
    return mod;
  });

  m.def("hip_module_get_function",
        [](void *module, const nb::bytes &name) -> void * {
          const HipApi &h = requireHip();
          hipFunction_t *f = new hipFunction_t();
          hipModule_t *mod = reinterpret_cast<hipModule_t *>(module);
          hipCheck(h.moduleGetFunction(
              f, *mod, reinterpret_cast<const char *>(name.data())));
          return f;
        });

  m.def("hip_module_unload", [](void *module) {
    const HipApi &h = requireHip();
    if (!module)
      return;
    hipModule_t *mod = reinterpret_cast<hipModule_t *>(module);
    hipCheck(h.moduleUnload(*mod));
    delete mod;
  });

  m.def("hip_function_free", [](void *function) {
    if (!function)
      return;
    hipFunction_t *f = reinterpret_cast<hipFunction_t *>(function);
    delete f;
  });

  m.def(
      "hip_module_launch_kernel",
      [](void *function, size_t gx, size_t gy, size_t gz, size_t bx, size_t by,
         size_t bz, nb::handle kernelParams) {
        const HipApi &h = requireHip();
        hipFunction_t *f = reinterpret_cast<hipFunction_t *>(function);

        void **kernelParamsPtr = nullptr;
        if (!kernelParams.is_none()) {
          PyObject *capsule = kernelParams.ptr();
          if (PyCapsule_CheckExact(capsule)) {
            void *ptr = PyCapsule_GetPointer(capsule, "nb_handle");
            kernelParamsPtr = reinterpret_cast<void **>(ptr);
          }
        }

        hipCheck(h.moduleLaunchKernel(
            *f, static_cast<uint32_t>(gx), static_cast<uint32_t>(gy),
            static_cast<uint32_t>(gz), static_cast<uint32_t>(bx),
            static_cast<uint32_t>(by), static_cast<uint32_t>(bz),
            /*sharedMem=*/0,
            /*stream=*/nullptr, kernelParamsPtr,
            /*extra=*/nullptr));
      },
      nb::arg("function"), nb::arg("gx"), nb::arg("gy"), nb::arg("gz"),
      nb::arg("bx"), nb::arg("by"), nb::arg("bz"),
      nb::arg("kernelParams") = nb::none());

  m.def("hip_device_synchronize", []() {
    const HipApi &h = requireHip();
    hipCheck(h.deviceSynchronize());
  });

  m.def("hip_malloc", [](size_t size) -> void * {
    const HipApi &h = requireHip();
    void *ptr = nullptr;
    hipCheck(h.malloc(&ptr, size));
    return ptr;
  });

  m.def("hip_free", [](void *ptr) {
    const HipApi &h = requireHip();
    if (!ptr)
      return;
    hipCheck(h.free(ptr));
  });

  m.def("hip_memcpy_host_to_device", [](void *dst, void *src, size_t size) {
    const HipApi &h = requireHip();
    hipCheck(h.memcpy(dst, src, size, hipMemcpyHostToDevice));
  });

  m.def("hip_memcpy_device_to_host", [](void *dst, void *src, size_t size) {
    const HipApi &h = requireHip();
    hipCheck(h.memcpy(dst, src, size, hipMemcpyDeviceToHost));
  });

  m.def("hip_memcpy_device_to_device", [](void *dst, void *src, size_t size) {
    const HipApi &h = requireHip();
    hipCheck(h.memcpy(dst, src, size, hipMemcpyDeviceToDevice));
  });

  m.def("hip_get_device_count", []() -> int {
    const HipApi &h = requireHip();
    int count = 0;
    hipCheck(h.getDeviceCount(&count));
    return count;
  });

  m.def("hip_set_device", [](int device_id) {
    const HipApi &h = requireHip();
    hipCheck(h.setDevice(device_id));
  });

  m.def("hip_device_reset", []() {
    const HipApi &h = requireHip();
    hipCheck(h.deviceReset());
  });

  m.def("hip_get_device", []() -> int {
    const HipApi &h = requireHip();
    int device_id = 0;
    hipCheck(h.getDevice(&device_id));
    return device_id;
  });

  m.def("hip_event_create", []() -> void * {
    const HipApi &h = requireHip();
    hipEvent_t *event = new hipEvent_t();
    hipCheck(h.eventCreate(event));
    return event;
  });

  m.def("hip_event_destroy", [](void *event) {
    const HipApi &h = requireHip();
    if (!event)
      return;
    hipEvent_t *e = reinterpret_cast<hipEvent_t *>(event);
    hipCheck(h.eventDestroy(*e));
    delete e;
  });

  m.def("hip_event_record", [](void *event) {
    const HipApi &h = requireHip();
    hipEvent_t *e = reinterpret_cast<hipEvent_t *>(event);
    hipCheck(h.eventRecord(*e, /*stream=*/nullptr));
  });

  m.def("hip_event_synchronize", [](void *event) {
    const HipApi &h = requireHip();
    hipEvent_t *e = reinterpret_cast<hipEvent_t *>(event);
    hipCheck(h.eventSynchronize(*e));
  });

  m.def("hip_event_elapsed_time", [](void *start, void *stop) -> float {
    const HipApi &h = requireHip();
    hipEvent_t *start_event = reinterpret_cast<hipEvent_t *>(start);
    hipEvent_t *stop_event = reinterpret_cast<hipEvent_t *>(stop);
    float ms = 0.0f;
    hipCheck(h.eventElapsedTime(&ms, *start_event, *stop_event));
    return ms;
  });

  m.def("hip_stream_create", []() -> void * {
    const HipApi &h = requireHip();
    hipStream_t *s = new hipStream_t();
    hipCheck(h.streamCreate(s));
    return s;
  });

  m.def("hip_stream_destroy", [](void *stream) {
    const HipApi &h = requireHip();
    if (!stream)
      return;
    hipStream_t *s = reinterpret_cast<hipStream_t *>(stream);
    hipCheck(h.streamDestroy(*s));
    delete s;
  });

  m.def("hip_stream_synchronize", [](void *stream) {
    const HipApi &h = requireHip();
    hipStream_t *s = reinterpret_cast<hipStream_t *>(stream);
    hipCheck(h.streamSynchronize(*s));
  });
}
