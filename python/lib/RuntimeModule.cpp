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
//
//===----------------------------------------------------------------------===//

#include <iostream>

#ifdef HAS_HIP_SUPPORT
#include "hip/hip_runtime.h"
#endif

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

// Macro for HIP error checking
#define hipCheck(call)                                                         \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - "     \
                << hipGetErrorString(err) << std::endl;                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Track HIP initialization state (thread-safe due to static initialization)
static bool &hip_initialized() {
  static bool initialized = false;
  return initialized;
}

NB_MODULE(_runtime_module, m) {
  m.doc() = "Python bindings for HIP runtime module";

#ifdef HAS_HIP_SUPPORT
  m.def("hip_init", []() {
    if (!hip_initialized()) {
      hipCheck(hipInit(0));
      hip_initialized() = true;
      hipCheck(hipSetDevice(0));
    }
  });

  m.def("hip_module_load_data", [](const nb::bytes &binary) -> void * {
    hipModule_t *m = new hipModule_t();
    hipCheck(hipModuleLoadData(
        m, const_cast<void *>(reinterpret_cast<const void *>(binary.data()))));
    return m;
  });

  m.def("hip_module_get_function",
        [](void *module, const nb::bytes &binary) -> void * {
          hipFunction_t *f = new hipFunction_t();
          hipModule_t *m = reinterpret_cast<hipModule_t *>(module);
          auto res = hipModuleGetFunction(
              f, *m, reinterpret_cast<const char *>(binary.data()));
          assert(res == hipError_t::hipSuccess);
          return f;
        });

  m.def("hip_module_unload", [](void *module) {
    if (module != nullptr) {
      hipModule_t *m = reinterpret_cast<hipModule_t *>(module);
      hipCheck(hipModuleUnload(*m));
      delete m;
    }
  });

  m.def("hip_function_free", [](void *function) {
    if (function != nullptr) {
      hipFunction_t *f = reinterpret_cast<hipFunction_t *>(function);
      delete f;
    }
  });

  m.def(
      "hip_module_launch_kernel",
      [](void *function, size_t gx, size_t gy, size_t gz, size_t bx, size_t by,
         size_t bz, nb::handle kernelParams) {
        void **extra = nullptr;
        hipFunction_t *f = reinterpret_cast<hipFunction_t *>(function);

        void **kernelParamsPtr = nullptr;
        if (!kernelParams.is_none()) {
          // Extract pointer from capsule using Python C API
          PyObject *capsule = kernelParams.ptr();
          if (PyCapsule_CheckExact(capsule)) {
            void *ptr = PyCapsule_GetPointer(capsule, "nb_handle");
            kernelParamsPtr = reinterpret_cast<void **>(ptr);
          }
        }

        hipCheck(hipModuleLaunchKernel(*f,         // func
                                       gx, gy, gz, // grid
                                       bx, by, bz, // block
                                       0,
                                       0,               // default stream
                                       kernelParamsPtr, // kernelParams
                                       extra // extra (exclusive w/ kernelParams)
        ));
      },
      nb::arg("function"), nb::arg("gx"), nb::arg("gy"), nb::arg("gz"),
      nb::arg("bx"), nb::arg("by"), nb::arg("bz"),
      nb::arg("kernelParams") = nb::none());

  m.def("hip_device_synchronize", []() {
    hipCheck(hipDeviceSynchronize());
  });

  // HIP memory allocation functions
  m.def("hip_malloc", [](size_t size) -> void * {
    void *ptr = nullptr;
    hipCheck(hipMalloc(&ptr, size));
    return ptr;
  });

  m.def("hip_free", [](void *ptr) {
    if (ptr != nullptr) {
      hipCheck(hipFree(ptr));
    }
  });

  m.def("hip_memcpy_host_to_device", [](void *dst, void *src, size_t size) {
    hipCheck(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
  });

  m.def("hip_memcpy_device_to_host", [](void *dst, void *src, size_t size) {
    hipCheck(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));
  });

  m.def("hip_memcpy_device_to_device", [](void *dst, void *src, size_t size) {
    hipCheck(hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice));
  });

  m.def("hip_get_device_count", []() -> int {
    int count = 0;
    hipCheck(hipGetDeviceCount(&count));
    return count;
  });

  m.def("hip_set_device", [](int device_id) {
    hipCheck(hipSetDevice(device_id));
  });

  m.def("hip_get_device", []() -> int {
    int device_id = 0;
    hipCheck(hipGetDevice(&device_id));
    return device_id;
  });

  // HIP event functions for timing
  m.def("hip_event_create", []() -> void * {
    hipEvent_t *event = new hipEvent_t();
    hipCheck(hipEventCreate(event));
    return event;
  });

  m.def("hip_event_destroy", [](void *event) {
    if (event != nullptr) {
      hipEvent_t *e = reinterpret_cast<hipEvent_t *>(event);
      hipCheck(hipEventDestroy(*e));
      delete e;
    }
  });

  m.def("hip_event_record", [](void *event) {
    hipEvent_t *e = reinterpret_cast<hipEvent_t *>(event);
    hipCheck(hipEventRecord(*e, 0));  // 0 = default stream
  });

  m.def("hip_event_synchronize", [](void *event) {
    hipEvent_t *e = reinterpret_cast<hipEvent_t *>(event);
    hipCheck(hipEventSynchronize(*e));
  });

  m.def("hip_event_elapsed_time", [](void *start, void *stop) -> float {
    hipEvent_t *start_event = reinterpret_cast<hipEvent_t *>(start);
    hipEvent_t *stop_event = reinterpret_cast<hipEvent_t *>(stop);
    float ms = 0.0f;
    hipCheck(hipEventElapsedTime(&ms, *start_event, *stop_event));
    return ms;
  });

#else
  m.def("hip_init", []() {
    printf("Warning: HIP support is not available, using a noop stub\n");
  });

  m.def("hip_module_load_data", [](const nb::bytes &binary) -> void * {
    printf("Warning: HIP support is not available, using a noop stub\n");
    return (void *)0xdeadbeef;
  });

  m.def("hip_module_get_function",
        [](void *module, const nb::bytes &binary) -> void * {
          printf("Warning: HIP support is not available, using a noop stub\n");
          return (void *)0xdeadbeef;
        });

  m.def("hip_module_unload", [](void *module) {
    printf("Warning: HIP support is not available, using a noop stub\n");
  });

  m.def("hip_function_free", [](void *function) {
    printf("Warning: HIP support is not available, using a noop stub\n");
  });

  m.def(
      "hip_module_launch_kernel",
      [](void *function, size_t gx, size_t gy, size_t gz, size_t bx, size_t by,
         size_t bz, void *kernelParams) {
        printf("Warning: HIP support is not available, using a noop stub\n");
      },
      nb::arg("function"), nb::arg("gx"), nb::arg("gy"), nb::arg("gz"),
      nb::arg("bx"), nb::arg("by"), nb::arg("bz"),
      nb::arg("kernelParams") = nullptr);

  m.def("hip_device_synchronize", []() {
    printf("Warning: HIP support is not available, using a noop stub\n");
  });

  // HIP memory allocation stubs
  m.def("hip_malloc", [](size_t size) -> void * {
    printf("Warning: HIP support is not available, using a noop stub\n");
    return (void *)0xdeadbeef;
  });

  m.def("hip_free", [](void *ptr) {
    printf("Warning: HIP support is not available, using a noop stub\n");
  });

  m.def("hip_memcpy_host_to_device", [](void *dst, void *src, size_t size) {
    printf("Warning: HIP support is not available, using a noop stub\n");
  });

  m.def("hip_memcpy_device_to_host", [](void *dst, void *src, size_t size) {
    printf("Warning: HIP support is not available, using a noop stub\n");
  });

  m.def("hip_memcpy_device_to_device", [](void *dst, void *src, size_t size) {
    printf("Warning: HIP support is not available, using a noop stub\n");
  });

  m.def("hip_get_device_count", []() -> int {
    printf("Warning: HIP support is not available, using a noop stub\n");
    return 1;
  });

  m.def("hip_set_device", [](int device_id) {
    printf("Warning: HIP support is not available, using a noop stub\n");
  });

  m.def("hip_get_device", []() -> int {
    printf("Warning: HIP support is not available, using a noop stub\n");
    return 0;
  });

  m.def("hip_event_create", []() -> void * {
    printf("Warning: HIP support is not available, using a noop stub\n");
    return (void *)0xdeadbeef;
  });

  m.def("hip_event_destroy", [](void *event) {
    printf("Warning: HIP support is not available, using a noop stub\n");
  });

  m.def("hip_event_record", [](void *event) {
    printf("Warning: HIP support is not available, using a noop stub\n");
  });

  m.def("hip_event_synchronize", [](void *event) {
    printf("Warning: HIP support is not available, using a noop stub\n");
  });

  m.def("hip_event_elapsed_time", [](void *start, void *stop) -> float {
    printf("Warning: HIP support is not available, using a noop stub\n");
    return 0.0f;
  });
#endif // HAS_HIP_SUPPORT
}
