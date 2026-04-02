//===- hip.h -------------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HIP type forward declarations and dynamic-loader API table.
//
// Adapted from ROCm 7 HIP headers.
// ROCm HIP attribution:
// ```
// Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// ```
//===----------------------------------------------------------------------===//

#ifndef ASTER_DYN_RUNTIME_HIP_H
#define ASTER_DYN_RUNTIME_HIP_H

#include <cstddef>
#include <cstdint>

//===----------------------------------------------------------------------===//
// HIP type forward declarations
//===----------------------------------------------------------------------===//

// Architectural feature bitfield
struct hipDeviceArch_t {
  unsigned hasGlobalInt32Atomics : 1;
  unsigned hasGlobalFloatAtomicExch : 1;
  unsigned hasSharedInt32Atomics : 1;
  unsigned hasSharedFloatAtomicExch : 1;
  unsigned hasFloatAtomicAdd : 1;
  unsigned hasGlobalInt64Atomics : 1;
  unsigned hasSharedInt64Atomics : 1;
  unsigned hasDoubles : 1;
  unsigned hasWarpVote : 1;
  unsigned hasWarpBallot : 1;
  unsigned hasWarpShuffle : 1;
  unsigned hasFunnelShift : 1;
  unsigned hasThreadFenceSystem : 1;
  unsigned hasSyncThreadsExt : 1;
  unsigned hasSurfaceFuncs : 1;
  unsigned has3dGrid : 1;
  unsigned hasDynamicParallelism : 1;
};

// UUID
struct hipUUID {
  char bytes[16];
};

// Device properties
struct hipDeviceProp_tR0600 {
  char name[256];
  hipUUID uuid;
  char luid[8];
  unsigned int luidDeviceNodeMask;
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  size_t texturePitchAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maxTexture1D;
  int maxTexture1DMipmap;
  int maxTexture1DLinear;
  int maxTexture2D[2];
  int maxTexture2DMipmap[2];
  int maxTexture2DLinear[3];
  int maxTexture2DGather[2];
  int maxTexture3D[3];
  int maxTexture3DAlt[3];
  int maxTextureCubemap;
  int maxTexture1DLayered[2];
  int maxTexture2DLayered[3];
  int maxTextureCubemapLayered[2];
  int maxSurface1D;
  int maxSurface2D[2];
  int maxSurface3D[3];
  int maxSurface1DLayered[2];
  int maxSurface2DLayered[3];
  int maxSurfaceCubemap;
  int maxSurfaceCubemapLayered[2];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  int tccDriver;
  int asyncEngineCount;
  int unifiedAddressing;
  int memoryClockRate;
  int memoryBusWidth;
  int l2CacheSize;
  int persistingL2CacheMaxSize;
  int maxThreadsPerMultiProcessor;
  int streamPrioritiesSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  size_t sharedMemPerMultiprocessor;
  int regsPerMultiprocessor;
  int managedMemory;
  int isMultiGpuBoard;
  int multiGpuBoardGroupID;
  int hostNativeAtomicSupported;
  int singleToDoublePrecisionPerfRatio;
  int pageableMemoryAccess;
  int concurrentManagedAccess;
  int computePreemptionSupported;
  int canUseHostPointerForRegisteredMem;
  int cooperativeLaunch;
  int cooperativeMultiDeviceLaunch;
  size_t sharedMemPerBlockOptin;
  int pageableMemoryAccessUsesHostPageTables;
  int directManagedMemAccessFromHost;
  int maxBlocksPerMultiProcessor;
  int accessPolicyMaxWindowSize;
  size_t reservedSharedMemPerBlock;
  int hostRegisterSupported;
  int sparseHipArraySupported;
  int hostRegisterReadOnlySupported;
  int timelineSemaphoreInteropSupported;
  int memoryPoolsSupported;
  int gpuDirectRDMASupported;
  unsigned int gpuDirectRDMAFlushWritesOptions;
  int gpuDirectRDMAWritesOrdering;
  unsigned int memoryPoolSupportedHandleTypes;
  int deferredMappingHipArraySupported;
  int ipcEventSupported;
  int clusterLaunch;
  int unifiedFunctionPointers;
  int reserved[63];
  int hipReserved[32];
  // HIP-only fields.
  char gcnArchName[256];
  size_t maxSharedMemoryPerMultiProcessor;
  int clockInstructionRate;
  hipDeviceArch_t arch;
  unsigned int *hdpMemFlushCntl;
  unsigned int *hdpRegFlushCntl;
  int cooperativeMultiDeviceUnmatchedFunc;
  int cooperativeMultiDeviceUnmatchedGridDim;
  int cooperativeMultiDeviceUnmatchedBlockDim;
  int cooperativeMultiDeviceUnmatchedSharedMem;
  int isLargeBar;
  int asicRevision;
};
using hipDeviceProp_t = hipDeviceProp_tR0600;

// Opaque GPU object handles.
struct ihipModule_t;
struct ihipFunction_t;
struct ihipStream_t;
struct ihipEvent_t;
using hipModule_t = ihipModule_t *;
using hipFunction_t = ihipFunction_t *;
using hipStream_t = ihipStream_t *;
using hipEvent_t = ihipEvent_t *;

// Error code (only hipSuccess is used for comparisons; all other values are
// returned by the library and forwarded to GetErrorString).
enum hipError_t : int { hipSuccess = 0 };

enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4,
};

//===----------------------------------------------------------------------===//
// HIP API function pointer table
//===----------------------------------------------------------------------===//

struct HipApi {
  hipError_t (*init)(unsigned int flags);
  hipError_t (*setDevice)(int deviceId);
  hipError_t (*getDevice)(int *deviceId);
  hipError_t (*getDeviceCount)(int *count);
  // Looked up as "hipGetDevicePropertiesR0600": the header macro
  // `#define hipGetDeviceProperties hipGetDevicePropertiesR0600` means compiled
  // code always resolves to this versioned symbol.
  // dlsym("hipGetDeviceProperties") returns the ROCm 4.2-era stub which fills
  // the old R0000 struct (no gcnArchName).
  hipError_t (*getDeviceProperties)(hipDeviceProp_t *prop, int deviceId);
  hipError_t (*deviceReset)();
  hipError_t (*moduleLoadData)(hipModule_t *module, const void *image);
  hipError_t (*moduleGetFunction)(hipFunction_t *function, hipModule_t module,
                                  const char *kname);
  hipError_t (*moduleUnload)(hipModule_t module);
  hipError_t (*moduleLaunchKernel)(hipFunction_t f, uint32_t gridX,
                                   uint32_t gridY, uint32_t gridZ,
                                   uint32_t blockX, uint32_t blockY,
                                   uint32_t blockZ, uint32_t sharedMem,
                                   hipStream_t stream, void **kernelParams,
                                   void **extra);
  hipError_t (*malloc)(void **ptr, size_t size);
  hipError_t (*free)(void *ptr);
  hipError_t (*memcpy)(void *dst, const void *src, size_t bytes,
                       hipMemcpyKind kind);
  hipError_t (*deviceSynchronize)();
  hipError_t (*peekAtLastError)();
  hipError_t (*getLastError)();
  const char *(*getErrorString)(hipError_t err);
  hipError_t (*eventCreate)(hipEvent_t *event);
  hipError_t (*eventDestroy)(hipEvent_t event);
  hipError_t (*eventRecord)(hipEvent_t event, hipStream_t stream);
  hipError_t (*eventSynchronize)(hipEvent_t event);
  hipError_t (*eventElapsedTime)(float *ms, hipEvent_t start, hipEvent_t stop);
  hipError_t (*streamCreate)(hipStream_t *stream);
  hipError_t (*streamDestroy)(hipStream_t stream);
  hipError_t (*streamSynchronize)(hipStream_t stream);

  // Library handle, closed on destruction.
  void *lib = nullptr;

  ~HipApi();

  // Opens libamdhip64.so, resolves all required symbols, and returns a pointer
  // to a static HipApi instance. Returns nullptr if the library is absent or
  // any symbol is missing (each missing symbol is reported to stderr).
  static const HipApi *load();
};

#endif // ASTER_DYN_RUNTIME_HIP_H
