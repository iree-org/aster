// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runtime shim for contrib/cpu AMX kernels.
// See: https://www.kernel.org/doc/html/latest/arch/x86/xstate.html
//
// Without this call on Linux >= 5.17, the first AMX instruction the
// thread executes traps with SIGILL.

#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define ARCH_XCOMP_TILEDATA 18

__attribute__((constructor)) static void amx_runtime_init(void) {
  syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, ARCH_XCOMP_TILEDATA);
}
