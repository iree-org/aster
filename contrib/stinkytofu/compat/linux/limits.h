/* Copyright 2026 The ASTER Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * macOS shim for <linux/limits.h>, which StinkyTofu's IntrinsicRegistry.cpp
 * includes for PATH_MAX. On macOS PATH_MAX lives in <limits.h>. This dir is put
 * ahead of the system search path only for the StinkyTofu targets.
 */
#pragma once
#include <limits.h>
