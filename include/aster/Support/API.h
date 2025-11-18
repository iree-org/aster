//===- API.h - API helpers --------------------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_SUPPORT_API_H
#define ASTER_SUPPORT_API_H

//===----------------------------------------------------------------------===//
// Visibility annotations.
// Use ASTER_EXPORTED for exported functions.
//
// On Windows, if ASTER_ENABLE_WINDOWS_DLL_DECLSPEC is defined, then
// __declspec(dllexport) and __declspec(dllimport) will be generated. This
// can only be enabled if actually building DLLs. It is generally, mutually
// exclusive with the use of other mechanisms for managing imports/exports
// (i.e. CMake's WINDOWS_EXPORT_ALL_SYMBOLS feature).
//===----------------------------------------------------------------------===//

#if (defined(_WIN32) || defined(__CYGWIN__)) &&                                \
    !defined(ASTER_ENABLE_WINDOWS_DLL_DECLSPEC)
// Visibility annotations disabled.
#define ASTER_EXPORTED
#elif defined(_WIN32) || defined(__CYGWIN__)
// Windows visibility declarations.
#if ASTER_BUILDING_LIBRARY
#define ASTER_EXPORTED __declspec(dllexport)
#else
#define ASTER_EXPORTED __declspec(dllimport)
#endif
#else
// Non-windows: use visibility attributes.
#define ASTER_EXPORTED __attribute__((visibility("default")))
#endif
#endif // ASTER_SUPPORT_API_H
