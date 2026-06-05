/* Copyright 2026 The ASTER Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Minimal stub of amd_comgr for no-ROCm builds (macOS). StinkyTofu compiles
 * src/hardware/ComgrProbe.cpp with STINKYTOFU_HAS_COMGR defined, so the real
 * header is unavailable here; this declares only the symbols ComgrProbe.cpp
 * uses. Every function returns AMD_COMGR_STATUS_ERROR, so the assembler probe
 * fails cleanly and the .stir round-trip proceeds without an assembler.
 */
#pragma once
#include <stddef.h>

typedef enum {
  AMD_COMGR_STATUS_SUCCESS = 0,
  AMD_COMGR_STATUS_ERROR = 1
} amd_comgr_status_t;

typedef enum { AMD_COMGR_DATA_KIND_SOURCE = 1 } amd_comgr_data_kind_t;

typedef enum { AMD_COMGR_LANGUAGE_NONE = 0 } amd_comgr_language_t;

typedef enum {
  AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE = 1
} amd_comgr_action_kind_t;

typedef struct {
  void *handle;
} amd_comgr_data_t;
typedef struct {
  void *handle;
} amd_comgr_data_set_t;
typedef struct {
  void *handle;
} amd_comgr_action_info_t;

#ifdef __cplusplus
extern "C" {
#endif

amd_comgr_status_t amd_comgr_create_data(amd_comgr_data_kind_t,
                                         amd_comgr_data_t *);
amd_comgr_status_t amd_comgr_release_data(amd_comgr_data_t);
amd_comgr_status_t amd_comgr_set_data(amd_comgr_data_t, size_t, const char *);
amd_comgr_status_t amd_comgr_set_data_name(amd_comgr_data_t, const char *);
amd_comgr_status_t amd_comgr_create_data_set(amd_comgr_data_set_t *);
amd_comgr_status_t amd_comgr_destroy_data_set(amd_comgr_data_set_t);
amd_comgr_status_t amd_comgr_data_set_add(amd_comgr_data_set_t,
                                          amd_comgr_data_t);
amd_comgr_status_t amd_comgr_create_action_info(amd_comgr_action_info_t *);
amd_comgr_status_t amd_comgr_destroy_action_info(amd_comgr_action_info_t);
amd_comgr_status_t amd_comgr_action_info_set_language(amd_comgr_action_info_t,
                                                      amd_comgr_language_t);
amd_comgr_status_t amd_comgr_action_info_set_isa_name(amd_comgr_action_info_t,
                                                      const char *);
amd_comgr_status_t
amd_comgr_action_info_set_option_list(amd_comgr_action_info_t,
                                      const char *const *, size_t);
amd_comgr_status_t amd_comgr_do_action(amd_comgr_action_kind_t,
                                       amd_comgr_action_info_t,
                                       amd_comgr_data_set_t,
                                       amd_comgr_data_set_t);

#ifdef __cplusplus
}
#endif
