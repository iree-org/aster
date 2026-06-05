/* Copyright 2026 The ASTER Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * No-op amd_comgr stub. Every entry point returns AMD_COMGR_STATUS_ERROR so
 * StinkyTofu's tryAssembleWithComgr() short-circuits to false on platforms with
 * no real ROCm. The .stir parse/print round-trip never assembles, so this is
 * the correct "no assembler available" behavior.
 */
#include "amd_comgr/amd_comgr.h"

amd_comgr_status_t amd_comgr_create_data(amd_comgr_data_kind_t k,
                                         amd_comgr_data_t *d) {
  (void)k;
  (void)d;
  return AMD_COMGR_STATUS_ERROR;
}
amd_comgr_status_t amd_comgr_release_data(amd_comgr_data_t d) {
  (void)d;
  return AMD_COMGR_STATUS_ERROR;
}
amd_comgr_status_t amd_comgr_set_data(amd_comgr_data_t d, size_t n,
                                      const char *p) {
  (void)d;
  (void)n;
  (void)p;
  return AMD_COMGR_STATUS_ERROR;
}
amd_comgr_status_t amd_comgr_set_data_name(amd_comgr_data_t d, const char *n) {
  (void)d;
  (void)n;
  return AMD_COMGR_STATUS_ERROR;
}
amd_comgr_status_t amd_comgr_create_data_set(amd_comgr_data_set_t *s) {
  (void)s;
  return AMD_COMGR_STATUS_ERROR;
}
amd_comgr_status_t amd_comgr_destroy_data_set(amd_comgr_data_set_t s) {
  (void)s;
  return AMD_COMGR_STATUS_ERROR;
}
amd_comgr_status_t amd_comgr_data_set_add(amd_comgr_data_set_t s,
                                          amd_comgr_data_t d) {
  (void)s;
  (void)d;
  return AMD_COMGR_STATUS_ERROR;
}
amd_comgr_status_t amd_comgr_create_action_info(amd_comgr_action_info_t *a) {
  (void)a;
  return AMD_COMGR_STATUS_ERROR;
}
amd_comgr_status_t amd_comgr_destroy_action_info(amd_comgr_action_info_t a) {
  (void)a;
  return AMD_COMGR_STATUS_ERROR;
}
amd_comgr_status_t amd_comgr_action_info_set_language(amd_comgr_action_info_t a,
                                                      amd_comgr_language_t l) {
  (void)a;
  (void)l;
  return AMD_COMGR_STATUS_ERROR;
}
amd_comgr_status_t amd_comgr_action_info_set_isa_name(amd_comgr_action_info_t a,
                                                      const char *n) {
  (void)a;
  (void)n;
  return AMD_COMGR_STATUS_ERROR;
}
amd_comgr_status_t
amd_comgr_action_info_set_option_list(amd_comgr_action_info_t a,
                                      const char *const *opts, size_t n) {
  (void)a;
  (void)opts;
  (void)n;
  return AMD_COMGR_STATUS_ERROR;
}
amd_comgr_status_t amd_comgr_do_action(amd_comgr_action_kind_t kind,
                                       amd_comgr_action_info_t a,
                                       amd_comgr_data_set_t in,
                                       amd_comgr_data_set_t out) {
  (void)kind;
  (void)a;
  (void)in;
  (void)out;
  return AMD_COMGR_STATUS_ERROR;
}
