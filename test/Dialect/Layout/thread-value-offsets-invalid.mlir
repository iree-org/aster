// RUN: aster-opt %s -split-input-file -verify-diagnostics

func.func @aliasing_thread_zero_stride(%tid: index) -> index {
  // expected-error @+1 {{thread_layout has a zero stride on a mode of size 4}}
  %offs:1 = layout.thread_value_offsets[%tid]
    thread_layout = #layout.strided_layout<[4, 8] : [0, 8]>
    value_layout = #layout.strided_layout<[1] : [0]> -> index
  return %offs#0 : index
}

// -----

func.func @aliasing_value_zero_stride(%tid: index) -> (index, index, index, index) {
  // expected-error @+1 {{value_layout has a zero stride on a mode of size 4}}
  %offs:4 = layout.thread_value_offsets[%tid]
    thread_layout = #layout.strided_layout<[2, 2] : [16, 8]>
    value_layout = #layout.strided_layout<[4] : [0]> -> index, index, index, index
  return %offs#0, %offs#1, %offs#2, %offs#3 : index, index, index, index
}

// -----

func.func @aliasing_tv_same_stride(%tid: index) -> index {
  // expected-error @+1 {{thread_layout o value_layout is not injective: cumulative max offset 1 through modes with stride <= 1 overlaps the next mode at stride 1}}
  %offs:1 = layout.thread_value_offsets[%tid]
    thread_layout = #layout.strided_layout<[2, 2] : [1, 1]>
    value_layout = #layout.strided_layout<[1] : [0]> -> index
  return %offs#0 : index
}

// -----

func.func @aliasing_value_overlaps_thread(%tid: index) -> (index, index, index, index) {
  // expected-error @+1 {{thread_layout o value_layout is not injective: cumulative max offset 3 through modes with stride <= 1 overlaps the next mode at stride 3}}
  %offs:4 = layout.thread_value_offsets[%tid]
    thread_layout = #layout.strided_layout<[4] : [1]>
    value_layout = #layout.strided_layout<[4] : [3]> -> index, index, index, index
  return %offs#0, %offs#1, %offs#2, %offs#3 : index, index, index, index
}

// -----

func.func @result_count_mismatch(%tid: index) -> (index, index) {
  // expected-error @+1 {{expected 1 results (= value_layout.size()), got 2}}
  %offs:2 = layout.thread_value_offsets[%tid]
    thread_layout = #layout.strided_layout<[16, 4] : [128, 16]>
    value_layout = #layout.strided_layout<[1] : [0]> -> index, index
  return %offs#0, %offs#1 : index, index
}
