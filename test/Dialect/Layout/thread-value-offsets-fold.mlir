// RUN: aster-opt %s --lower-layout-to-affine | FileCheck %s

// CHECK-LABEL: func.func @fold_single_transfer
// CHECK-SAME:  (%[[TID:.*]]: index)
func.func @fold_single_transfer(%tid: index) -> index {
  // CHECK:     %[[D:.*]]:2 = affine.delinearize_index %[[TID]] into (16, 4)
  // CHECK:     %[[A:.+]] = affine.linearize_index_by_strides[%[[D]]#0, %[[D]]#1] by (128, 16) : index
  // CHECK-NOT: layout.thread_value_offsets
  // CHECK-NOT: layout.apply
  // CHECK:     return %[[A]]
  %off:1 = layout.thread_value_offsets[%tid]
    thread_layout = #layout.strided_layout<[16, 4] : [128, 16]>
    value_layout = #layout.strided_layout<[1] : [0]> -> index
  return %off#0 : index
}

// CHECK-LABEL: func.func @fold_2d_thread_1d_value
// CHECK-SAME:  (%[[TID:.*]]: index)
func.func @fold_2d_thread_1d_value(%tid: index) -> (index, index) {
  // CHECK:     %[[D:.*]]:2 = affine.delinearize_index %[[TID]] into (16, 4)
  // CHECK:     %[[A:.+]] = affine.linearize_index_by_strides[%[[D]]#0, %[[D]]#1] by (128, 16) : index
  // CHECK:     %[[R1:.+]] = affine.apply #{{.+}}(%[[A]])
  // CHECK-NOT: layout.thread_value_offsets
  // CHECK:     return %[[A]], %[[R1]]
  %offs:2 = layout.thread_value_offsets[%tid]
    thread_layout = #layout.strided_layout<[16, 4] : [128, 16]>
    value_layout = #layout.strided_layout<[2] : [4]> -> index, index
  return %offs#0, %offs#1 : index, index
}

// CHECK-LABEL: func.func @fold_3d_thread_2d_value
// CHECK-SAME:  (%[[TID:.*]]: index)
func.func @fold_3d_thread_2d_value(%tid: index) -> (index, index, index, index) {
  // CHECK:         %[[D:.*]]:3 = affine.delinearize_index %[[TID]] into (2, 4, 8)
  // CHECK:         %[[A:.+]] = affine.linearize_index_by_strides[%[[D]]#0, %[[D]]#1, %[[D]]#2] by (256, 64, 8) : index
  // CHECK-COUNT-3: affine.apply #{{.+}}(%[[A]])
  // CHECK:         return
  %offs:4 = layout.thread_value_offsets[%tid]
    thread_layout = #layout.strided_layout<[2, 4, 8] : [256, 64, 8]>
    value_layout = #layout.strided_layout<[2, 2] : [4, 1]> -> index, index, index, index
  return %offs#0, %offs#1, %offs#2, %offs#3 : index, index, index, index
}

// CHECK-LABEL: func.func @fold_2d_thread_3d_value
// CHECK-SAME:  (%[[TID:.*]]: index)
func.func @fold_2d_thread_3d_value(%tid: index)
    -> (index, index, index, index, index, index, index, index) {
  // CHECK:         %[[D:.*]]:2 = affine.delinearize_index %[[TID]] into (4, 4)
  // CHECK:         %[[A:.+]] = affine.linearize_index_by_strides[%[[D]]#0, %[[D]]#1] by (64, 8) : index
  // CHECK-COUNT-7: affine.apply #{{.+}}(%[[A]])
  // CHECK:         return
  %offs:8 = layout.thread_value_offsets[%tid]
    thread_layout = #layout.strided_layout<[4, 4] : [64, 8]>
    value_layout = #layout.strided_layout<[2, 2, 2] : [4, 2, 1]>
        -> index, index, index, index, index, index, index, index
  return %offs#0, %offs#1, %offs#2, %offs#3, %offs#4, %offs#5, %offs#6, %offs#7
      : index, index, index, index, index, index, index, index
}

// CHECK-LABEL: func.func @fold_nested_thread
// CHECK-SAME:  (%[[TID:.*]]: index)
func.func @fold_nested_thread(%tid: index) -> (index, index) {
  // CHECK:     %[[D0:.*]]:2 = affine.delinearize_index %[[TID]] into (4, 4)
  // CHECK:     %[[D1:.*]]:2 = affine.delinearize_index %[[D0]]#0 into (2, 2)
  // CHECK:     %[[A:.+]] = affine.linearize_index_by_strides[%[[D1]]#0, %[[D1]]#1, %[[D0]]#1] by (512, 128, 8) : index
  // CHECK:     %[[R1:.+]] = affine.apply #{{.+}}(%[[A]])
  %offs:2 = layout.thread_value_offsets[%tid]
    thread_layout = #layout.strided_layout<[(2, 2), 4] : [(512, 128), 8]>
    value_layout = #layout.strided_layout<[2] : [4]> -> index, index
  return %offs#0, %offs#1 : index, index
}
