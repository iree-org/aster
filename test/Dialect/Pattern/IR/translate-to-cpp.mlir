// RUN: aster-translate -mlir-to-cpp %s | FileCheck %s

// CHECK:      struct simple_pat : OpRewrite<MyOp> {
// CHECK-NEXT:   int32_t counter;
// CHECK-NEXT:   MyAnalysis analysis;
// CHECK-NEXT:   LogicalResult matchRewrite(MyOp op, PatternRewriter &rewriter) {
// CHECK-NEXT:     if (!true)
// CHECK-NEXT:       return failure();
// CHECK:        }
// CHECK-NEXT: };
pattern.rewrite_pattern @simple_pat benefit(1) op(@MyOp) fields {
  %f0 = pattern.field @counter : i32
  %f1 = pattern.field @analysis : !emitc.opaque<"MyAnalysis">
  pattern.yield %f0, %f1 : i32, !emitc.opaque<"MyAnalysis">
} body {
  %cond = emitc.literal "true" : i1
  pattern.action %cond {
    pattern.yield
  }
}

// CHECK:      struct no_fields_pat : OpRewrite<AnotherOp> {
// CHECK-NEXT:   LogicalResult matchRewrite(AnotherOp op, PatternRewriter &rewriter) {
// CHECK-NEXT:     if (!true)
// CHECK-NEXT:       return failure();
// CHECK:        }
// CHECK-NEXT: };
pattern.rewrite_pattern @no_fields_pat benefit(42) op(@AnotherOp) body {
  %cond = emitc.literal "true" : i1
  pattern.action %cond {
    pattern.yield
  }
}

// CHECK:      struct get_field_pat : OpRewrite<TestOp> {
// CHECK-NEXT:   int32_t counter;
// CHECK-NEXT:   LogicalResult matchRewrite(TestOp op, PatternRewriter &rewriter) {
// CHECK-NEXT:     int32_t v1 = counter;
// CHECK-NEXT:     if (!true)
// CHECK-NEXT:       return failure();
// CHECK:        }
// CHECK-NEXT: };
pattern.rewrite_pattern @get_field_pat benefit(3) op(@TestOp) fields {
  %f0 = pattern.field @counter : i32
  pattern.yield %f0 : i32
} body {
  %v = pattern.get_field @counter : i32
  %cond = emitc.literal "true" : i1
  pattern.action %cond {
    pattern.yield
  }
}

// CHECK:      struct action_body_pat : OpRewrite<FooOp> {
// CHECK-NEXT:   LogicalResult matchRewrite(FooOp op, PatternRewriter &rewriter) {
// CHECK-NEXT:     if (!true)
// CHECK-NEXT:       return failure();
// CHECK-NEXT:     doSomething();
// CHECK:        }
// CHECK-NEXT: };
pattern.rewrite_pattern @action_body_pat benefit(1) op(@FooOp) body {
  %cond = emitc.literal "true" : i1
  pattern.action %cond {
    emitc.call_opaque "doSomething"() : () -> ()
    pattern.yield
  }
}

// CHECK:      struct method_dot_pat : OpRewrite<BarOp> {
// CHECK-NEXT:   LogicalResult matchRewrite(BarOp op, PatternRewriter &rewriter) {
// CHECK-NEXT:     rewriter.eraseOp(op);
// CHECK-NEXT:     if (!true)
// CHECK-NEXT:       return failure();
// CHECK:        }
// CHECK-NEXT: };
pattern.rewrite_pattern @method_dot_pat benefit(1) op(@BarOp) body {
  %rw = emitc.literal "rewriter" : !emitc.opaque<"PatternRewriter">
  %op = emitc.literal "op" : !emitc.opaque<"BarOp">
  pattern.method_call @eraseOp %rw(%op) : (!emitc.opaque<"PatternRewriter">, !emitc.opaque<"BarOp">) -> ()
  %cond = emitc.literal "true" : i1
  pattern.action %cond {
    pattern.yield
  }
}

// CHECK:      struct method_arrow_pat : OpRewrite<BazOp> {
// CHECK-NEXT:   LogicalResult matchRewrite(BazOp op, PatternRewriter &rewriter) {
// CHECK-NEXT:     MLIRContext v2 = ptr->getContext();
// CHECK-NEXT:     if (!true)
// CHECK-NEXT:       return failure();
// CHECK:        }
// CHECK-NEXT: };
pattern.rewrite_pattern @method_arrow_pat benefit(1) op(@BazOp) body {
  %ptr = emitc.literal "ptr" : !emitc.ptr<!emitc.opaque<"Builder">>
  %r = pattern.method_call @getContext %ptr() : (!emitc.ptr<!emitc.opaque<"Builder">>) -> !emitc.opaque<"MLIRContext">
  %cond = emitc.literal "true" : i1
  pattern.action %cond {
    pattern.yield
  }
}

// CHECK:      struct method_args_pat : OpRewrite<QuxOp> {
// CHECK-NEXT:   LogicalResult matchRewrite(QuxOp op, PatternRewriter &rewriter) {
// CHECK-NEXT:     int32_t v3 = obj.compute(a, b);
// CHECK-NEXT:     if (!true)
// CHECK-NEXT:       return failure();
// CHECK:        }
// CHECK-NEXT: };
pattern.rewrite_pattern @method_args_pat benefit(1) op(@QuxOp) body {
  %obj = emitc.literal "obj" : !emitc.opaque<"MyClass">
  %a = emitc.literal "a" : i32
  %b = emitc.literal "b" : i32
  %r = pattern.method_call @compute %obj(%a, %b) : (!emitc.opaque<"MyClass">, i32, i32) -> i32
  %cond = emitc.literal "true" : i1
  pattern.action %cond {
    pattern.yield
  }
}
