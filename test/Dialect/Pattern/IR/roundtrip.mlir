// RUN: aster-opt %s --verify-roundtrip

pattern.rewrite_pattern @with_fields benefit(1) op(@SomeOp) fields {
  %f0 = pattern.field @counter : i32
  %f1 = pattern.field @analysis : !emitc.opaque<"MyAnalysis">
  pattern.yield %f0, %f1 : i32, !emitc.opaque<"MyAnalysis">
} body {
  %v = pattern.get_field @counter : i32
  %cond = arith.constant true
  pattern.action %cond {
    pattern.yield
  }
}

pattern.rewrite_pattern @no_fields benefit(42) op(@AnotherOp) body {
  %cond = arith.constant true
  pattern.action %cond {
    pattern.yield
  }
}

pattern.rewrite_pattern @action_with_body benefit(5) op(@TestOp) body {
  %cond = arith.constant true
  pattern.action %cond {
    emitc.call_opaque "doSomething"() : () -> ()
    pattern.yield
  }
}
