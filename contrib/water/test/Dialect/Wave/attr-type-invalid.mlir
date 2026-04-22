// RUN: water-opt %s --allow-unregistered-dialect --water-test-wave-dialect-functions --split-input-file --verify-diagnostics

// expected-error @below {{expected element type to be integer, index or floating point scalar}}
func.func private @unspecified_tensor() -> !wave.tensor<any of !wave.tensor<any of bf16>>

// -----

// expected-error @below {{shape not expected for non-fully specified tensors}}
"wave_test.create_tensor"() {fully_specified = false, shape = [@A, @B]} : () -> ()

// -----

// expected-error @below {{"wave.hyperparameters" expects a WaveHyperparameterAttr}}
module attributes {wave.hyperparameters = 1} {}

// -----

// expected-error @below {{"wave.elements_per_thread" expects an IntegerAttr}}
module attributes {wave.elements_per_thread = "abc"} {}

// -----

// expected-error @below {{unexpected wave dialect attribute "wave.unexpected"}}
module attributes {wave.unexpected = 42} {}

// -----

// expected-error @below {{symbols names starting with '_' are reserved for internal use}}
module attributes {wave_test.symbol = #wave.symbol<"_A">}

// -----

// expected-error @below {{dimension name 'A' is used more than once}}
func.func private @duplicate_dim_name() attributes { wave.test_index = #wave.expr_list<[](A, A) -> (A)>}

// -----

// expected-error @below {{dimension name 'A' is already used as a symbol name}}
func.func private @duplicate_dim_sym_name() attributes { wave_test.index = #wave.expr_list<[#wave.symbol<"A">](A) -> (A)>}

// -----

// expected-error @below {{dimension name '_A' is reserved for internal use}}
func.func private @reserved_dim_name() attributes { wave_test.index = #wave.expr_list<[](_A) -> (_A)>}

// -----

// expected-error @below {{duplicate symbol #wave.symbol<"A"> in shape}}
"wave_test.create_tensor"() {fully_specified = true, shape = [@A, @B, @A]} : () -> ()
