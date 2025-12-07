# Bazel Build

## Prerequisites

```bash
# Install bazelisk (recommended)
brew install bazelisk
```

## Step 1: Initialize submodule

```bash
git submodule update --init --recursive toolchains_llvm_bootstrapped
```

## Step 2: Smoke check LLVM and MLIR builds

```bash
bazel build @llvm-project//llvm:llc
bazel build @llvm-project//mlir:mlir-opt
```

## Step 3: Cross-compile for Linux amd64

```bash
bazel build @llvm-project//llvm:llc --config=linux_amd64
bazel build @llvm-project//mlir:mlir-opt --config=linux_amd64
```

## Next: Query available platforms

```bash
bazel query 'kind(platform, @toolchains_llvm_bootstrapped//platforms/...)'
```
