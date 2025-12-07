# Bazel Build

## Prerequisites

```bash
brew install bazelisk
```

## Initialize

```bash
git submodule update --init --recursive toolchains_llvm_bootstrapped
```

## Smoke check that LLVM targets build

```bash
bazel build @llvm-project//llvm:llc
bazel build @llvm-project//mlir:mlir-opt
```

and cross-compile:

```bash
bazel build @llvm-project//llvm:llc --config=linux_amd64
bazel build @llvm-project//mlir:mlir-opt --config=linux_amd64
```

## Build Aster tools

```bash
# Build all tools
bazel build //:aster-tools
```

and cross-compile:

```bash
bazel build //:aster-tools --config=linux_amd64
```

## Available targets

| Target | Description |
|--------|-------------|
| `//:aster-tools` | Build all Aster tools (aster-opt and aster-translate) |
| `//:aster-opt` | MLIR optimizer driver |
| `//:aster-translate` | Translation tool |
| `//:amdgcn-tblgen` | Custom tablegen for AMDGCN |


## Cheatsheet to keep depencies trimmed

```
# What are all our dependencies?
bazel query 'kind("cc_library", deps(//:aster-tools, 100))'
bazel query 'kind("cc_library", deps(//:aster-tools, 100))' --output=graph
bazel query 'somepath(//:aster-opt, @llvm-project//mlir:SparseTensorDialect)' 2>/dev/null
bazel query 'allpaths(//:aster-tools, //:AMDGCNDialect)' --output=graph | dot -Tpng -o deps.png
```
