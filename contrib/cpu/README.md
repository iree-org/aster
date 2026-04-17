# contrib/cpu

Minimal x86_64 AMX CPU backend for ASTER.

## Build

```bash
ASTER_ENABLE_CPU=1 bash tools/setup.sh
```

If you already have a shared LLVM with x86 support, skip the LLVM rebuild:

```bash
(
  cd build && \
  cmake . -DASTER_ENABLE_CPU=ON && \
  ninja install
)
```

## Run tests

```bash
lit build/contrib/cpu/test && pytest contrib/cpu
```
