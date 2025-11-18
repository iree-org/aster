//===- CompileAsm.cpp - AMDGPU Assembly Compilation ------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions for compiling AMDGPU assembly to binary and
// linking binaries to create HSA code objects.
//
//===----------------------------------------------------------------------===//

#include "aster/Target/Binary/CompileAsm.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::aster;

#if ASTER_ENABLE_TARGET == 1
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Driver.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

static void initBackend() {
  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
    // If the `AMDGPU` LLVM target was built, initialize it.
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
  });
}

bool mlir::aster::amdgcn::target::hasAMDGPUTarget() { return true; }

LogicalResult mlir::aster::amdgcn::target::compileAsm(
    Location loc, StringRef asmCode, SmallVectorImpl<char> &binary,
    StringRef chip, StringRef features, StringRef triple) {
  initBackend();
  // Normalize the target triple
  llvm::Triple targetTriple(llvm::Triple::normalize(triple));

  // Lookup the target
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple.str(), error);
  if (!target)
    return emitError(loc, Twine("failed to lookup target: ") + error);

  // Create source manager with the assembly code
  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(asmCode),
                            llvm::SMLoc());

  // Create MC target options
  const llvm::MCTargetOptions mcOptions;

  // Create register info, asm info, and subtarget info
  std::unique_ptr<llvm::MCRegisterInfo> mri(
      target->createMCRegInfo(targetTriple.str()));
  assert(mri && "failed to create MCRegisterInfo");
  std::unique_ptr<llvm::MCAsmInfo> mai(
      target->createMCAsmInfo(*mri, targetTriple.str(), mcOptions));
  assert(mai && "failed to create MCAsmInfo");
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(targetTriple.str(), chip, features));
  assert(sti && "failed to create MCSubtargetInfo");

  // Create MC context
  llvm::MCContext ctx(targetTriple, mai.get(), mri.get(), sti.get(), &srcMgr,
                      &mcOptions);

  // Create object file info
  std::unique_ptr<llvm::MCObjectFileInfo> mofi(
      target->createMCObjectFileInfo(ctx, /*PIC=*/false,
                                     /*LargeCodeModel=*/false));
  assert(mofi && "failed to create MCObjectFileInfo");
  ctx.setObjectFileInfo(mofi.get());

  // Set compilation directory
  SmallString<128> cwd;
  if (!llvm::sys::fs::current_path(cwd))
    ctx.setCompilationDir(cwd);

  // Create instruction info
  std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());
  assert(mcii && "failed to create MCInstrInfo");

  // Create output stream
  llvm::raw_svector_ostream os(binary);

  // Create code emitter and assembler backend
  llvm::MCCodeEmitter *ce = target->createMCCodeEmitter(*mcii, ctx);
  assert(ce && "failed to create MCCodeEmitter");
  llvm::MCAsmBackend *mab = target->createMCAsmBackend(*sti, *mri, mcOptions);
  assert(mab && "failed to create MCAsmBackend");

  // Create object streamer
  std::unique_ptr<llvm::MCStreamer> mcStreamer;
  mcStreamer.reset(target->createMCObjectStreamer(
      targetTriple, ctx, std::unique_ptr<llvm::MCAsmBackend>(mab),
      mab->createObjectWriter(os), std::unique_ptr<llvm::MCCodeEmitter>(ce),
      *sti));
  assert(mcStreamer && "failed to create MCObjectStreamer");

  // Create assembly parser
  std::unique_ptr<llvm::MCAsmParser> parser(
      createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
  assert(parser && "failed to create MCAsmParser");

  // Create target-specific assembly parser
  std::unique_ptr<llvm::MCTargetAsmParser> tap(
      target->createMCAsmParser(*sti, *parser, *mcii, mcOptions));
  assert(tap && "assembler initialization error");

  // Set the target parser and run the assembly parser
  parser->setTargetParser(*tap);
  if (parser->Run(false))
    return emitError(loc, "assembly parsing failed");
  return success();
}

LLD_HAS_DRIVER(elf)

LogicalResult mlir::aster::amdgcn::target::linkBinary(
    Location loc, SmallVectorImpl<char> &binary, std::optional<StringRef> path,
    bool isLLDPath) {
  // Save the input binary to a temporary file
  int tempIsaBinaryFd = -1;
  SmallString<128> tempIsaBinaryFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel%%", "o", tempIsaBinaryFd,
                                         tempIsaBinaryFilename)) {
    return emitError(
        loc, "failed to create a temporary file for dumping the ISA binary");
  }
  llvm::FileRemover cleanupIsaBinary(tempIsaBinaryFilename);

  // Write the binary to the temporary file
  {
    llvm::raw_fd_ostream tempIsaBinaryOs(tempIsaBinaryFd, /*shouldClose=*/true);
    tempIsaBinaryOs.write(binary.data(), binary.size());
    tempIsaBinaryOs.flush();
  }

  // Create a temporary file for the HSA code object
  SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco",
                                         tempHsacoFilename)) {
    return emitError(
        loc, "failed to create a temporary file for the HSA code object");
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  static llvm::sys::Mutex mutex;
  {
    const llvm::sys::ScopedLock lock(mutex);
    // Invoke lld. Expect a true return value from lld.
    if (!lld::elf::link({"ld.lld", "-shared", tempIsaBinaryFilename.c_str(),
                         "-o", tempHsacoFilename.c_str()},
                        llvm::outs(), llvm::errs(), false, false)) {
      return emitError(loc) << "lld invocation error";
    }
    lld::CommonLinkerContext::destroy();
  }

  // Load the HSA code object
  auto hsacoFile =
      llvm::MemoryBuffer::getFile(tempHsacoFilename, /*IsText=*/false);
  if (!hsacoFile) {
    return emitError(loc,
                     "failed to read the HSA code object from the temp file");
  }

  // Replace the input binary with the linked HSA code object
  StringRef buffer = (*hsacoFile)->getBuffer();
  binary.clear();
  binary.append(buffer.begin(), buffer.end());
  return success();
}
#else
bool mlir::aster::amdgcn::target::hasAMDGPUTarget() { return false; }

LogicalResult mlir::aster::amdgcn::target::compileAsm(
    Location loc, StringRef asmCode, SmallVectorImpl<char> &binary,
    StringRef chip, StringRef features, StringRef triple) {
  return emitError(loc, "AMDGPU target support not enabled in this build");
}
LogicalResult mlir::aster::amdgcn::target::linkBinary(
    Location loc, SmallVectorImpl<char> &binary, std::optional<StringRef> path,
    bool isLLDPath) {
  return emitError(loc, "AMDGPU target support not enabled in this build");
}
#endif // ASTER_ENABLE_TARGET == 1
