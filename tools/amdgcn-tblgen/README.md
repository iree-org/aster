## Interacting with LLVM Tablegen

### Producing MLIR operations

To convert AMDGPU target instructions records to MLIR operations use:

```bash
./build/bin/amdgcn-tblgen -I ${LLVM_PROJECT_SRC}/llvm/include/ \
    -I ${LLVM_PROJECT_SRC}/llvm/lib/Target/AMDGPU/ \
    ${LLVM_PROJECT_SRC}/llvm/lib/Target/AMDGPU/AMDGPU.td \ # This file represents the entire AMDGPU backend.
    --amdgcn-op-gen -o amdgcn-ops.td
```

### Notes on the converter

Currently the converter MLIR is only a prototype, and needs to be improved, take
for example:

```tblgen
def ScratchLoadDwordx4Op : AMDGCN_Op<"scratch_load_dwordx4", []> {
  let summary = [{AMDGCN for `scratch_load_dwordx4`}];
  let arguments = (ins /*OutOperands=*/AnyType:$vdst, /*InOperands=*/AnyType:$offset, AnyType:$cpol);
  let assemblyFormat = [{
    $operands attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    static constexpr std::string_view asmStrings[] = {
      "scratch_load_dwordx4 $vdst, $vaddr, $saddr$offset$cpol",
      "scratch_load_dwordx4 $vdst, $vaddr, off$offset$cpol",
      "scratch_load_dwordx4 $vdst, off, off$offset$cpol"
    };
    static constexpr std::string_view decoders[] = {
      "GFX9"
    };
    static constexpr std::string_view tblgenNames[] = {
      "SCRATCH_LOAD_DWORDX4_ST_gfx940",
      "SCRATCH_LOAD_DWORDX4_SVS_gfx940",
      "SCRATCH_LOAD_DWORDX4_VE_gfx940"
    };
  }];
}
```

For `gfx940` the `scratch_load_dwordx4` instruction has three variants, as symbolized
by the three ASM formats. These three variants all represent the same instruction
semantics, but specify slightly different behaviors. For example the first variant,
allows specifying both a VGPR and SGPR address component, while the last variant
has only a constant integer offset field.

These variants need to be properly reflected in the arguments of the operations,
and it should be the job of the operation verifier to determine which configuration
is the one being used, and complain if the operation is invalid.
These means marking some of the arguments as optional.
Further, the converter should add a custom Op builder symbolizing each of the variants,
allowing to only create valid variants.

Currently, the converter also prints which TBLGEN records were used to produce the
operation. These records can be explored using the options in the next section.

Finally, the converter only converts operations supported in GFX940. This is not
a restriction on the converter, but rather a demonstration on how to filter
targets. See lines 116-123 in `tools/amdgcn-tblgen/OpGen.cpp`.

### Exploring AMDGPU tablegen records

To dump all the records use:

```bash
# NOTE: This will create a large file difficult to manually explore (350MB+)
./build/bin/amdgcn-tblgen -I ${LLVM_PROJECT_SRC}/llvm/include/ \
    -I ${LLVM_PROJECT_SRC}/llvm/lib/Target/AMDGPU/ \
    ${LLVM_PROJECT_SRC}/llvm/lib/Target/AMDGPU/AMDGPU.td \
    -o records.log --print-records
```

Dump a subset of records using a regex:

```bash
./build/bin/amdgcn-tblgen -I ${LLVM_PROJECT_SRC}/llvm/include/ \
    -I ${LLVM_PROJECT_SRC}/llvm/lib/Target/AMDGPU/ \
    ${LLVM_PROJECT_SRC}/llvm/lib/Target/AMDGPU/AMDGPU.td \
    -o load_dword.log --print-records --amdgcn-dump-insts --amdgcn-regex="SCRATCH_LOAD_DWORDX4"
```

Example record:

```
V_MFMA_F32_16X16X128_F8F6F4_f4_f4_gfx940_acd {	// VOP_Real InstructionEncoding Instruction AMDGPUInst PredicateControl InstSI SIMCInstr VOP3_Real VOP3P_Real Enc64 VOP3Pe_MAI_Base VOP3Pe_MAI MFMA_F8F6F4_WithSizeTable MFMA_F8F6F4_WithSizeTable_Helper
  field bit isRegisterLoad = 0;
  field bit isRegisterStore = 0;
  field bit SALU = 0;
  field bit VALU = 1;
  field bit SOP1 = 0;
  field bit SOP2 = 0;
  field bit SOPC = 0;
  field bit SOPK = 0;
  field bit SOPP = 0;
  field bit VOP1 = 0;
  field bit VOP2 = 0;
  field bit VOPC = 0;
  field bit VOP3 = 1;
  field bit VOP3P = 0;
  field bit VINTRP = 0;
  field bit SDWA = 0;
  field bit DPP = 0;
  field bit TRANS = 0;
  field bit MUBUF = 0;
  field bit MTBUF = 0;
  field bit SMRD = 0;
  field bit MIMG = 0;
  field bit VIMAGE = 0;
  field bit VSAMPLE = 0;
  field bit EXP = 0;
  field bit FLAT = 0;
  field bit DS = 0;
  field bit Spill = 0;
  field bit LDSDIR = 0;
  field bit VINTERP = 0;
  field bit VOPD3 = 0;
  field bit VM_CNT = 0;
  field bit EXP_CNT = 0;
  field bit LGKM_CNT = 0;
  field bit WQM = 0;
  field bit DisableWQM = 0;
  field bit Gather4 = 0;
  field bit TENSOR_CNT = 0;
  field bit ScalarStore = 0;
  field bit FixedSize = 0;
  field bit ASYNC_CNT = 0;
  field bit VOP3_OPSEL = 0;
  field bit maybeAtomic = 1;
  field bit FPClamp = 0;
  field bit IntClamp = 0;
  field bit ClampLo = 0;
  field bit ClampHi = 0;
  field bit IsPacked = 0;
  field bit D16Buf = 0;
  field bit FlatGlobal = 0;
  field bit ReadsModeReg = 0;
  field bit FPDPRounding = 0;
  field bit FPAtomic = 0;
  field bit IsMAI = 0;
  field bit IsDOT = 0;
  field bit FlatScratch = 0;
  field bit IsAtomicNoRet = 0;
  field bit IsAtomicRet = 0;
  field bit IsWMMA = 0;
  field bit TiedSourceNotRead = 0;
  field bit IsNeverUniform = 0;
  field bit GWS = 0;
  field bit IsSWMMAC = 0;
  field bits<64> Inst = { blgp{2}, blgp{1}, blgp{0}, src1{9}, src0{9}, src2{8}, src2{7}, src2{6}, src2{5}, src2{4}, src2{3}, src2{2}, src2{1}, src2{0}, src1{8}, src1{7}, src1{6}, src1{5}, src1{4}, src1{3}, src1{2}, src1{1}, src1{0}, src0{8}, src0{7}, src0{6}, src0{5}, src0{4}, src0{3}, src0{2}, src0{1}, src0{0}, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, cbsz{2}, cbsz{1}, cbsz{0}, vdst{7}, vdst{6}, vdst{5}, vdst{4}, vdst{3}, vdst{2}, vdst{1}, vdst{0} };
  Instruction Opcode = V_MFMA_F32_16X16X128_F8F6F4_f4_f4_gfx940_acd;
  bit IsSingle = 1;
  int Size = 8;
  string DecoderNamespace = "GFX940";
  list<Predicate> Predicates = [HasGFX950Insts, isGFX940Plus];
  string DecoderMethod = "";
  bit hasCompleteDecoder = 1;
  string Namespace = "AMDGPU";
  dag OutOperandList = (outs ADst_128:$vdst);
  dag InOperandList = (ins AVSrc_128:$src0, AVSrc_128:$src1, AISrc_128_f32:$src2, CBSZ:$cbsz, blgp:$blgp);
  string AsmString = "v_mfma_f32_16x16x128_f8f6f4$vdst, $src0, $src1, $src2$cbsz$blgp";
  EncodingByHwMode EncodingInfos = ?;
  list<dag> Pattern = [];
  list<Register> Uses = [MODE, EXEC];
  list<Register> Defs = [];
  int CodeSize = 0;
  int AddedComplexity = 0;
  bit isPreISelOpcode = 0;
  bit isReturn = 0;
  bit isBranch = 0;
  bit isEHScopeReturn = 0;
  bit isIndirectBranch = 0;
  bit isCompare = 0;
  bit isMoveImm = 0;
  bit isMoveReg = 0;
  bit isBitcast = 0;
  bit isSelect = 0;
  bit isBarrier = 0;
  bit isCall = 0;
  bit isAdd = 0;
  bit isTrap = 0;
  bit canFoldAsLoad = 0;
  bit mayLoad = 0;
  bit mayStore = 0;
  bit mayRaiseFPException = 0;
  bit isConvertibleToThreeAddress = 0;
  bit isCommutable = 0;
  bit isTerminator = 0;
  bit isReMaterializable = 0;
  bit isPredicable = 0;
  bit isUnpredicable = 0;
  bit hasDelaySlot = 0;
  bit usesCustomInserter = 0;
  bit hasPostISelHook = 0;
  bit hasCtrlDep = 0;
  bit isNotDuplicable = 0;
  bit isConvergent = 1;
  bit isAuthenticated = 0;
  bit isAsCheapAsAMove = 0;
  bit hasExtraSrcRegAllocReq = 1;
  bit hasExtraDefRegAllocReq = 0;
  bit isRegSequence = 0;
  bit isPseudo = 0;
  bit isMeta = 0;
  bit isExtractSubreg = 0;
  bit isInsertSubreg = 0;
  bit variadicOpsAreDefs = 0;
  bit hasSideEffects = ?;
  bit isCodeGenOnly = 0;
  bit isAsmParserOnly = 1;
  bit hasNoSchedulingInfo = 0;
  InstrItinClass Itinerary = NullALU;
  list<SchedReadWrite> SchedRW = [Write32Bit];
  string Constraints = "";
  string PostEncoderMethod = "";
  bits<64> TSFlags = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
  string AsmMatchConverter = "";
  string TwoOperandAliasConstraint = "";
  string AsmVariantName = "VOP3";
  bit UseNamedOperandTable = 1;
  bit UseLogicalOperandMappings = 0;
  bit FastISelShouldIgnore = 0;
  bit HasPositionOrder = 0;
  Predicate SubtargetPredicate = HasGFX950Insts;
  Predicate AssemblerPredicate = isGFX940Plus;
  Predicate WaveSizePredicate = TruePredicate;
  True16PredicateClass True16Predicate = NoTrue16Predicate;
  list<Predicate> OtherPredicates = [];
  string PseudoInstr = "V_MFMA_F32_16X16X128_F8F6F4_f4_f4_e64";
  int Subtarget = 9;
  VOPProfile Pfl = VOPProfileMAI_F32_V4I32_V4I32_X128;
  bits<8> vdst = { ?, ?, ?, ?, ?, ?, ?, ? };
  bits<10> src0 = { ?, ?, ?, ?, ?, ?, ?, ?, ?, ? };
  bits<10> src1 = { ?, ?, ?, ?, ?, ?, ?, ?, ?, ? };
  bits<9> src2 = { ?, ?, ?, ?, ?, ?, ?, ?, ? };
  bits<3> blgp = { ?, ?, ? };
  bits<3> cbsz = { ?, ?, ? };
  bits<4> abid = { ?, ?, ?, ? };
  Instruction F8F8Opcode = V_MFMA_F32_16X16X128_F8F6F4_f8_f8_gfx940_acd;
  bits<8> NumRegsSrcA = { 0, 0, 0, 0, 0, 1, 0, 0 };
  bits<8> NumRegsSrcB = { 0, 0, 0, 0, 0, 1, 0, 0 };
}
```

The most important fields in the above record are, the `Inst` field which specifies
the actual machine encoding of the instruction, and the `AsmString`, encoding how to
parse and print the instruction.
