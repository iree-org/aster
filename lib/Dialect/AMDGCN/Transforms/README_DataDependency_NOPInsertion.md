The following is extracted verbatim from amd-instinct-mi300-cdna3-instruction-set-architecture.pdf


Warning:

10.2.2. Important Timing Consideration
Since the data for a FLAT load can come from either LDS or the texture cache, and because these units have
different latencies, there is a potential race condition with respect to the VM_CNT and LGKM_CNT counters.
Because of this, the only sensible S_WAITCNT value to use after FLAT instructions is zero.


4.4. Data Dependency Resolution
Shader hardware resolves most data dependencies, but a few cases must be explicitly handled by the shader program. In these cases, the program must insert S_WAITCNT instructions to ensure that previous operations have completed before continuing.
The shader has three counters that track the progress of issued instructions. S_WAITCNT waits for the values of these counters to be at, or below, specified values before continuing.
These allow the shader writer to schedule long-latency instructions, execute unrelated work, and specify when results of long-latency operations are needed.
Instructions of a given type return in order, but instructions of different types can complete out-of-order. For example, both GWS and LDS instructions use LGKM_cnt, but they can return out-of-order.


VM_CNT: Vector memory count.
Determines when memory reads have returned data to VGPRs, or memory writes have completed.

Incremented every time a vector-memory read or write (MUBUF, MTBUF, or FLAT format) instruction is issued.
Decremented for reads when the data has been written back to the VGPRs, and for writes when the data has been written to the L2 cache. Ordering: Memory reads and writes return in the order they were issued, including mixing reads and writes.


LGKM_CNT: (LDS, GWS, (K)constant, (M)essage)
Determines when one of these low-latency instructions have completed.

Incremented by 1 for every LDS or GWS instruction issued, as well as by Dword-count for scalar-memory reads (1 for 1-dword loads, 2 for 2-dword or larger loads). S_memtime counts the same as an s_load_dwordx2.
Incremented by 1 for every FLAT instruction issued.
Decremented by 1 for LDS/GWS reads or atomic-with-return when the data has been returned to VGPRs.
Incremented by 1 for each S_SENDMSG issued. Decremented by 1 when message is sent out.
Decremented by 1 for LDS/GWS writes when the data has been written to LDS/GWS.
Decremented by 1 for each Dword returned from the data-cache (SMEM).
Ordering:

Instructions of different types are returned out-of-order.
Instructions of the same type are returned in the order they were issued, except scalar-memory reads, which can return out-of-order (in which case only S_WAITCNT 0 is the only legitimate value).


EXP_CNT: VGPR-export count.
Determines when data has been read out of the VGPR and sent to GWS, at which time it is safe to overwrite the contents of that VGPR.

Incremented when an GWS instruction is issued from the wavefront buffer.
Decremented for GWS when the last cycle of the GWS instruction is granted and executed (VGPRs read out).


4.5. Manually Inserted Wait States (NOPs)
The hardware does not check for the following dependencies; they must be resolved by inserting NOPs or independent instructions.

Comma-sparated tables follow (Table 11. and Table 12.).

Table 11. Required Software-inserted Wait States
First Instruction,Second Instruction,Wait,Notes

❌ Case 1, S_SETREG <*>,S_GETREG <same reg>,2,

❌ Case 2, S_SETREG <*>,S_SETREG <same reg>,2,

❌ Case 3, SET_VSKIP,S_GETREG MODE,2,Reads VSKIP from MODE.

❌ Case 4, S_SETREG MODE.vskip,any vector op,2,Requires two nops or non-vector instructions.

❌ Case 5, VALU that sets VCC or EXEC,VALU that uses EXECZ or VCCZ as a data source,5,

❌ Case 6, VALU writes SGPR/VCC (readlane, cmp, add/sub, div_scale),V_[READ,WRITE]LANE using that SGPR/VCC as the lane select,4,

❌ Case 7, VALU writes VCC (including v_div_scale),V_DIV_FMAS,4,

⚠️ Case 8,
✅ FLAT_STORE_X3
✅ FLAT_STORE_X4
FLAT_ATOMIC_[F]CMPSWAP_X2
(and global & scratch stores/atomics)
BUFFER_STORE_DWORD_X3
BUFFER_STORE_DWORD_X4
BUFFER_STORE_FORMAT_XYZ
BUFFER_STORE_FORMAT_XYZW
BUFFER_ATOMIC_[F]CMPSWAP_X2,Write VGPRs holding writedata from those instructions.,1,BUFFER_STORE_* operations that use an SGPR for "offset" do not require any wait states.


⚠️ Case 9,
✅ FLAT_STORE_X3
✅ FLAT_STORE_X4
(and global & scratch stores/atomics)
FLAT_ATOMIC_[F]CMPSWAP_X2
BUFFER_STORE_DWORD_X3
BUFFER_STORE_DWORD_X4
BUFFER_STORE_FORMAT_XYZ
BUFFER_STORE_FORMAT_XYZW
BUFFER_ATOMIC_[F]CMPSWAP_X2,VALU writes VGPRs holding writedata from those instructions.,2,BUFFER_STORE_* operations that use an SGPR for "offset" do not require any wait states.

✅ Case 10, VALU writes SGPR,VMEM reads that SGPR,5,Hardware assumes that there is no dependency here. If the VALU writes the SGPR that is used by a VMEM, the user must add five wait states.

❌ Case 11, SALU writes M0,GDS, S_SENDMSG,1,

❌ Case 12, VALU writes VGPR,VALU DPP reads that VGPR,2,

❌ Case 13, VALU writes EXEC,VALU DPP op,5,ALU does not forward EXEC to DPP.

❌ Case 14, Mixed use of VCC: alias vs SGPR#
v_readlane, v_readfirstlane
v_cmp
v_add* i/u
v_sub_i/u
v_div_scale (writes vcc),VALU which reads VCC as a constant (not as a carry-in which is 0 wait states).,1,VCC can be accessed by name or by the logical SGPR which holds VCC. The data dependency check logic does not understand that these are the same register and do not prevent races.

❌ Case 15, S_SETREG TRAPSTS,RFE, RFE_restore,1,

❌ Case 16, SALU writes M0,LDS "add-TID" instruction, buffer_store_LDS_dword, scratch or global with LDS=1,1,

❌ Case 17, SALU writes M0,S_MOVEREL,1,

❌ Case 18, VALU writes SGPR/VCC:
v_readlane, v_readfirstlane, v_cmp,
v_add_i/u, v_sub_i/u, v_div_scale*,VALU reads SGPR as constant,2,
,VALU reads SGPR as carry-in,0,
,v_readlane, v_writelane reads SGPR as lane-select,4,
v_cmpx,VALU reads EXEC as constant,2,
,V_readlane, v_readfirstlane, v_writelane,4,
,Other VALU,0,

❌ Case 19, VALU writes VGPRn,v_readlane vsrc0 reads VGPRn,1,

❌ Case 20, VALU op which uses OPSEL or SDWA with changes the result's bit position,VALU op consumes result of that op,1,

❌ Case 21, VALU Trans op,Non-trans VALU op consumes result of that op,1,

Table 12. Trans Ops
V_EXP_F32,V_LOG_F32,V_RCP_F32,V_RCP_IFLAG_F32
V_RSQ_F32,V_RCP_F64,V_RSQ_F64,V_SQRT_F32
V_SQRT_F64,V_SIN_F32,V_COS_F32,V_RCP_F16
V_SQRT_F16,V_RSQ_F16,V_LOG_F16,V_EXP_F16
V_SIN_F16,V_COS_F16,V_EXP_LEGACY_F32,V_LOG_LEGACY_F32

7.5. Dependency Resolution: Required Independent
Instructions
The table below indicates timing conditions which require the user to insert NOPs (or independent VALU
instructions).
DLop Dot products
XDLOP Matrix math on {I8, F16, BF16}
DGEMM V_MFMA…F64
PASS 4 clock cycles
Table 37. VOP3P-Matrix Opcodes Required NOPs

First Instruction,Second Instruction,Required Waits,Comments

✅ Case 100, Non-DLops VALU Write VGPR,"V_MFMA* read VGPR OR V_SMFMA* read VGPR",2,"No internal 4 & 8 cycle forwarding path."

❌ Case 101, DL ops Write VGPR,"DLops read VGPR as SrcC, and the opcode is exactly the same as 1st DLops",0,"supports same opcode of DLops back-to-back SrcC forwarding which is used for accumulation."
DL ops Write VGPR,"DLops read VGPR as SrcA/B, and the opcode is exactly the same as 1st DLops",3,"does not support SrcA/B forwarding in DLops"
DL ops Write VGPR,"Any opcode read/write VGPR that is not the same as 1st DLops opcode (RAW + WAW)",3,"Disable all of the forwarding path from DL ops to normal VALU/VM/LDS/FLAT ops"

❌ Case 102, "XDL Write VGPR or V_SMFMA* Write VGPR","XDL read VGPR as Source C exactly same with 1st vDst OR V_SMFMA* read VGPR for Matrix C exactly same with 1st vDst","2 if 1st V_MFMA is 2 passes
0 if 1st V_MFMA is 4 passes
0 if 1st V_MFMA is 8 passes
0 if 1st V_MFMA is 16 passes","the two V_MFMA must be the same number passes and vDst and vSrc start from the same offset and same VGPR size. V_MFMA & V_SMFMA must be the same number passes and both vDst start from the same offset and same VGPR size. Note: V_SMFMA reads vdst for Matrix C."

❌ Case 103, "XDL Write VGPR or V_SMFMA* Write VGPR","XDL read VGPR as Source C overlapped with 1st vDst OR V_SMFMA* read VGPR for Matrix C overlapped with 1st vDst","3 if 1st V_MFMA is 2 passes
5 if 1st V_MFMA is 4 passes
9 if 1st V_MFMA is 8 passes
17 if 1st V_MFMA is 16 passes","overlapped with XDL. Note: V_SMFMA reads vdst for Matrix C."

❌ Case 104, "XDL Write VGPR or V_SMFMA* Write VGPR","S/DGEMM read VGPR as Source C","3 if 1st V_MFMA is 2 passes
5 if 1st V_MFMA is 4 passes
9 if 1st V_MFMA is 8 passes
17 if 1st V_MFMA is 16 passes","Overlapped with S/DGEMM"

❌ Case 105, "XDL Write VGPR or V_SMFMA* Write VGPR","V_MFMA read VGPR as SrcA or SrcB OR V_SMFMA* read VGPR as SrcA or SrcB or Index SrcC","5 if 1st V_MFMA is 2 passes
7 if 1st V_MFMA is 4 passes
11 if 1st V_MFMA is 8 passes
19 if 1st V_MFMA is 16 passes","No internal forwarding path waits for previous V_MFMA/V_SMFMA* commit result to VGPR. V_SMFMA uses srcC address for extra Index C Reads"

❌ Case 106, "XDL Write VGPR or V_SMFMA* Write VGPR","1) VM, L/GDS, FLAT, Export Read VGPR overlapped with 1st vDst
2) VALU read/write VGPR (RAW + WAW)","5 if 1st V_MFMA is 2 passes
7 if 1st V_MFMA is 4 passes
11 if 1st V_MFMA is 8 passes
19 if 1st V_MFMA is 16 passes","V_MFMA_F32_4X4X4F16
V_MFMA_F32_16X16X16F16
V_MFMA_F32_32X32X8F16
V_MFMA_F32_32X32X4F16"

❌ Case 107, SGEMM Write VGPR,"XDL read VGPR as Source C exactly same with 1st vDst OR V_SMFMA* read VGPR for Matrix C exactly same with 1st vDst",0,"the two V_MFMA must be the same number passes and vDst and vSrc start from the same offset and same VGPR size. V_MFMA & V_SMFMA must be the same number passes and both vDst start from the same offset and same VGPR size. Note: V_SMFMA reads vdst for Matrix C."

❌ Case 108, "SGEMM Write VGPR","XDL read VGPR as Source C overlapped with 1st vDst OR V_SMFMA* read VGPR for Matrix C overlapped with 1st vDst","2 if 1st V_MFMA is 2 passes; 4 if 4 passes; 8 if 8 passes; 16 if 16 passes","V_SMFMA reads vdst for Matrix C."

❌ Case 109, "SGEMM Write VGPR","S/DGEMM read VGPR as Source C","2 if 1st V_MFMA is 2 passes; 4 if 4 passes; 8 if 8 passes; 16 if 16 passes","Overlapped with S/DGEMM"

❌ Case 110, "SGEMM Write VGPR","V_MFMA read VGPR as SrcA or SrcB OR V_SMFMA* read VGPR as SrcA/SrcB/Index SrcC","4 if 1st V_MFMA is 2 passes; 6 if 4 passes; 10 if 8 passes; 18 if 16 passes","No internal forwarding path; V_SMFMA uses SrcC for Index reads."

❌ Case 111, "SGEMM Write VGPR","1) VM, L/GDS, FLAT, Export Read VGPR overlapped with 1st vDst  2) VALU read/write VGPR (RAW+WAW)","4 if 1st V_MFMA is 2 passes; 6 if 4 passes; 10 if 8 passes; 18 if 16 passes",""

❌ Case 112, "V_MFMA_16x16x4_F64 Write VGPR","V_MFMA_16x16x4_F64 read VGPR as Source C exactly same with 1st vDst","0","The two V_MFMA must be the same number passes and vDst and vSrc start from the same offset." :contentReference[oaicite:0]{index=0}

❌ Case 113, "V_MFMA_16x16x4_F64 Write VGPR","S/DGEMM read VGPR as Source C overlapped with 1st vDst","9","Overlapped, different VGPR access sequence" :contentReference[oaicite:1]{index=1}

❌ Case 114, "V_MFMA_16x16x4_F64 Write VGPR","XDL read VGPR as Source C overlapped with 1st vDst","0","" :contentReference[oaicite:2]{index=2}

❌ Case 115, "V_MFMA_16x16x4_F64 Write VGPR","V_SMFMA* read VGPR for Matrix C overlapped with 1st vDst","0","V_SMFMA reads vdst for Matrix C." :contentReference[oaicite:3]{index=3}

❌ Case 116, "V_MFMA_16x16x4_F64 Write VGPR","S/DGEMM read VGPR as SrcA or SrcB","11","No internal forwarding path, need to wait previous V_MFMA commit result to VGPR" :contentReference[oaicite:4]{index=4}

❌ Case 117, "V_MFMA_16x16x4_F64 Write VGPR","XDL read VGPR as SrcA or SrcB","11","" :contentReference[oaicite:5]{index=5}

❌ Case 118, "V_MFMA_16x16x4_F64 Write VGPR","V_SMFMA* read VGPR as SrcA or SrcB or Index SrcC","11","V_SMFMA uses srcC address for extra Index C Reads" :contentReference[oaicite:6]{index=6}

❌ Case 119, "V_MFMA_16x16x4_F64 Write VGPR","VALU read/write VGPR (RAW + WAW)","11","" :contentReference[oaicite:7]{index=7}

❌ Case 120, "V_MFMA_16x16x4_F64 Write VGPR","VM, L/GDS, FLAT and Export Read VGPR overlapped with 1st vDst","18","No internal forwarding path, need to wait previous V_MFMA commit result to VGPR" :contentReference[oaicite:8]{index=8}
