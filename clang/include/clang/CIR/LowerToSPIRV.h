//====- LowerToSPIRV.h- Lowering from CIR to LLVM -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an interface for converting MLIR modules to SPIR-V
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_CIR_LOWERTOSPIRV_H
#define CLANG_CIR_LOWERTOSPIRV_H

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include <memory>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {
class MLIRContext;
class ModuleOp;

spirv::ModuleOp lowerFromMLIRToSPIRV(mlir::ModuleOp theModule,
                                    mlir::MLIRContext *mlirCtx);
} // namespace mlir

#endif // CLANG_CIR_LOWERTOSPIRV_H
