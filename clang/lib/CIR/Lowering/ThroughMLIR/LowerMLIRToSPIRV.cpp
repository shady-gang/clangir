#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "clang/CIR/Passes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"

#include "mlir/Conversion/ConvertToSPIRV/ConvertToSPIRVPass.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

namespace mlir {

mlir::ModuleOp lowerFromMLIRToSPIRV(mlir::ModuleOp theModule,
                                    mlir::MLIRContext *mlirCtx) {
  llvm::TimeTraceScope scope("Lower from MLIR to SPIR-V");

  mlir::PassManager pm(mlirCtx);

  pm.addPass(cir::createConvertCIRToMLIRPass());
  pm.addPass(mlir::createConvertToSPIRVPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    llvm::report_fatal_error("The pass manager failed to lower CIR to SPIRV dialect!");

  return theModule;
}

} // namespace mlir