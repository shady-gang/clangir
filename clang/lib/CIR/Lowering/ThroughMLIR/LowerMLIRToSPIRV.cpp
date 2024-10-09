#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "clang/CIR/Passes.h"
#include "llvm/Support/TimeProfiler.h"

#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

namespace mlir {

mlir::ModuleOp lowerFromMLIRToSPIRV(mlir::ModuleOp theModule,
                                    mlir::MLIRContext *mlirCtx) {
  llvm::TimeTraceScope scope("Lower from MLIR to SPIR-V");

  mlir::PassManager pm(mlirCtx);

  // TODO
  //pm.addPass(cir::createConvertMLIRToSPIRVPass());

  //auto result = !mlir::failed(pm.run(theModule));
  //if (!result)
  //  report_fatal_error("The pass manager failed to lower CIR to SPIRV dialect!");
}

} // namespace mlir