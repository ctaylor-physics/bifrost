#ifndef BF_GRID_H_INCLUDE_GUARD_
#define BF_GRID_H_INCLUDE_GUARD_

// Bifrost Includes
#include <bifrost/common.h>
#include <bifrost/memory.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BFVGrid_impl* BFvgrid;

BFstatus bfVGridCreate(BFvgrid* plan);
BFstatus bfVGridInit(BFvgrid       plan,
                      BFarray const* positions,
                      BFarray const* kernels,
                      BFsize         ngrid,
                      BFbool         polmajor);
BFstatus bfVGridSetStream(BFvgrid    plan,
                           void const* stream);
BFstatus bfVGridSetPositions(BFvgrid       plan, 
                              BFarray const* positions);
BFstatus bfVGridSetKernels(BFvgrid      plan, 
                            BFarray const* kernels);
BFstatus bfVGridExecute(BFvgrid          plan,
                         BFarray const* in,
                         BFarray const* out);
BFstatus bfVGridDestroy(BFvgrid plan);

#ifdef __cplusplus
}
#endif

#endif // BFGRID_H_INCLUDE_GUARD
