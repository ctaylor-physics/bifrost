#ifndef BF_XGRID_H_INCLUDE_GUARD_
#define BF_XGRID_H_INCLUDE_GUARD_

// Bifrost Includes
#include <bifrost/common.h>
#include <bifrost/memory.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BFxgrid_impl* BFxgrid;

BFstatus bfxGridCreate(BFxgrid* plan);
BFstatus bfxGridInit(BFxgrid       plan,
                      BFsize         ngrid,
                      BFbool         polmajor);
BFstatus bfxGridSetStream(BFxgrid    plan,
                           void const* stream);
BFstatus bfxGridExecute(BFxgrid          plan,
                         BFarray const* in,
                         BFarray const* out);
BFstatus bfxGridDestroy(BFxgrid plan);

#ifdef __cplusplus
}
#endif

#endif
