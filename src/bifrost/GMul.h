#ifndef BF_GRIDMUL_H_INCLUDE_GUARD_
#define BF_GRIDMUL_H_INCLUDE_GUARD_

// Bifrost Includes
#include <bifrost/common.h>
#include <bifrost/memory.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BFgmul_impl* BFgmul;

BFstatus bfgMulCreate(BFgmul* plan);
BFstatus bfgMulInit(BFgmul       plan,
                      BFsize         ngrid,
                      BFbool         polmajor);
BFstatus bfgMulSetStream(BFgmul    plan,
                           void const* stream);
BFstatus bfgMulExecute(BFgmul          plan,
                         BFarray const* in,
                         BFarray const* out);
BFstatus bfgMulDestroy(BFgmul plan);

#ifdef __cplusplus
}
#endif

#endif // BF_GRIDMUL_H_INCLUDE_GUARD
