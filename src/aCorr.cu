
/*
 * Copyright (c) 2018-2020, The Bifrost Authors. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of The Bifrost Authors nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* 

 

Implements the grid-Correlation onto a GPU using CUDA. 

*/
#include <iostream>
#include "bifrost/aCorr.h"
#include "assert.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include "cuda.hpp"
#include "cuda/stream.hpp"
//#include <complex.h>
#include "Complex.hpp"

#include <thrust/device_vector.h>

#define tile 256 // Number of threads per thread-block

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

struct __attribute__((aligned(1))) nibble2 {
    // Yikes!  This is dicey since the packing order is implementation dependent!  
    signed char y:4, x:4;
};

struct __attribute__((aligned(1))) blenib2 {
    // Yikes!  This is dicey since the packing order is implementation dependent!
    signed char x:4, y:4;
};

template<typename In, typename Out>
__global__ void ACorr(int nantennas, int npol, int nbatch, int nchan,
		     In* d_in,
		     Out* d_out){

        int bid_x = blockIdx.x, bid_y = blockIdx.y, bid_z = blockIdx.z ;
        int blk_x = blockDim.x;
        int grid_x = gridDim.x, grid_y = gridDim.y, grid_z = gridDim.z ;
        int tid_x = threadIdx.x ;
        int pol_skip = grid_z*blk_x;
       // Making use of shared memory for faster memory accesses by the threads
        
	extern  __shared__ float2 shared[] ;
        In* xx = reinterpret_cast<In *>(shared);
        In* yy= xx + blk_x; 	

        // Access pattern is such that coaelescence is achieved both for read and writes to global and shared memory
        int bid1 =  ((bid_x * grid_y + bid_y ) * npol * grid_z + bid_z) * blk_x;
	int bid2 =  ((bid_x * grid_y + bid_y ) * npol * npol * grid_z + bid_z) * blk_x ;
	
	for (int i = 0;i<(npol*npol);i++){
			
		xx[tid_x]=d_in[bid1+i/2*pol_skip+tid_x];
        	yy[tid_x]=d_in[bid1+i%2*pol_skip+tid_x];

                d_out[bid2+i*pol_skip+tid_x].x +=  xx[tid_x].x*yy[tid_x].x + xx[tid_x].y*yy[tid_x].y;
          	d_out[bid2+i*pol_skip+tid_x].y +=  xx[tid_x].y*yy[tid_x].x - xx[tid_x].x*yy[tid_x].y;   
	}
       	__syncthreads();
}
 
template<typename In, typename Out>
inline void launch_acorr_kernel(int nantennas, int npol, bool polmajor, int nbatch, int nchan,
                               In*  d_in,
                               Out* d_out,
                               cudaStream_t stream=0) {
    cudaDeviceProp dev;
    cudaError_t error;
    error = cudaGetDeviceProperties(&dev, 0);
    if(error != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(error));
    int block_x=std::min(nantennas,dev.maxThreadsPerBlock/2);
    int grid_z=nantennas/block_x ;
    dim3 block(block_x,1);
    if(polmajor)npol=1;
    dim3 grid(nbatch, nchan, grid_z);

      
    void* args[] = {&nantennas,
	            &npol,
                    &nbatch,
		    &nchan,
		    &d_in,
                    &d_out};
     size_t loc_size=2 * block.x * sizeof(float2);
; // Shared memory size to be allocated for the kernel
	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)ACorr<In,Out>,
						 grid, block,
						 &args[0], loc_size*sizeof(float2), stream),BF_STATUS_INTERNAL_ERROR);
  
}

class BFaCorr_impl {
    typedef int    IType;
    typedef double FType;
public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
    typedef float  DType;

private:
    IType        _nantenna;
    IType        _npol;
    bool         _polmajor;
    cudaStream_t _stream;
public:
    BFaCorr_impl() : _nantenna(1), _npol(1), _polmajor(true), \
                      _stream(g_cuda_stream) {}
    inline IType nantenna()  const { return _nantenna;  }
    inline IType npol()       const { return _npol;       }
    inline bool polmajor()    const { return _polmajor;   }
    void init(IType nantenna,
              IType npol,
              bool  polmajor) {
        BF_TRACE();
        _nantenna  = nantenna;
        _npol       = npol;
        _polmajor   = polmajor;
    }
   void execute(BFarray const* in, BFarray const* out, int nbatch, int nchan) {
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION((out->dtype == BF_DTYPE_CF32) \
                                          || (out->dtype == BF_DTYPE_CF64), BF_STATUS_UNSUPPORTED_DTYPE);
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
               
#define LAUNCH_ACORR_KERNEL(IterType,OterType) \
        launch_acorr_kernel(_nantenna, _npol, _polmajor, nbatch, nchan,\
                             (IterType)in->data, (OterType)out->data, \
                             _stream)
        
        switch( in->dtype ) {
            case BF_DTYPE_CI4:
                if( in->big_endian ) {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(nibble2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(nibble2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                } else {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(blenib2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(blenib2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                }
                break;
            case BF_DTYPE_CI8:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(char2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(char2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                };
                break;
            case BF_DTYPE_CI16:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(short2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(short2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(int2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(int2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(long2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(long2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(float2*, Complex32*);   break;
                    case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(float2*, Complex64*);   break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(double2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(double2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        }
#undef LAUNCH_ACORR_KERNEL
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
    }
    void set_stream(cudaStream_t stream) {
        _stream = stream;
    }
};

BFstatus bfaCorrCreate(BFacorr* plan_ptr) {
    BF_TRACE();
    BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN_ELSE(*plan_ptr = new BFaCorr_impl(),
                       *plan_ptr = 0);
}
BFstatus bfaCorrInit(BFacorr       plan,
                      BFarray const* positions,
                      BFbool         polmajor) {
  
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(positions,                                BF_STATUS_INVALID_POINTER);
    BF_ASSERT(positions->ndim >= 4,                     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(positions->shape[0] == 3, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(space_accessible_from(positions->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    
    // Discover the dimensions of the positions/kernels.
    int npositions, nantenna, npol;
    npositions = positions->shape[1];
    for(int i=2; i<positions->ndim-2; ++i) {
        npositions *= positions->shape[i];
    }
    if( polmajor ) {
         npol = positions->shape[positions->ndim-2];
         nantenna = positions->shape[positions->ndim-1];
    } else {
        nantenna = positions->shape[positions->ndim-2];
        npol = positions->shape[positions->ndim-1];
    }
    // Validate
    BF_TRY_RETURN(plan->init(nantenna, npol, polmajor));
}
BFstatus bfaCorrSetStream(BFacorr    plan,
                           void const* stream) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfaCorrExecute(BFacorr          plan,
                         BFarray const* in,
                         BFarray const* out) {

    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
    BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
    BF_ASSERT( in->ndim >= 3,          BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->ndim == in->ndim, BF_STATUS_INVALID_SHAPE);

    int nbatch = in->shape[0];
    int nchan  = in->shape[1];
       
    BFarray in_flattened;
    if( in->ndim > 3 ) {
        // Keep the last two dim but attempt to flatten all others
        unsigned long keep_dims_mask = padded_dims_mask(in);
        keep_dims_mask |= 0x1 << (in->ndim-1);
        keep_dims_mask |= 0x1 << (in->ndim-2);
        keep_dims_mask |= 0x1 << (in->ndim-3);
        flatten(in,   &in_flattened, keep_dims_mask);
        in  =  &in_flattened;
        BF_ASSERT(in_flattened.ndim == 3, BF_STATUS_UNSUPPORTED_SHAPE);
    }

    if( plan->polmajor() ) {
        BF_ASSERT( in->shape[1] == plan->npol(),      BF_STATUS_INVALID_SHAPE);
        BF_ASSERT( in->shape[2] == plan->nantenna(), BF_STATUS_INVALID_SHAPE);
    } else {
        BF_ASSERT( in->shape[1] == plan->nantenna(), BF_STATUS_INVALID_SHAPE);
        BF_ASSERT( in->shape[2] == plan->npol(),      BF_STATUS_INVALID_SHAPE);
    }


    BFarray out_flattened;

    if( out->ndim > 3 ) {
        // Keep the last two dim but attempt to flatten all others
        unsigned long keep_dims_mask = padded_dims_mask(in);
        keep_dims_mask |= 0x1 << (out->ndim-1);
        keep_dims_mask |= 0x1 << (out->ndim-2);
        keep_dims_mask |= 0x1 << (out->ndim-3);
        flatten(out,   &out_flattened, keep_dims_mask);
        out  =  &out_flattened;
        BF_ASSERT(out_flattened.ndim == 3, BF_STATUS_UNSUPPORTED_SHAPE);
    }
 
    BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_TRY_RETURN(plan->execute(in, out, nbatch, nchan));
}
BFstatus bfaCorrDestroy(BFacorr plan) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    delete plan;
    return BF_STATUS_SUCCESS;
}
