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

This module takes in the two-dimensional voltage patterns of the
X and Y uv-grids to perform element-wise complex multiplication 
producing the full stokes output sequence in the order XX*, YY*, XY* & YX*.
The products are estimated for every time-stamp and frequency channel separately 
which keep accumulating and updated for the specified inegration time in the output.

*/
#include <iostream>
#include "bifrost/XGrid.h"
#include "assert.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include "cuda.hpp"
#include "cuda/stream.hpp"
#include "Complex.hpp"

struct __attribute__((aligned(1))) nibble2 {
    // Yikes!  This is dicey since the packing order is implementation dependent!  
    signed char y:4, x:4;
};

struct __attribute__((aligned(1))) blenib2 {
    // Yikes!  This is dicey since the packing order is implementation dependent!
    signed char x:4, y:4;
};

template<typename In, typename Out>
__global__ void XGrid_kernel(int npol, int gridsize, int nbatch, int nchan,
		     const In* __restrict__  d_in,
                     Out* d_out){

        int bid_x = blockIdx.x, bid_y = blockIdx.y, bid_z = blockIdx.z ;
        int blk_x = blockDim.x ;
        int grid_x = gridDim.x, grid_y = gridDim.y , grid_z = gridDim.z ;
        int tid_x = threadIdx.x ;
	int pol_skip = grid_z*blk_x;
    
        extern  __shared__ Complex<float> shared[] ;
        In* xx = reinterpret_cast<In *>(shared);
        In* yy = xx + blk_x;
 
        int tt = 1;
        if(npol>1) tt=(int)npol/2;

        int bid  = ((bid_x * grid_y + bid_y) * tt  * grid_z  + bid_z) * blk_x ;
	int bid2 = ((bid_x * grid_y + bid_y) * npol * grid_z  + bid_z) * blk_x ;

        #pragma unroll
	for(int i=0;i<npol;i++){

                xx[tid_x] = d_in[bid+i/2*pol_skip+tid_x];
		yy[tid_x] = d_in[bid+i%2*pol_skip+tid_x];
	        
		d_out[bid2+i*pol_skip+tid_x].x += xx[tid_x].x*yy[tid_x].x + xx[tid_x].y*yy[tid_x].y;  
	       	d_out[bid2+i*pol_skip+tid_x].y += xx[tid_x].y*yy[tid_x].x - xx[tid_x].x*yy[tid_x].y;
	}
        __syncthreads();
}

template<typename In, typename Out>
inline void launch_xgrid_kernel(int npol, bool polmajor, int gridsize, int nbatch, int nchan, 
                               In*  d_in,
                               Out* d_out,
                               cudaStream_t stream=0) {
   
    cudaDeviceProp dev;
    cudaError_t error;
    int grid_pix = gridsize * gridsize ;
    error = cudaGetDeviceProperties(&dev, 0);
    if(error != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(error));
    int block_x=std::min(grid_pix, dev.maxThreadsPerBlock/2);
    int grid_z=grid_pix/block_x ;
    dim3 block(block_x,1);
    if(polmajor)npol=1;
    dim3 grid(nbatch, nchan, grid_z);

    void* args[] = {&npol,
                    &gridsize, 
                    &nbatch,
		    &nchan,
		    &d_in,
                    &d_out};
    size_t loc_size=2*block.x;
    BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)XGrid_kernel<In,Out>,
						 grid, block,
						 &args[0], loc_size*sizeof(Complex<float>), stream),BF_STATUS_INTERNAL_ERROR);    
}

class BFxgrid_impl {
    typedef int    IType;
    typedef double FType;
public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
    typedef float  DType;
private:
    bool         _polmajor;
    IType        _gridsize;
    cudaStream_t _stream;
public:
    BFxgrid_impl() : _polmajor(true), _stream(g_cuda_stream) {}
    inline bool polmajor()    const { return _polmajor;   }
    inline IType gridsize()   const { return _gridsize;   }
    void init(bool  polmajor, IType gridsize) {
        BF_TRACE();
        _polmajor   = polmajor;
        _gridsize   = gridsize;
    }
   void execute(BFarray const* in, BFarray const* out, int nbatch, int nchan, int npol) {
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION((out->dtype == BF_DTYPE_CF32) \
                                          || (out->dtype == BF_DTYPE_CF64), BF_STATUS_UNSUPPORTED_DTYPE);
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
    
        
#define LAUNCH_XGRID_KERNEL(IterType,OterType) \
        launch_xgrid_kernel(npol, _polmajor, _gridsize, nbatch, nchan, \
                             (IterType)in->data, (OterType)out->data, \
                             _stream)
        
        switch( in->dtype ) {
            case BF_DTYPE_CI4:
                if( in->big_endian ) {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_XGRID_KERNEL(nibble2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_XGRID_KERNEL(nibble2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                } else {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_XGRID_KERNEL(blenib2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_XGRID_KERNEL(blenib2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                }
                break;
            case BF_DTYPE_CI8:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_XGRID_KERNEL(char2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_XGRID_KERNEL(char2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                };
                break;
            case BF_DTYPE_CI16:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_XGRID_KERNEL(short2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_XGRID_KERNEL(short2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_XGRID_KERNEL(int2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_XGRID_KERNEL(int2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_XGRID_KERNEL(long2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_XGRID_KERNEL(long2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_XGRID_KERNEL(float2*, Complex32*);   break;
                    case BF_DTYPE_CF64: LAUNCH_XGRID_KERNEL(float2*, Complex64*);   break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_XGRID_KERNEL(double2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_XGRID_KERNEL(double2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        }
#undef LAUNCH_XGRID_KERNEL
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
    }
    void set_stream(cudaStream_t stream) {
        _stream = stream;
    }
};

BFstatus bfxGridCreate(BFxgrid* plan_ptr) {
    BF_TRACE();
    BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN_ELSE(*plan_ptr = new BFxgrid_impl(),
                       *plan_ptr = 0);
}
BFstatus bfxGridInit(BFxgrid       plan,
                      BFsize         gridsize,
                      BFbool         polmajor) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);    
    BF_TRY_RETURN(plan->init(polmajor, gridsize));
}
BFstatus bfxGridSetStream(BFxgrid    plan,
                           void const* stream) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfxGridExecute(BFxgrid          plan,
                         BFarray const* in,
                         BFarray const* out) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
    BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
    BF_ASSERT( in->ndim == 6,          BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->ndim == in->ndim-1, BF_STATUS_INVALID_SHAPE);
    BFarray in_flattened;
        
    int nbatch = in->shape[1];
    int nchan = in->shape[2];
    int npol = in->shape[3];
     
    if( in->ndim > 5 ) {
        unsigned long keep_dims_mask = padded_dims_mask(out);
        keep_dims_mask |= 0x1 << (in->ndim-1);
        keep_dims_mask |= 0x1 << (in->ndim-2);
        keep_dims_mask |= 0x1 << (in->ndim-3);
        keep_dims_mask |= 0x1 << (in->ndim-4);
        keep_dims_mask |= 0x1 << (in->ndim-5);
        keep_dims_mask |= 0x1 << (in->ndim-6);
	flatten(in,   &in_flattened, keep_dims_mask);
        in  =  &in_flattened;
       BF_ASSERT(in_flattened.ndim == 6, BF_STATUS_UNSUPPORTED_SHAPE); 
    }

    BFarray out_flattened;
    if( out->ndim > 4 ) {
        unsigned long keep_dims_mask = padded_dims_mask(out);
        keep_dims_mask |= 0x1 << (out->ndim-1);
        keep_dims_mask |= 0x1 << (out->ndim-2);
        keep_dims_mask |= 0x1 << (out->ndim-3);
        keep_dims_mask |= 0x1 << (out->ndim-4);
        flatten(out,   &out_flattened, keep_dims_mask);
        out  =  &out_flattened;
        BF_ASSERT(out_flattened.ndim == 4, BF_STATUS_UNSUPPORTED_SHAPE);
    }

    BF_ASSERT(out->shape[2] == plan->gridsize(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[3] == plan->gridsize(), BF_STATUS_INVALID_SHAPE);    
    BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_TRY_RETURN(plan->execute(in, out, nbatch, nchan, npol));
}

BFstatus bfxGridDestroy(BFxgrid plan) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    delete plan;
    return BF_STATUS_SUCCESS;
}
