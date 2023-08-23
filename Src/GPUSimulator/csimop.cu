// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "csimop.cuh"

#include <stdio.h>

//#include "device_launch_parameters.h"

namespace LoboLab {

  __host__ __device__ CSimOp::CSimOp() {}

  __host__ __device__ CSimOp::CSimOp(FuncType funcType, int to, int from, double dc, double hc)
    : op_func_(NULL), funcType_(funcType), from_{ from }, to_{ to }, hillCoef_(hc), disConstN_(dc) {}

  __device__ void CSimOp::compute() {
      op_func_(this);
  }

  __device__ void CSimOp::and_func(CSimOp *op) {
    double rcn = CSimOp::powTrick((op->from_.pointer[threadIdx.x] * op->disConstN_), op->hillCoef_);
    op->to_.pointer[threadIdx.x] *= rcn;
  }


  __device__ void CSimOp::or_func(CSimOp *op) {
    double rcn = CSimOp::powTrick((op->from_.pointer[threadIdx.x] * op->disConstN_), op->hillCoef_);
    op->to_.pointer[threadIdx.x] += (1 + op->to_.pointer[threadIdx.x]) * rcn;
  }

  __device__ void CSimOp::zero_func(CSimOp *op) {
    op->to_.pointer[threadIdx.x] = 0.0;
  }

  __device__ void CSimOp::one_func(CSimOp *op) {
    op->to_.pointer[threadIdx.x] = 1.0;
  }
   
  __device__ void CSimOp::sum_func(CSimOp *op) {
    op->to_.pointer[threadIdx.x] += 0.01;
  }

  __device__ void CSimOp::div_func(CSimOp *op) {
    double rcn = CSimOp::powTrick((op->from_.pointer[threadIdx.x] * op->disConstN_), op->hillCoef_);
    op->to_.pointer[threadIdx.x] /= (1+rcn);
  }

  __device__ void CSimOp::linkFuncPointer(double *ratios, double *oldConc, int nThreads) {

    switch (funcType_) {
    case OpAnd:
      op_func_ = and_func;
      from_.pointer = &oldConc[from_.index*nThreads];
      to_.pointer = &ratios[to_.index*nThreads];
      break;
    case OpOr:
      op_func_ = or_func;
      from_.pointer = &oldConc[from_.index*nThreads];
      to_.pointer = &ratios[to_.index*nThreads];
      break;
    case OpZero:
      op_func_ = zero_func;
      to_.pointer = &ratios[to_.index*nThreads];
      break;
    case OpOne:
      op_func_ = one_func;
      to_.pointer = &ratios[to_.index*nThreads];
      break;
    case OpSum:
      op_func_ = sum_func;
      to_.pointer = &ratios[to_.index*nThreads];
      break;
    case OpDiv:
      op_func_ = div_func;
      from_.pointer = &oldConc[from_.index*nThreads];
      to_.pointer = &ratios[to_.index*nThreads];
      break;
    }
  }
}
