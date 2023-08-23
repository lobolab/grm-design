// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#pragma once
#include <cuda_runtime.h>

namespace LoboLab {
  // GPU DEVICE PARAMETERS
#define NTHREADS 1024
#define MAXSHAREDMEM 49152

struct CExperimentsData;
struct CModelData;
struct CSimTempData;

void launchKernel(CExperimentsData *cExperimentsDataDevice, CModelData *cModelDataDevice, CSimTempData *cSimTempDataDevice_, int nProducts, cudaStream_t stream, double *gpu_answer, long long int *gpu_time);

}
