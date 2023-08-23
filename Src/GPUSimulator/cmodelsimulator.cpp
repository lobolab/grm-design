// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "cmodelsimulator.h"

#include "Model/model.h"
#include "Model/modelprod.h"
#include "Model/modellink.h"
#include "Simulator/simulatorconfig.h"
#include "Simulator/simstate.h"
#include "csimstate.cuh"
#include "csimop.cuh"
#include "cerrorhandle.h"
#include "modelsimulatordevice.cuh"
#include "cexperimentsdata.h"
#include "cmodeldata.h"
#include "csimtempdata.h"

#include <QElapsedTimer>

namespace LoboLab {

CModelSimulator::CModelSimulator(int dimX, int dimY, int nTotalMorphogens, 
                                  cudaStream_t stream)
  : dimX_(dimX), dimY_(dimY), stream_(stream), nAllocatedProducts_(0), 
    nAllocatedOps_(0) {

  cErrorHandle(cudaHostAlloc(&cModelDataHostDevice_, sizeof(CModelData), cudaHostAllocMapped));

  cErrorHandle(cudaMallocHost(&cSimTempDataHost_, sizeof(CSimTempData)));
  cSimTempDataHost_->cSimState1.initialize(dimX_, dimY_, 0);
  cSimTempDataHost_->cSimState2.initialize(dimX_, dimY_, 0);
  
  cErrorHandle(cudaMalloc(&cSimTempDataDevice_, sizeof(CSimTempData)));
  // Mapped memory to avoid a copy of a single double; improves performance notably
  cErrorHandle(cudaHostAlloc(&cErrorHostDevice_, sizeof(double), cudaHostAllocMapped));
  cErrorHandle(cudaHostAlloc(&cTimeHostDevice_, sizeof(long long int), cudaHostAllocMapped));

  // Default allocation
  // Here to avoid mem allocations during first kernels
  const int initialNumProducts = 4;
  const int initialNumOps = 10;

  allocateProducts(initialNumProducts);
  allocateOps(initialNumOps);
  
}

CModelSimulator::~CModelSimulator() {
  clearAll();

  cErrorHandle(cudaFreeHost(cModelDataHostDevice_));

  cErrorHandle(cudaFree(cSimTempDataDevice_));
  cSimTempDataHost_->cSimState1.freeMatrix();
  cSimTempDataHost_->cSimState2.freeMatrix();
  cErrorHandle(cudaFreeHost(cSimTempDataHost_));

  cErrorHandle(cudaFreeHost(cErrorHostDevice_));
  cErrorHandle(cudaFreeHost(cTimeHostDevice_));
}

void CModelSimulator::clearAll() {
  clearLabels();
  clearProducts();
  clearOps();
}


void CModelSimulator::clearLabels() {
  labels_.clear();
  labelsInd_.clear();
}

void CModelSimulator::clearProducts() {
  cErrorHandle(cudaFreeHost(cModelDataHostDevice_->limits));
  cErrorHandle(cudaFreeHost(cModelDataHostDevice_->degradations));
  cErrorHandle(cudaFreeHost(cModelDataHostDevice_->difProdInd));
  cErrorHandle(cudaFreeHost(cModelDataHostDevice_->difConsts));

  cErrorHandle(cudaFree(cSimTempDataHost_->ratios));
  cErrorHandle(cudaFree(cSimTempDataHost_->oldConcs));
    
  nAllocatedProducts_ = 0;
}

void CModelSimulator::allocateProducts(int nProducts) {
  cErrorHandle(cudaHostAlloc(&cModelDataHostDevice_->limits, sizeof(double)*nProducts, cudaHostAllocMapped));
  cErrorHandle(cudaHostAlloc(&cModelDataHostDevice_->degradations, sizeof(double)*nProducts, cudaHostAllocMapped));
  cErrorHandle(cudaHostAlloc(&cModelDataHostDevice_->difProdInd, sizeof(double)*nProducts, cudaHostAllocMapped));
  cErrorHandle(cudaHostAlloc(&cModelDataHostDevice_->difConsts, sizeof(double)*nProducts, cudaHostAllocMapped));

  cSimTempDataHost_->cSimState1.resize(nProducts);
  cSimTempDataHost_->cSimState2.resize(nProducts);

  cErrorHandle(cudaMalloc(&cSimTempDataHost_->ratios, sizeof(double)*NTHREADS*nProducts));
  cErrorHandle(cudaMalloc(&cSimTempDataHost_->oldConcs, sizeof(double)*NTHREADS*nProducts));

  cErrorHandle(cudaMemcpyAsync(cSimTempDataDevice_, cSimTempDataHost_, sizeof(CSimTempData), cudaMemcpyHostToDevice, stream_));

  nAllocatedProducts_ = nProducts;
}


void CModelSimulator::clearOps() {
  cErrorHandle(cudaFreeHost(cModelDataHostDevice_->ops));
  nAllocatedOps_ = 0;
}

void CModelSimulator::allocateOps(int nOps) {
  cErrorHandle(cudaHostAlloc(&cModelDataHostDevice_->ops, sizeof(CSimOp) * nOps, cudaHostAllocMapped));
  nAllocatedOps_ = nOps;
}

void CModelSimulator::loadModel(const Model &model, int nMorphogens) {
  clearLabels();
  QVector<int> labelSet = model.calcProductLabelsInUse(nMorphogens);
  labels_ = labelSet;
  qSort(labels_);
  int nProducts = labels_.size();
  cModelDataHostDevice_->nProducts = nProducts;

  labelsInd_ = QHash<int, int>();
  for (int i = 0; i < nProducts; ++i)
    labelsInd_[labels_.at(i)] = i;

  // Notice that reserving and freeing memory in the device prevents kernel 
  // concurrency.
  if (nProducts > nAllocatedProducts_) {
    clearProducts();
    allocateProducts(nProducts);
  }

  QVector< CSimOp > opsList; // Temporary storage for the operations

  cModelDataHostDevice_->nDif = 0;
  for (int i = 0; i < nProducts; ++i) {
    // Process product constants
    ModelProd *prod = model.prodWithLabel(labels_.at(i));
    cModelDataHostDevice_->limits[i] = prod->lim();
    cModelDataHostDevice_->degradations[i] = prod->deg();

    // Process product diffusion
    if (prod->dif() > 0) {
      cModelDataHostDevice_->difProdInd[cModelDataHostDevice_->nDif] = i;
      cModelDataHostDevice_->difConsts[cModelDataHostDevice_->nDif] = prod->dif();
      cModelDataHostDevice_->nDif++;
    }

    // Process product links
    QVector<ModelLink*> links = model.linksToLabel(labels_.at(i));
    QVector<ModelLink*> orLinks;
    QVector<ModelLink*> andLinks;

    // Categorize links
    int n = links.size();
    for (int j = 0; j < n; ++j) {
      ModelLink *link = links[j];
      if (labelSet.contains(link->regulatorProdLabel())) {
        if (link->isAndReg())
          andLinks.append(link);
        else
          orLinks.append(link);
      }
    }
    if (orLinks.isEmpty() && andLinks.isEmpty()) { // No links
      opsList.append(CSimOp(CSimOp::OpZero, i));
    }
    else
      opsList.append(createProductOps(i, orLinks, andLinks));
  }

  cModelDataHostDevice_->nOps = opsList.size();
  
  if (cModelDataHostDevice_->nOps > nAllocatedOps_) {
    clearOps();
    allocateOps(cModelDataHostDevice_->nOps);
  }

  for (int i = 0; i < cModelDataHostDevice_->nOps; ++i)
    cModelDataHostDevice_->ops[i] = opsList.at(i);
}

QVector<CSimOp> CModelSimulator::createProductOps(int p, const QVector<ModelLink*> &orLinks, const QVector<ModelLink*> &andLinks) {
  QVector<CSimOp> opsList;
  bool ratiosTempUsed = false;

  // Process OR links
  int n = orLinks.size();
  for (int i = 0; i < n; ++i) {
    ModelLink *link = orLinks[i];
    if (link->hillCoef() >= 0) {
      if (!ratiosTempUsed) {
        opsList.append(CSimOp(CSimOp::OpZero, p));
        ratiosTempUsed = true;
      }
      opsList.append(CSimOp(CSimOp::OpOr, p, labelsInd_[link->regulatorProdLabel()], link->disConst(), link->hillCoef()));
    }
  }

  // Process AND links
  n = andLinks.size();
  for (int i = 0; i < n; ++i) {
    ModelLink *link = andLinks[i];
    if (link->hillCoef() >= 0) {
      if (!ratiosTempUsed) {
        opsList.append(CSimOp(CSimOp::OpOne, p));
        ratiosTempUsed = true;
      }
      opsList.append(CSimOp(CSimOp::OpAnd, p, labelsInd_[link->regulatorProdLabel()], link->disConst(), link->hillCoef()));
    }
  }

  if (!ratiosTempUsed) // No activators
    opsList.append(CSimOp(CSimOp::OpOne, p));

  // Process division for And link
  n = andLinks.size();
  for (int i = 0; i < n; ++i) {
    ModelLink *link = andLinks[i];
    opsList.append(CSimOp(CSimOp::OpDiv, p, labelsInd_[link->regulatorProdLabel()], link->disConst(), fabs(link->hillCoef())));
  }

  // Process division for Or link
  n = orLinks.size();
  for (int i = 0; i < n; ++i) {
    ModelLink *link = orLinks[i];
    opsList.append(CSimOp(CSimOp::OpDiv, p, labelsInd_[link->regulatorProdLabel()], link->disConst(), fabs(link->hillCoef())));
  }

  return opsList;
}

double CModelSimulator::simulateExperiment(CExperimentsData *cExperimentsDataDevice_, double gpuDeviceId, double maxError, double *simTime) {
  cModelDataHostDevice_->maxError = maxError;
  //cErrorHandle(cudaMemcpyAsync(cModelDataDevice_, cModelDataHost_, sizeof(CModelData), cudaMemcpyHostToDevice, stream_));

  // default error, in case of problems
  *cErrorHostDevice_ = -1;
  *cTimeHostDevice_ = -1;


  launchKernel(cExperimentsDataDevice_, cModelDataHostDevice_, cSimTempDataDevice_, cModelDataHostDevice_->nProducts, stream_, cErrorHostDevice_, cTimeHostDevice_);

  cErrorHandle(cudaPeekAtLastError());
  cErrorHandle(cudaStreamSynchronize(stream_));

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpuDeviceId);
  long long int clockRate = deviceProp.clockRate * 1000; 

  *simTime = (double)*cTimeHostDevice_ / clockRate;
  return *cErrorHostDevice_;
}

}


