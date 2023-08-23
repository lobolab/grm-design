// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#pragma once

#include <QVector>
#include <QHash>
#include <QSize>

#include <cuda_runtime.h>

namespace LoboLab {
class SimState;
class SimOp;
class Model;
class ModelLink;
struct CExperimentsData;
struct CModelData;
struct CSimTempData;
struct CSimState;
class CSimOp;
  
class CModelSimulator {
  public:
	  CModelSimulator(int dimX, int dimY, int nTotalMorphogens, cudaStream_t stream);
    ~CModelSimulator();

    void loadModel(const Model &model, int nMorphogens);
    void clearAll();
    void clearLabels();
    void clearProducts();
    void allocateProducts(int nProducts);
    void clearOps();
    void allocateOps(int nOps);

    //void CModelSimulator::simulateManipulation(double distErrorThres, cudaStream_t stream,CudaDataParam* cudadataparams_);

    inline QVector<int> productLabels() const { return labels_; }

      double simulateExperiment(CExperimentsData *cExperimentsDataDevice_, double gpuDeviceId, double maxError, double *simTime);

  private:
	  CModelSimulator(const CModelSimulator &source);
   
    QVector<CSimOp> createProductOps(int p, const QVector<ModelLink*> &orLinks,
                                const QVector<ModelLink*> &andLinks);
    // CSimOp createHillOpForLink(ModelLink *link, double to) const;


    cudaStream_t stream_;

    CModelData *cModelDataHostDevice_;

    CSimTempData *cSimTempDataHost_;
    CSimTempData *cSimTempDataDevice_;

    double *cErrorHostDevice_;
    long long int *cTimeHostDevice_;

    QVector<int> labels_;
    QHash<int, int> labelsInd_;

    int dimX_;
    int dimY_;
    int nAllocatedProducts_;
    int nAllocatedOps_;
  };
}