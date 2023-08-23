// Copyright (c) Lobo Lab (lobo@umbc.edu)
// All rights reserved.

#include "simulator.h"

#include "simparams.h"
#include "Experiment/experiment.h"
#include "Search/search.h"

namespace LoboLab {

Simulator::Simulator(const Search &search)
  : search_(search),
  modelSimulator_(search_.simParams()->dt,
    search_.simParams()->size.width(),
    search_.simParams()->size.height(),
    search_.simParams()->minConc),
  initialState_(modelSimulator_.domainSize(),
    search_.simParams()->nInputMorphogens,
    search_.simParams()->nTargetMorphogens,
    search_.simParams()->distErrorThreshold,
    search_.simParams()->kernel),
  simulatedState_(modelSimulator_.domainSize(),
    search_.simParams()->nInputMorphogens,
    search_.simParams()->nTargetMorphogens,
    search_.simParams()->distErrorThreshold,
    search_.simParams()->kernel),
  experiment_(NULL),
  model_(NULL),
  currStep_(0),
  minConcChange_(search_.simParams()->minConcChange) {
}

Simulator::~Simulator() {
}

void Simulator::loadModel(const Model *model) {
  model_ = model;
  int nMorphogens = search_.simParams()->nInputMorphogens +
    search_.simParams()->nTargetMorphogens;
  modelSimulator_.loadModel(*model_, nMorphogens);
  initialState_.initialize(modelSimulator_.nProducts());
}

void Simulator::loadExperiment(const Experiment *exp) {
  experiment_ = exp;
  initialState_.loadGradient();
}

void Simulator::initialize() {
  if (model_ && experiment_) {
    simulatedState_ = initialState_;
    currStep_ = 0;
  }
}

double Simulator::simulate(int nSteps) {
  double change = 1;
  int lastStep = currStep_ + nSteps;
  while (currStep_ < lastStep && change > minConcChange_) {
    change = modelSimulator_.simulateStep(&simulatedState_, &initialState_);
    ++currStep_;
  }

  return change;
}


}