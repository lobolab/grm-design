// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#pragma once

#include <QVector>

namespace LoboLab {

class Search;
class Model;
class Simulator;
class SimState;

class EvaluatorMorphogens {

 public:
  explicit EvaluatorMorphogens(const Search &search);
  ~EvaluatorMorphogens();

  double evaluate(const Model &model, double maxError);

 private:
  const Search &search_;
  Simulator *simulator_;
  QVector<SimState*> targetStates_;

  int nSims_;
  int nExperiments_;
  double minConcChange_;
  int errorPrecision_;
  bool manipulationSim_;
};

} // namespace LoboLab
