// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#pragma once

#include <QVector>

namespace LoboLab {

class SearchParams;
class SimParams;
class Search;
class Generation;
class Deme;
class Individual;
class DB;

class SearchAlgoDetCrowd {
 public:
  SearchAlgoDetCrowd(Deme *deme, Search *s);
  ~SearchAlgoDetCrowd(void);
  
  inline Deme *deme() const { return deme_; }
  inline Generation *currentGeneration() const { return generation_; }

  QVector<Individual*> calcInitialPopulation();
  const QVector<Individual*> &reproduce();
  void chooseNextGeneration();

 private:
  Individual *newRandIndividual(int nProducts, int nMorphogens) const;
  //void copyPopulation();
  
  Search *search_;
  Deme *deme_;

  SearchParams *searchParams_;
  SimParams *simParams_;
  Generation *generation_;
  QVector<Individual*> children_;
  int populationSize_;
  int *randPopulationInd_;
  int nMorphogens_;
  int nMinProducts_; 
  int nMaxProducts_;
  int nMaxLinks_;
  int nInputMorphogens_;
  int nTargetMorphogens_;
};

} // namespace LoboLab
