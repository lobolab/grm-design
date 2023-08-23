// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#pragma once

#include <QVector>

namespace LoboLab {

class Search;
class Individual;

class ErrorCalculator {

 public:
  ErrorCalculator();
  virtual ~ErrorCalculator();

  virtual void process(int nDeme, const QVector<Individual*> &individuals) = 0;
  virtual int waitForAnyDeme() = 0;
};

} // namespace LoboLab
