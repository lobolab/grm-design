// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#pragma once

#include "simop.h"

namespace LoboLab {

class SimOpSum : public SimOp {
public:
  SimOpSum(double *from2AndTo);
  
  ~SimOpSum();

  void compute() const;

private:
  double *to_;
};

} // namespace LoboLab
