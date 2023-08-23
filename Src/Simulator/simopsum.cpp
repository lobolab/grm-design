// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "simopsum.h"
#include "Common/mathalgo.h"

namespace LoboLab {

SimOpSum::SimOpSum(double *to)
  : to_(to) {
}

SimOpSum::~SimOpSum() {
}

void SimOpSum::compute() const {
  *to_ += 0.01;
}

}