// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "simopone.h"

namespace LoboLab {

SimOpOne::SimOpOne(double *to)
  : to_(to) {
}

SimOpOne::~SimOpOne() {
}

void SimOpOne::compute() const {
  *to_ = 1;
}

}