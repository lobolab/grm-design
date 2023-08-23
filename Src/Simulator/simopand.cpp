// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "simopand.h"
#include "Common/mathalgo.h"

namespace LoboLab {

SimOpAnd::SimOpAnd(const double *f1, double dc, double hc, double *to)
  : from1_(f1), disConst_(dc), hillCoef_(hc), to_(to) {
}

SimOpAnd::~SimOpAnd() {
}

void SimOpAnd::compute() const {
  double rcn = pow(*from1_ * disConst_, hillCoef_);
  *to_ *= rcn;
}

}