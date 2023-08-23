// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "simopdiv.h"
#include "Common/mathalgo.h"

namespace LoboLab {

SimOpDiv::SimOpDiv(const double *f1, double dc, double hc, double *to)
  : from1_(f1), disConst_(dc), hillCoef_(hc), to_(to) {
}

SimOpDiv::~SimOpDiv() {
}

void SimOpDiv::compute() const {
  double rcn = pow(*from1_ * disConst_, hillCoef_);
  *to_ /= (1 + rcn);
}

}