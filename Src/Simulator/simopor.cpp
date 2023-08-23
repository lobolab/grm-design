// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "simopor.h"
#include "Common/mathalgo.h"

namespace LoboLab {

SimOpOr::SimOpOr(const double *f1, double dc, double hc, double *to)
  : from1_(f1), disConst_(dc), hillCoef_(hc), to_(to) {
}

SimOpOr::~SimOpOr() {
}

void SimOpOr::compute() const {
  double rcn = pow(*from1_ * disConst_, hillCoef_);
  *to_ += (1 + *to_) * rcn;
}

}