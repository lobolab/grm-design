// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#pragma once

#include "formulawidget.h"

#include <QMenu>

namespace LoboLab {

class Model;
class ModelProd;
class ModelLink;

class ModelFormulaWidget : public FormulaWidget {
  Q_OBJECT

  public:
    ModelFormulaWidget(Model *m, int nInputMorphogens, int nTargetMorphogens,
      QWidget * parent = NULL);
    virtual ~ModelFormulaWidget();

    void updateFormula(int nInputMorphogens, int nTargetMorphogens);

  signals:
    void modified();

  private:
    QString createHillFormula(const ModelLink *link, const QString &regulator) const;
    static bool linkRegulatorLessThan(ModelLink *l1, ModelLink *l2);

    Model *model_;
};

} // namespace LoboLab
