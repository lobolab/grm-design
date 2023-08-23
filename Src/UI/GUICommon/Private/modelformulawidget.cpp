// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "modelformulawidget.h"

#include "colorsconfig.h"
#include "Model/model.h"
#include "Model/modelprod.h"
#include "Model/modellink.h"
#include "Common/mathalgo.h"

#include <QAction>
#include <QHeaderView>
#include <QInputDialog>
#include <QMessageBox>
#include <QMenu>

namespace LoboLab {

ModelFormulaWidget::ModelFormulaWidget(Model *m, int nInputMorphogens, int nTargetMorphogens, QWidget * parent)
  : FormulaWidget(parent),
  model_(m) {
  updateFormula(nInputMorphogens, nTargetMorphogens);
}

ModelFormulaWidget::~ModelFormulaWidget() {
}

void ModelFormulaWidget::updateFormula(int nInputMorphogens, int nTargetMorphogens) {
  QString mathMLStr("<math xmlns='http://www.w3.org/1998/Math/MathML'>");
  QList<int> usedLabels = model_->calcProductLabelsInUse(nInputMorphogens + nTargetMorphogens).toList();
  qSort(usedLabels);
  int nProducts = usedLabels.size();

  QStringList names;
  names.append("Red");
  names.append("Green");
  names.append("Blue");

  int n = nProducts - names.size();
  for (int i = 0; i < n; ++i)
    names.append(QChar(i + 97));

  mathMLStr += "<mtable>";
  for (int i = nInputMorphogens; i < nProducts; ++i) {
    ModelProd *prod = model_->prodWithLabel(usedLabels[i]);

    mathMLStr += "<mtr><mtd columnalign='right'>";
    mathMLStr += QString("<mfrac><mrow><mo>&#x2202;</mo><mi mathvariant='italic'>%1</mi></mrow><mrow><mo>&#x2202;</mo><mi>t</mi></mrow></mfrac>").arg(names.at(i));
    mathMLStr += "</mtd><mtd columnalign='center'><mo>=</mo></mtd><mtd columnalign='left'>";

    // links
    QVector<ModelLink*> links = model_->linksToLabel(prod->label());
    qSort(links.begin(), links.end(), linkRegulatorLessThan);
    int nLinks = links.size();

    if (nLinks > 0) {
      QVector<ModelLink*> orLinks, andLinks;
      bool anyActivator = false;
      for (int j = 0; j < nLinks; ++j) {
        ModelLink *link = links[j];
        if (usedLabels.contains(link->regulatorProdLabel())) {
          if (link->isAndReg())
            andLinks.append(link);
          else
            orLinks.append(link);

          if (link->hillCoef() >= 0)
            anyActivator = true;
        }
      }

      int nOrLinks = orLinks.size();
      int nAndLinks = andLinks.size();
      bool activatorTempUsed = false;
      QString formulaStr;
      QString hillStr;
      mathMLStr += QString("<mn>%1</mn>").arg(prod->lim(), 0, 'g', 2);
      mathMLStr += QString("<mo>&#x22c5;</mo>");
      mathMLStr += "<mfrac>";
      if (anyActivator) {
        for (int j = 0; j < nOrLinks; ++j) {
          ModelLink *link = orLinks[j];
          if (link->hillCoef() >= 0) {
            hillStr = createHillFormula(link,
              names[usedLabels.indexOf(link->regulatorProdLabel())]);
            if (!activatorTempUsed) {
              formulaStr = "<mrow>" + hillStr + "</mrow>";
              activatorTempUsed = true;
            }
            else
              formulaStr = "<mrow><mfenced><mrow>" + formulaStr + "<mo>&#x22c5;</mo>" + hillStr + "<mo>+</mo>" + formulaStr + "<mo>+</mo>" + hillStr + "</mrow></mfenced></mrow>";
          }
        }

        for (int j = 0; j < nAndLinks; ++j) {
          ModelLink *link = andLinks[j];
          if (link->hillCoef() >= 0) {
            hillStr = createHillFormula(link,
              names[usedLabels.indexOf(link->regulatorProdLabel())]);
            if (!activatorTempUsed) {
              formulaStr = "<mrow>" + hillStr + "</mrow>";
              activatorTempUsed = true;
            }
            else
              formulaStr = "<mrow>" + formulaStr + "<mo>&#x22c5;</mo>" + hillStr + "</mrow>";
          }
        }
        if (nLinks == 1)
          mathMLStr += "<mrow>" + formulaStr + "</mrow>";
        else
          mathMLStr += formulaStr ;
      }
      else {
        mathMLStr += QString("<mo>1</mo>");
      }

      for (int j = 0; j < nOrLinks; ++j) {
        hillStr = createHillFormula(orLinks[j],
          names[usedLabels.indexOf(orLinks[j]->regulatorProdLabel())]);
        if (j==0)
          formulaStr = "<mrow><mfenced><mrow>" + QString("<mo>1</mo>") + "<mo>+</mo>" + hillStr + "</mrow></mfenced></mrow>";
        else
          formulaStr = "<mrow>" + formulaStr + "<mo>&#x22c5;</mo><mfenced><mrow>" + QString("<mo>1</mo>") + "<mo>+</mo>" + hillStr + "</mrow></mfenced></mrow>";
      }

      for (int j = 0; j < nAndLinks; ++j) {
        hillStr = createHillFormula(andLinks[j],
          names[usedLabels.indexOf(andLinks[j]->regulatorProdLabel())]);
        if (j == 0 && nOrLinks == 0)
          formulaStr = "<mrow><mfenced><mrow>" + QString("<mo>1</mo>") + "<mo>+</mo>" + hillStr + "</mrow></mfenced></mrow>";
        else
          formulaStr = "<mrow>" + formulaStr + "<mo>&#x22c5;</mo><mfenced><mrow>" + QString("<mo>1</mo>") + "<mo>+</mo>" + hillStr + "</mrow></mfenced></mrow>";
      }

      mathMLStr += formulaStr;
      mathMLStr += "</mfrac>";
    }

    mathMLStr += QString("<mo>-</mo><mn>%1</mn><mo>&#x22c5;</mo><mi mathvariant='italic'>%2</mi>").arg(prod->deg(), 0, 'g', 2).arg(names.at(i));

    if (prod->dif() > 0)
      mathMLStr += QString("<mo>+</mo><mn>%1</mn><mo>&#x22c5;</mo><msup><mo>&#x2207;</mo><mn>2</mn></msup><mi mathvariant='italic'>%2</mi>").arg(prod->dif(), 0, 'f', 0).arg(names.at(i));

    mathMLStr += "</mtd></mtr>";
  }

  mathMLStr += "</mtable>";
  mathMLStr += "</math>";
  setMathMLStr(mathMLStr);
}

QString ModelFormulaWidget::createHillFormula(const ModelLink *link,
  const QString &regulator) const {
  double exp = abs(link->hillCoef());
  QString regTerm = QString("<mn>%1</mn><mo>&#x22c5;</mo><msup><mi mathvariant='italic'>%2</mi>"
    "<mn>%3</mn></msup>").arg(link->disConst(), 0, 'g', 2).arg(regulator).arg(exp, 0, 'g', 2);

  return regTerm;
}

// For sorting
bool ModelFormulaWidget::linkRegulatorLessThan(ModelLink *l1, ModelLink *l2) {
  return (l1->regulatorProdLabel() < l2->regulatorProdLabel());
}

}