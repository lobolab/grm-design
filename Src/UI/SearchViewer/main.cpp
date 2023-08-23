// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "mainwindow.h"

#include <QApplication>

#include <qwt_text.h>
#include <qwt_mathml_text_engine.h>

#ifdef QT_DEBUG
#ifndef Q_WS_X11
#endif
#endif

int main(int argc, char *argv[]) {

  QApplication a(argc, argv);

#ifdef QT_DEBUG
#ifndef Q_WS_X11
 
#endif
#endif

  // This needs to be done only once before using the MathML engine
  QwtText::setTextEngine(QwtText::MathMLText, new QwtMathMLTextEngine());

  LoboLab::MainWindow mw;
  mw.show();

  return a.exec();
}
