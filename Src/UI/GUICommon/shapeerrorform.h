// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#pragma once

#include <QDialog>
#include <QLineEdit>
#include <QLabel>
#include <QDialogButtonBox>
#include <QComboBox>
#include <QGroupBox>
#include <QSqlQueryModel>
#include <QHBoxLayout>
#include "Common/mathalgo.h"
#include <QImage>

namespace LoboLab {

class Experiment;
class Search;
class Simulator;
class SimState;
class Model;
class SimParams;
class DB;
class ImageView;

class ShapeErrorForm : public QDialog {
  Q_OBJECT

  public:
    ShapeErrorForm(const Search &search, const Model &model, QWidget *parent = NULL);
    virtual ~ShapeErrorForm();
    QPixmap createPixmap();
    public slots:

    private slots :
      void nameChanged();
      void run();
      void saveImage();
      void deleteImage();
    void createActions();

    void keyPressEvent(QKeyEvent *e);
    void formAccepted();

  signals:
    void deleted();

  protected:
    virtual void resizeEvent(QResizeEvent *event);

  private:
    void createWidgets();
    void readSettings();
    void writeSettings();
    void calcImage();
    void updateLabels();

    QLineEdit *nameEdit_;
    QLineEdit *nameInputImage_;
    QLineEdit *nameOutputImage_;
    QString originalName_;
    SimState *state_;
    Experiment *experiment_;
    DB *db_;
    Simulator *simulator_;
    int nInputMorphogens_;
    int nTargetMorphogens_;
    int kernelSize_;

    QLabel *targetLabel_;
    QLabel *developedLabel_;
    QLabel *shapeErrLabel_;
    QLabel *blurTargetLabel_;
    QLabel *blurDevelopedLabel_;
    QLabel *blurShapeErrLabel_;

    QSize size_;
    QImage *shapErrImage_;
    QImage *targetImage_;
    QImage *developedImage_;
    QImage *blurShapErrImage_;
    QImage *blurTargetImage_;
    QImage *blurDevelopedImage_;
    const Search &search_;
    const Model &model_;
    QPixmap pixmap_;
    bool useSelecLines_;
    bool drawingRect_;
    QPoint lastPoint_;
    QPoint currentPoint_;
    bool drawBorder_;
    
};

} // namespace LoboLab
