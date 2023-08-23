// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "Shapeerrorform.h"
#include "Experiment/experiment.h"
#include "Experiment/morphologyimage.h" 
#include "Private/morphologyimagewidget.h"
#include "Simulator/simulator.h"
#include "Simulator/simparams.h"
#include "DB/db.h"
#include "Model/model.h"
#include "Simulator/simstate.h"
#include "Search/search.h"
#include "UI/GUICommon/imagesaver.h"
#include "UI/GUICommon/Simulator/imageview.h"
#include <QDataWidgetMapper>
#include <QGridLayout>
#include <QPushButton>
#include <QInputDialog>
#include <QKeyEvent>
#include <QMessageBox>
#include <QSettings>
#include <QSplitter>
#include <QFormLayout>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QAction>
#include <QFileDialog>
#include <qwt_plot_renderer.h>

namespace LoboLab {

ShapeErrorForm::ShapeErrorForm(const Search &search, const Model &model, QWidget *parent)
  :QDialog(parent), search_(search), model_(model), simulator_(NULL), state_(NULL) {
  
  experiment_ = search_.experiment(0);

  if (experiment_->id()) {
    setWindowTitle(tr("Shape Error"));
    originalName_ = experiment_->name();
  }
  else
    setWindowTitle(tr("Shape Error"));

  createWidgets();
  readSettings();
  createActions();

  QTimer::singleShot(0, this, SLOT(run()));

}

ShapeErrorForm::~ShapeErrorForm() {
  writeSettings();

  delete simulator_;

}

void ShapeErrorForm::readSettings() {
  QSettings settings;
  settings.beginGroup("ShapeErrorForm");
  resize(1200, 500);
  if (settings.value("maximized", false).toBool())
    showMaximized();
}

void ShapeErrorForm::writeSettings() {
  QSettings settings;
  settings.beginGroup("ShapeErrorForm");
  if (isMaximized())
    settings.setValue("maximized", isMaximized());
  else {
    settings.setValue("maximized", false);
    settings.setValue("size", size());
  }
  settings.endGroup();
}

void ShapeErrorForm::formAccepted() {
    accept();
}

void ShapeErrorForm::createWidgets() {
 
  setWindowFlags(Qt::Window);


  nameEdit_ = new QLineEdit();
  nameEdit_->setText(experiment_->name());
  QLabel *nameLabel = new QLabel("Name:");
  nameLabel->setBuddy(nameEdit_);
  targetLabel_ = new QLabel(this);
  targetLabel_->setMinimumSize(300,300);
  targetLabel_->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

  developedLabel_ = new QLabel(this);
  developedLabel_->setMinimumSize(300, 300);
  developedLabel_->width();
  developedLabel_->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

  shapeErrLabel_ = new QLabel(this);
  shapeErrLabel_->setMinimumSize(300, 300);
  shapeErrLabel_->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

  blurTargetLabel_ = new QLabel(this);
  blurTargetLabel_->setMinimumSize(300, 300);
  blurTargetLabel_->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

  blurDevelopedLabel_ = new QLabel(this);
  blurDevelopedLabel_->setMinimumSize(300, 300);
  blurDevelopedLabel_->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

  blurShapeErrLabel_ = new QLabel(this);
  blurShapeErrLabel_->setMinimumSize(300, 300);
  blurShapeErrLabel_->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

  QGroupBox *targetImageBox = new QGroupBox("Target Pattern");
  QGroupBox *developedImageBox = new QGroupBox("Developed Pattern");
  QGroupBox *shapeErrImageBox = new QGroupBox("Shape Error Pattern");
  QGroupBox *blurTargetImageBox = new QGroupBox("Blur Target Pattern");
  QGroupBox *blurDevelopedImageBox = new QGroupBox("Blur Developed Pattern");
  QGroupBox *blurShapeErrImageBox = new QGroupBox("Blur Shape Error Pattern");

  QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok |
    QDialogButtonBox::Cancel);
  connect(buttonBox, SIGNAL(accepted()), this, SLOT(formAccepted()));
  connect(buttonBox, SIGNAL(rejected()), this, SLOT(reject()));

  // Layouts

  QHBoxLayout *nameLay = new QHBoxLayout();
  nameLay->addWidget(nameLabel);
  nameLay->addWidget(nameEdit_);

  QGridLayout *targetImageLayout = new QGridLayout;
  targetImageLayout->addWidget(targetLabel_);
  targetImageBox->setLayout(targetImageLayout);
  connect(targetImageBox, SIGNAL(clicked()), this, SLOT(createActions()));

  QGridLayout *developedImageLayout = new QGridLayout;
  developedImageLayout->addWidget(developedLabel_);
  developedImageBox->setLayout(developedImageLayout);

  QGridLayout *shapeErrImageLayout = new QGridLayout;
  shapeErrImageLayout->addWidget(shapeErrLabel_);
  shapeErrImageBox->setLayout(shapeErrImageLayout);

  QGridLayout *blurTargetImageLayout = new QGridLayout;
  blurTargetImageLayout->addWidget(blurTargetLabel_);
  blurTargetImageBox->setLayout(blurTargetImageLayout);

  QGridLayout *blurDevelopedImageLayout = new QGridLayout;
  blurDevelopedImageLayout->addWidget(blurDevelopedLabel_);
  blurDevelopedImageBox->setLayout(blurDevelopedImageLayout);

  QGridLayout *blurShapeErrImageLayout = new QGridLayout;
  blurShapeErrImageLayout->addWidget(blurShapeErrLabel_);
  blurShapeErrImageBox->setLayout(blurShapeErrImageLayout);

  QHBoxLayout *imagelayout = new QHBoxLayout();
  imagelayout->addWidget(targetImageBox);
  imagelayout->addWidget(developedImageBox);
  imagelayout->addWidget(shapeErrImageBox);

  QHBoxLayout *blurImagelayout = new QHBoxLayout();
  blurImagelayout->addWidget(blurTargetImageBox);
  blurImagelayout->addWidget(blurDevelopedImageBox);
  blurImagelayout->addWidget(blurShapeErrImageBox);

  QVBoxLayout *mainLay = new QVBoxLayout();
  mainLay->addLayout(nameLay);
  mainLay->addSpacing(10);
  mainLay->addLayout(imagelayout);
  mainLay->addSpacing(10);
  mainLay->addLayout(blurImagelayout);
  mainLay->addSpacing(10);
  mainLay->addWidget(buttonBox);
  setMinimumSize(1200,900);

  setLayout(mainLay);
}

void ShapeErrorForm::createActions() {

  setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  QAction *saveImageAction = new QAction(tr("Save image..."), this);
  saveImageAction->setIcon(QIcon(":/Images/famfamfam_silk_icons/picture_save.png"));
  connect(saveImageAction, SIGNAL(triggered()), this, SLOT(saveImage()));

  QAction *deleteImageAction = new QAction(tr("Delete image"), this);
  deleteImageAction->setIcon(QIcon(":/Images/famfamfam_silk_icons/picture_delete.png"));
  connect(deleteImageAction, SIGNAL(triggered()), this, SLOT(deleteImage()));

  setContextMenuPolicy(Qt::ActionsContextMenu);
  addAction(saveImageAction);
  addAction(deleteImageAction);

}
// private slot
void ShapeErrorForm::nameChanged() {
  experiment_->setName(nameEdit_->text());
}

void ShapeErrorForm::run() {

  SimParams *simParams = search_.simParams();

  simulator_ = new Simulator(search_);
  nInputMorphogens_ = simParams->nInputMorphogens;
  nTargetMorphogens_ = simParams->nTargetMorphogens;
  kernelSize_ = simParams->kernel;

  state_ = new SimState(simParams->size,
    simParams->nInputMorphogens, simParams->nTargetMorphogens, simParams->distErrorThreshold, simParams->kernel);
  state_->loadMorphologyImage(*experiment_->outputMorphology(), simParams->nTargetMorphogens);
  //state_->preprocessOutputMorphology(*experiment_->outputMorphology(), simParams->nTargetMorphogens);
  simulator_->loadModel(&model_);
  simulator_->loadExperiment(experiment_);
  simulator_->initialize();
  size_ = simulator_->domainSize();

  shapErrImage_ = new QImage(size_, QImage::Format_ARGB32);
  targetImage_ = new QImage(size_, QImage::Format_ARGB32);
  developedImage_ = new QImage(size_, QImage::Format_ARGB32);

  blurShapErrImage_ = new QImage(size_, QImage::Format_ARGB32);
  blurTargetImage_ = new QImage(size_, QImage::Format_ARGB32);
  blurDevelopedImage_ = new QImage(size_, QImage::Format_ARGB32);

  simulator_->simulate(simParams->NumSimSteps);
  calcImage();
  updateLabels();

}

void ShapeErrorForm::calcImage() {
  
  const SimState &simState = simulator_->simulatedState();
  // static int kernelSize2 = kernelSize_*kernelSize_;
  int radius = kernelSize_ / 2;;
  int lowerBound = -(kernelSize_ / 2) + (kernelSize_ % 2 == 0);  // Second part is to check if the kernel size is even or odd
  int upperBound = kernelSize_ / 2;

  for (int j = 0; j < size_.height(); ++j) {
    for (int i = 0; i < size_.width(); ++i) {
      int red = 0, green = 0, blue = 0;
      
      double a = simState.product(2)(i, j);
      double b = state_->product(0)(i, j);
      double ka = 0.0;
      double kb = 0.0;

      int count = 0;
      int minI = MathAlgo::max(0, i + lowerBound);
      int maxI = MathAlgo::min(size_.width() - 1, i + upperBound);
      int minJ = MathAlgo::max(0, j + lowerBound);
      int maxJ = MathAlgo::min(size_.height() - 1, j + upperBound);
      for (int ki = minI; ki <= maxI; ++ki) {
        for (int kj = minJ; kj <= maxJ; ++kj) {
          ka += simState.product(2)(ki, kj);
          kb += state_->product(0)(ki, kj);
          count++;
        }
      }

      ka = ka/count;
      kb = kb/count;

      ((QRgb*) targetImage_->scanLine(j))[i] = qRgb(0, 0, 255 * b); //qRgb(0, 0, 255*b);
      ((QRgb*) blurTargetImage_->scanLine(j))[i] = qRgb(0, 0, 255*kb);

      if (a >= 1)
      {
        ((QRgb*) developedImage_->scanLine(j))[i] = qRgb(0, 0, 255);
        ((QRgb*) blurDevelopedImage_->scanLine(j))[i] = qRgb(0, 0, 255);
      }
      else
      {
        ((QRgb*) developedImage_->scanLine(j))[i] = qRgb(0, 0, 255 * a);
        ((QRgb*) blurDevelopedImage_->scanLine(j))[i] = qRgb(0, 0, 255 * ka);
      }

      double kernelErr = kb - ka;
      if (kernelErr >= 0)
        green = 255 * kernelErr;
      else
        red = 255 * fabs(kernelErr);

      ((QRgb*) blurShapErrImage_->scanLine(j))[i] = qRgb(red, green, blue);

      red = 0, green = 0;
      double cellErr = b - a;
      if (cellErr >= 0)
        green = 255 * cellErr;
      else 
        red = 255 * fabs(cellErr);
    
      ((QRgb*) shapErrImage_->scanLine(j))[i] = qRgb(red, green, blue);
    }
  }
}

void ShapeErrorForm::updateLabels() {

  if (targetLabel_->width() >= targetLabel_->height()) {
    targetLabel_->setPixmap(QPixmap::fromImage(*targetImage_).scaled(targetLabel_->height(), targetLabel_->height()));
    developedLabel_->setPixmap(QPixmap::fromImage(*developedImage_).scaled(targetLabel_->height(), targetLabel_->height()));
    shapeErrLabel_->setPixmap(QPixmap::fromImage(*shapErrImage_).scaled(targetLabel_->height(), targetLabel_->height()));
    blurTargetLabel_->setPixmap(QPixmap::fromImage(*blurTargetImage_).scaled(targetLabel_->height(), targetLabel_->height()));
    blurDevelopedLabel_->setPixmap(QPixmap::fromImage(*blurDevelopedImage_).scaled(targetLabel_->height(), targetLabel_->height()));
    blurShapeErrLabel_->setPixmap(QPixmap::fromImage(*blurShapErrImage_).scaled(targetLabel_->height(), targetLabel_->height()));
  }
  else
  {
    targetLabel_->setPixmap(QPixmap::fromImage(*targetImage_).scaled(targetLabel_->width(), targetLabel_->width()));
    developedLabel_->setPixmap(QPixmap::fromImage(*developedImage_).scaled(targetLabel_->width(), targetLabel_->width()));
    shapeErrLabel_->setPixmap(QPixmap::fromImage(*shapErrImage_).scaled(targetLabel_->width(), targetLabel_->width()));
    blurTargetLabel_->setPixmap(QPixmap::fromImage(*blurTargetImage_).scaled(targetLabel_->width(), targetLabel_->width()));
    blurDevelopedLabel_->setPixmap(QPixmap::fromImage(*blurDevelopedImage_).scaled(targetLabel_->width(), targetLabel_->width()));
    blurShapeErrLabel_->setPixmap(QPixmap::fromImage(*blurShapErrImage_).scaled(targetLabel_->width(), targetLabel_->width()));
  }

  targetLabel_->show();
  developedLabel_->show();
  shapeErrLabel_->show();
  blurTargetLabel_->show();
  blurDevelopedLabel_->show();
  blurShapeErrLabel_->show();

}

void ShapeErrorForm::resizeEvent(QResizeEvent *event) {

  if (simulator_) {

    updateLabels();
    
    QDialog::resizeEvent(event);

  }

}

// private slot
void ShapeErrorForm::saveImage() {
  ImageSaver::savePixmap(createPixmap(), this);
}

QPixmap ShapeErrorForm::createPixmap() {
  QPixmap pixmap(size());
  pixmap.fill(Qt::transparent);

  bool prevUseSelecLines = useSelecLines_;
  useSelecLines_ = false;
  drawBorder_ = false;
  render(&pixmap);
  drawBorder_ = true;
  useSelecLines_ = prevUseSelecLines;

  return pixmap;
}

// private slot
void ShapeErrorForm::deleteImage() {
  setVisible(false);
  emit deleted();
}

void ShapeErrorForm::keyPressEvent(QKeyEvent *e) {
  if (e->key() != Qt::Key_Escape)
    QDialog::keyPressEvent(e);
}

}
