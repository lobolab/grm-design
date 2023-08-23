// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#pragma once

#include <QVector>
#include <QVector>
#include <QTextStream>

namespace LoboLab {

class ModelProd;
class ModelLink;

class Model {

 public:
  Model();
  ~Model();

  Model(const Model &source);
  Model &operator=(const Model &source);

  static Model *createRandom(int nProducts, int nMorphogens, int nInputMorphogens, int nTargetMorphogens);
  static void cross(const Model *parent1, const Model *parent2,
                    Model *&child1, Model *&child2, int nMorphogens,
                    int nMinProducts, int nMaxProducts, int nMaxLinks);

  QVector<int> calcProductLabels() const;
  QVector<int> calcProductLabelsInUse(int nMorphogens) const;

  inline int nProducts() const { return products_.size(); }
  inline ModelProd *product(int i) const { return products_.at(i); }
  ModelProd *prodWithLabel(int label, int *i = NULL) const;
  
  void addRandomProduct(int label, int nMorphogens, int nInputMorphogens);
  void duplicateProduct(int i, int nMorphogens, int nInputMorphogens, int nTargetMorphogens);
  void removeProduct(int i);
  void removeProductWithLabel(int label);

  inline int nLinks() const { return links_.size(); }
  inline ModelLink *link(int i) const { return links_.at(i); }
  inline QVector<ModelLink*> links() const { return links_; }
  QVector<ModelLink*> linksToLabel(int label) const;
  QVector<ModelLink*> calcLinksInUse(int nStrucLabels) const;
  int calcNLinksFromProd(int label) const;
  ModelLink *findLink(int regulator, int regulated) const;

  void addOrReplaceRandomLink();
  void addOrReplaceRandomLink(int regulatorLabel, int regulatedLabel);
  void duplicateLink(int i, int nMorphogens, int nInputMorphogens, int nTargetMorphogens);
  void removeLink(int i);
  void removeLink(ModelLink *modelLink);
  void removeLink(int regulatorLabel, int regulatedLabel);

  int calcComplexity() const;
  int calcComplexityInUse(int nStrucProd) const;

  void mutate(int nMorphogens, int nMinProducts, int nMaxProducts, int nMaxLinks, int nInputMorphogens, int nTargetMorphogens);
  void removeExcess(int nMinProducts, int nMaxProducts, int nMaxLinks);
  void clear();

  // Text Serialization
  void loadFromString(QString &str);
  QString toString() const;

  friend QTextStream &operator<<(QTextStream &stream, const Model &model);
  friend QTextStream &operator>>(QTextStream &stream, Model &model);

  static double parseDouble(QTextStream &stream);

 private:
  static void distributeProducts(const QSet<int> &fromProds, 
                                 QVector<int> *toProds1, QVector<int> *toProds2);
  static void copyProductsAndLinks(Model *to,
                                const Model *from1, const QVector<int> &products1,
                                const Model *from2, const QVector<int> &products2,
								int numMorphogens);
  static void copyProducts(Model *to,const Model *from, 
                           const QVector<int> &products);
  static void copyLinks(Model *to, 
                        const Model *from1, const QVector<int> &products1,
                        const Model *from2, const QVector<int> &products2,
						int numMorphogens);
  QMap<int, QVector<int> > calcProductRegulators() const;
  QMap<int, QVector<int> > calcRegulatedProducs() const;
  int createNewLabel();


  QVector<ModelProd*> products_;
  QVector<ModelLink*> links_;
};

} // namespace LoboLab
