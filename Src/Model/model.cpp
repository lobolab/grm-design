// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "model.h"

#include "modelprod.h"
#include "modellink.h"
#include "Common/mathalgo.h"
#include "Common/log.h"

#include <QSet>

namespace LoboLab {

Model::Model() {}

Model::~Model() {
  clear();
}

// Product-based uniform cross, including exclusive products.
// Child1 is more similar to parent1 and child2 is more similar to parent2.
void Model::cross(const Model *parent1, const Model *parent2,
                  Model *&child1, Model *&child2, int nMorphogens, 
                  int nMinProducts, int nMaxProducts, int nMaxLinks) {
    QSet<int> prodsParent1 = parent1->calcProductLabels().toList().toSet();
    QSet<int> prodsParent2 = parent2->calcProductLabels().toList().toSet();

    // Common products
    QVector<int> commProdsCopied;
    QVector<int> commProdsSwapped;
    QSet<int> commonProds = prodsParent1;
    commonProds.intersect(prodsParent2);
    distributeProducts(commonProds, &commProdsCopied, &commProdsSwapped);

    // Exclusive products 1
    QVector<int> exclusive1to1;
    QVector<int> exclusive1to2;
    distributeProducts(prodsParent1.subtract(commonProds), &exclusive1to1, 
      &exclusive1to2);

    // Exclusive products 2
    QVector<int> exclusive2to1;
    QVector<int> exclusive2to2;
    distributeProducts(prodsParent2.subtract(commonProds), &exclusive2to2, 
      &exclusive2to1);
    
    child1 = new Model();
    child2 = new Model();

    // Copy the components using the selected products
    copyProductsAndLinks(child1, parent1, commProdsCopied + exclusive1to1,
      parent2, commProdsSwapped + exclusive2to1, nMorphogens);
    copyProductsAndLinks(child2, parent2, commProdsCopied + exclusive2to2,
      parent1, commProdsSwapped + exclusive1to2, nMorphogens);

    // Check if child1 is closer to parent2
    if (commProdsCopied.size() + exclusive1to1.size() < 
        commProdsSwapped.size() + exclusive2to1.size()) {
      Model *temp = child2;
      child2 = child1;
      child1 = temp;
    }

    // Remove excess elements
    child1->removeExcess(nMinProducts, nMaxProducts, nMaxLinks);
    child2->removeExcess(nMinProducts, nMaxProducts, nMaxLinks);

#ifdef QT_DEBUG
    // Check that the models are coherent
    QVector<int> labels = child1->calcProductLabels();
    int n = child1->nLinks();
    for (int i = 0; i < n; ++i) {
      ModelLink * link = child1->link(i);
      Q_ASSERT(labels.contains(link->regulatedProdLabel()) &&
        labels.contains(link->regulatedProdLabel()));
    }

    labels = child2->calcProductLabels();
    n = child2->nLinks();
    for (int i = 0; i < n; ++i) {
      ModelLink * link = child2->link(i);
      Q_ASSERT(labels.contains(link->regulatedProdLabel()) &&
        labels.contains(link->regulatedProdLabel()));
    }
#endif
}

void Model::distributeProducts(const QSet<int> &fromProds, QVector<int> *toProds1, 
                               QVector<int> *toProds2) {
  static const int productCrossRate = 50;

  QSetIterator<int> ite(fromProds);
  while (ite.hasNext()) {
    int label = ite.next();
    if (MathAlgo::rand100() < productCrossRate)
      toProds2->append(label);
    else // products copied
      toProds1->append(label);
  }
}

void Model::copyProductsAndLinks(Model *to,
                                 const Model *from1, const QVector<int> &products1,
                                 const Model *from2, const QVector<int> &products2,
								 int nMorphogens) {
  copyProducts(to, from1, products1);
  copyProducts(to, from2, products2);
  copyLinks(to, from1, products1, from2, products2, nMorphogens);
}

void Model::copyProducts(Model *to,
                         const Model *from, const QVector<int> &products) {
  int n = from->nProducts();
  for (int i = 0; i<n; ++i) {
    ModelProd *prod = from->product(i);
    if (products.contains(prod->label()))
      to->products_.append(new ModelProd(*prod));
  }
}

void Model::copyLinks(Model *to, const Model *from1, const QVector<int> &products1,
                      const Model *from2, const QVector<int> &products2, int nMorphogens) {
  // regulators list are possible regulators for substitution
  QVector<int> regulators1 = products1;
  qSort(regulators1);
  int i = 0;
  while (i < nMorphogens && i < regulators1.size()) {
    if (regulators1[i] < nMorphogens)
      regulators1.takeAt(i);
    else
      ++i;
  }

  QVector<int> regulators2 = products2;
  qSort(regulators2);
  i = 0;
  while (i < nMorphogens && i < regulators2.size()){
    if (regulators2[i] < nMorphogens)
      regulators2.takeAt(i);
    else
      ++i;
  }

  int n = from1->nLinks();
  for (int i = 0; i < n; ++i) {
    ModelLink *link = from1->link(i);
    int regulated = link->regulatedProdLabel();

    if (products1.contains(regulated)) {
      int regulator = link->regulatorProdLabel();
      if (products1.contains(regulator) || // Use the original regulator
          products2.contains(regulator)) {
        to->links_.append(new ModelLink(*link));
      } else if (!regulators2.isEmpty()) { // Substitute the regulator
        int newRegulator = regulators2[MathAlgo::randInt(regulators2.size())];
        if (!from1->findLink(newRegulator, regulated)) {
          ModelLink *newLink = new ModelLink(*link);
          newLink->setRegulator(newRegulator);
          to->links_.append(newLink);
        }
      }
    }
  }
  
  n = from2->nLinks();
  for (int i = 0; i < n; ++i) {
    ModelLink *link = from2->link(i);
    int regulated = link->regulatedProdLabel();

    if (products2.contains(regulated)) {
      int regulator = link->regulatorProdLabel();
      if (products2.contains(regulator) || // Use the original regulator
          products1.contains(regulator)) {
        to->links_.append(new ModelLink(*link));
      } else if (!regulators1.isEmpty()) { // Substitute the regulator
        int newRegulator = regulators1[MathAlgo::randInt(regulators1.size())];
        if (!from2->findLink(newRegulator, regulated)) {
          ModelLink *newLink = new ModelLink(*link);
          newLink->setRegulator(newRegulator);
          to->links_.append(newLink);
        }
      }
    }
  }
}

Model::Model(const Model &source) {
  *this = source;
}

Model &Model::operator=(const Model &source) {
  clear();

  int n = source.nProducts();
  for (int i = 0; i<n; ++i)
    products_.append(new ModelProd(*source.product(i)));

  n = source.nLinks();
  for (int i = 0; i<n; ++i)
    links_.append(new ModelLink(*source.link(i)));

  return *this;
}

void Model::clear() {
  int n = products_.size();
  for (int i=0; i<n; ++i)
    delete products_.at(i);

  products_.clear();

  n = links_.size();
  for (int i=0; i<n; ++i)
    delete links_.at(i);

  links_.clear();
}

QVector<int> Model::calcProductLabels() const {
  QVector<int> labels;
  int n = products_.size();
  for (int i = 0; i < n; ++i)
    labels += products_.at(i)->label();

  return labels;
}

ModelProd *Model::prodWithLabel(int label, int *ind) const {
  int nProducts = products_.size();
  int i = 0;
  while (i < nProducts && products_.at(i)->label() != label)
    ++i;

  if (i < nProducts) {
    if (ind)
      *ind = i;
    return products_.at(i);
  } else {
    return NULL;
  }
}

// The products in use are those with a path starting and ending in a structural product.
QVector<int> Model::calcProductLabelsInUse(int nMorphogens) const {
  const int nStrucLabels = nMorphogens;

  // Pre-calculate the links in the model
  QMap<int, QVector<int> > productRegulators = calcProductRegulators();
  QMap<int, QVector<int> > regulatedProducts = calcRegulatedProducs();

  // All struct products are in use
  QVector<int> productsInUse;
  for (int i = 0; i < nStrucLabels; ++i)
    productsInUse.append(i);

  // Products regulating a product in use are in use (Backward from output)
  QVector<int> productsToVisit = productsInUse;
  QVector<int> regulatorsInUse;
  while (!productsToVisit.isEmpty()) {
    const QVector<int> &regulators = productRegulators[productsToVisit.takeFirst()];
    int nRegulators = regulators.size();
    for (int j = 0; j < nRegulators; ++j) {
      int r = regulators[j];
      if (!productsInUse.contains(r) && !regulatorsInUse.contains(r)) {
        regulatorsInUse.append(r);
        productsToVisit.append(r);
      }
    }
  }

  // Products regulated by a product in use are in use (Forward from input)
  productsToVisit = productsInUse;
  while (!productsToVisit.isEmpty()) {
    const QVector<int> &regulatedProdLabels = regulatedProducts[productsToVisit.takeFirst()];
    int nRegulatedProds = regulatedProdLabels.size();
    for (int j = 0; j < nRegulatedProds; ++j) {
      int r = regulatedProdLabels[j];
      if (!productsInUse.contains(r) && regulatorsInUse.contains(r)) {
        productsInUse.append(r);
        productsToVisit.append(r);
      }
    }
  }
  return productsInUse;
}

QMap<int, QVector<int> > Model::calcProductRegulators() const {
  QMap<int, QVector<int> > productRegulators;

  int n = links_.size();
  for (int i = 0; i < n; ++i) {
    ModelLink *link = links_.at(i);
    productRegulators[link->regulatedProdLabel()].append(
      link->regulatorProdLabel());
  }

  return productRegulators;
}

QMap<int, QVector<int> > Model::calcRegulatedProducs() const {
  QMap<int, QVector<int> > regulatedProducts;

  int n = links_.size();
  for (int i = 0; i < n; ++i) {
    ModelLink *link = links_.at(i);
    regulatedProducts[link->regulatorProdLabel()].append(
      link->regulatedProdLabel());
  }

  return regulatedProducts;

}

Model *Model::createRandom(int nProducts, int nMorphogens, int nInputMorphogens, int nTargetMorphogens) {
  Model *model = new Model();
  QVector<int> prodLabels;

  for (int i = 0; i < nProducts; ++i) {
    model->addRandomProduct(i, nMorphogens, nInputMorphogens);
    prodLabels.append(i);
  }

  // 0-nProducts regulatory connections for morphogens and the rest
  for (int i = 0; i < nProducts; ++i) {
    if (i < nInputMorphogens || i >= nInputMorphogens + nTargetMorphogens){ // Avoiding output genes to regulate other genes.
      int nReg = MathAlgo::randInt(1 + nProducts);
      for (int j = 0; j < nReg; ++j)
        model->addOrReplaceRandomLink(prodLabels[i],
          nInputMorphogens + prodLabels[MathAlgo::randInt(nProducts - nInputMorphogens)]);
    }
  }


#ifdef QT_DEBUG
  // Check that the model is coherent
  QVector<int> labels = model->calcProductLabels();
  int n = model->nLinks();
  for (int i = 0; i < n; ++i) {
    ModelLink * link = model->link(i);
    Q_ASSERT(labels.contains(link->regulatedProdLabel()) &&
             labels.contains(link->regulatedProdLabel()));
  }
#endif

  return model;
}

void Model::duplicateProduct(int i, int nMorphogens, int nInputMorphogens, int nTargetMorphogens) {

  int newLabel = createNewLabel();
  ModelProd *newProd = new ModelProd(nMorphogens, newLabel, nInputMorphogens);
  products_.append(newProd);
  
  // Create two random links
  QVector<int> labels = calcProductLabels();
  qSort(labels);

  int regulatorLabel = labels[MathAlgo::randIntSkip(nProducts(), nInputMorphogens, nTargetMorphogens)];
  int regulatedLabel = labels[nInputMorphogens + MathAlgo::randInt(nProducts() - nInputMorphogens)];

  addOrReplaceRandomLink(newLabel, regulatedLabel);

  addOrReplaceRandomLink(regulatorLabel, newLabel);
}


void Model::removeProduct(int i) {
  ModelProd *prod = products_.takeAt(i);
  int label = prod->label();
  delete prod;

  int n = links_.size();
  for (int i = n-1; i >= 0; --i) {
    ModelLink *link = links_.at(i);
    if (link->regulatorProdLabel() == label ||
        link->regulatedProdLabel() == label) {
      links_.removeAt(i);
      delete link;
    }
  }
}

void Model::removeProductWithLabel(int label) {
  int i;
  ModelProd *prod = prodWithLabel(label, &i);

  if (prod) {
    products_.takeAt(i);
    int label = prod->label();
    delete prod;

    int n = links_.size();
    for (int i = n-1; i >= 0; --i) {
      ModelLink *link = links_.at(i);
      if (link->regulatorProdLabel() == label ||
          link->regulatedProdLabel() == label) {
        links_.removeAt(i);
        delete link;
      }
    }
  }
}

int Model::createNewLabel() {
    // Select always a new label
  static int newLabel = 10;
  newLabel++;

  return newLabel;
}

void Model::addRandomProduct(int label, int nMorphogens, int nInputMorphogens) {
  if (ModelProd *oldProd = prodWithLabel(label)) {
    products_.removeOne(oldProd);
    delete oldProd;
  }

  ModelProd *newProd = new ModelProd(nMorphogens, label, nInputMorphogens);
  products_.append(newProd);
}

//void Model::addRandomProductWithLinks() {
//  int newLabel = addRandomProduct();
//  int nProducts = products_.size();
//
//  // Create regulators for the product
//  int n = MathAlgo::randInt(1, 2);
//  for (int i = 0; i < n; ++i)
//    addOrReplaceRandomLink(products_[MathAlgo::randInt(nProducts)]->label(),
//                           newLabel);
//
//  // Create regulations by the product
//  n = MathAlgo::randInt(1, 2);
//  for (int i = 0; i < n; ++i)
//    addOrReplaceRandomLink(newLabel,
//                           products_[MathAlgo::randInt(nProducts)]->label());
//}

//void Model::addRandomProductInLink(int iLink) {
//  int newProductLabel = addRandomProduct();
//  ModelLink *link = links_.at(iLink);
//  int regulatorProdLabel = link->regulatorProdLabel();
//  link->setRegulator(newProductLabel);
//
//  // Create an activation regulation for the product
//  ModelLink *newLink = new ModelLink(regulatorProdLabel, newProductLabel, true);
//  links_.insert(MathAlgo::randInt(links_.size() + 1), newLink);
//}

QVector<ModelLink*> Model::linksToLabel(int label) const {
  QVector<ModelLink*> linksToLabel;
  int nLinks = links_.size();
  for (int i = 0; i < nLinks; ++i)
    if (links_.at(i)->regulatedProdLabel() == label)
      linksToLabel.append(links_.at(i));

  return linksToLabel;
}

QVector<ModelLink*> Model::calcLinksInUse(int nStrucLabels) const {
  QVector<ModelLink*> linksInUse;
  QVector<int> labelsInUse = calcProductLabelsInUse(nStrucLabels);
  int nLinks = links_.size();
  for (int i = 0; i < nLinks; ++i) {
    ModelLink *link = links_.at(i);
    if (labelsInUse.contains(link->regulatorProdLabel()) &&
        labelsInUse.contains(link->regulatedProdLabel()))
      linksInUse.append(link);
  }

  return linksInUse;
}

int Model::calcNLinksFromProd(int prodLabel) const {
  int n = 0;
  int nLinks = links_.size();
  for (int i = 0; i < nLinks; ++i)
    if (prodLabel == links_[i]->regulatorProdLabel())
      ++n;

  return n;
}

ModelLink *Model::findLink(int regulator, int regulated) const {
  ModelLink *foundLink = NULL;
  int nLinks = links_.size();
  int i = 0;
  while (foundLink == NULL && i < nLinks) {
    ModelLink *link = links_[i];
    if (link->regulatorProdLabel() == regulator && 
        link->regulatedProdLabel() == regulated)
      foundLink = link;
    else
      ++i;
  }

  return foundLink;
}


void Model::addOrReplaceRandomLink() {
  int nProducts = products_.size();
  int regulatorLabel = products_.at(MathAlgo::randInt(nProducts))->label();
  int regulatedLabel = products_.at(MathAlgo::randInt(nProducts))->label();

  addOrReplaceRandomLink(regulatorLabel, regulatedLabel);
}

void Model::addOrReplaceRandomLink(int regulatorLabel, int regulatedLabel) {
    removeLink(regulatorLabel, regulatedLabel);
    ModelLink *newLink = new ModelLink(regulatorLabel, regulatedLabel);
    links_.append(newLink);
}

//If the link exists, delete it
void Model::removeLink(int regulatorLabel, int regulatedLabel) {
  int nLinks = links_.size();
  bool found = false;
  int i = 0;
  while (!found && i < nLinks) {
    ModelLink *link = links_.at(i);
    if (regulatorLabel == link->regulatorProdLabel() &&
        regulatedLabel == link->regulatedProdLabel()) {
      found = true;
      links_.removeAt(i);
      delete link;
    } else
      ++i;
  }
}

void Model::duplicateLink(int i, int nMorphogens, int nInputMorphogens, int nTargetMorphogens) {
  ModelLink *newLink = new ModelLink(*links_.at(i));
  QVector<int> labels = calcProductLabels();
  qSort(labels);
  // Autoregulating structural morphogens
  int regulatorLabel = labels.at(MathAlgo::randIntSkip(nProducts(), nInputMorphogens, nTargetMorphogens));
  int regulatedLabel = labels.at(MathAlgo::randInt(nInputMorphogens, nProducts()-1));
  
  //if (regulatorLabel < nMorphogens)
  //  regulatedLabel = regulatorLabel;
  //else
  //  regulatedLabel = products_.at(MathAlgo::randInt(nProducts()))->label();

  //// No autoregulating structural morphogens
  //QVector<int> labels = calcProductLabels();
  //qSort(labels);
  //int regulatorLabel = labels.at(3+MathAlgo::randInt(nProducts()-3));
  //int regulatedLabel = products_.at(MathAlgo::randInt(nProducts()))->label();
  
  removeLink(regulatorLabel, regulatedLabel);
  newLink->setRegulator(regulatorLabel);
  newLink->setRegulated(regulatedLabel);
  links_.append(newLink);

}

void Model::removeLink(int i) {
  delete links_.takeAt(i);
}

void Model::removeLink(ModelLink *modelLink) {
  int i = 0;
  bool found = false;
  while (i < nLinks() && !(found = modelLink == link(i)))
    ++i;

  if (found)
    removeLink(i);
}

void Model::mutate(int nMorphogens, int nMinProducts, int nMaxProducts, int nMaxLinks, int nInputMorphogens, int nTargetMorphogens) {

  // Copy based mutations

  // Product mutations
  for (int i = nProducts()-1; i >= 0;  --i) {
    ModelProd *prod = products_[i];

    if (MathAlgo::rand100() < 1) // Copy product
      duplicateProduct(i, nMorphogens, nInputMorphogens, nTargetMorphogens);

    if (prod->label() >= nMinProducts && MathAlgo::rand1000() < 50) // Remove product
      removeProduct(i);
  }
  
  // Link mutations
  for (int i = nLinks()-1; i >= 0;  --i) {

    if (MathAlgo::rand100() < 1) // Copy link
      duplicateLink(i, nMorphogens, nInputMorphogens, nTargetMorphogens);
      
    if (MathAlgo::rand1000() < 50) // Remove link
      removeLink(i);
  }

  // Remove excess elements
  removeExcess(nMinProducts, nMaxProducts, nMaxLinks);

  //// Independent topological mutations

  //// Remove a product
  //if (MathAlgo::rand100() < 5 && nProducts() > 4) { 
  //  QVector<int> labels = calcProductLabels();
  //  for (int i=0; i < 4; ++i)
  //    labels.remove(i);

  //  int pLabel = labels.at(MathAlgo::randInt(labels.size()));

  //  if (MathAlgo::randBool())
  //    removeProductWithLabel(pLabel);
  //  else
  //    replaceProductWithLabelByRandomLinks(pLabel);
  //}  
  //
  //// Add a product
  //if (MathAlgo::rand100() < 4)
  //    addRandomProductWithLinks();

  //// Duplicate product
  //if (MathAlgo::rand100() < 1)
  //  duplicateProduct(MathAlgo::randInt(nProducts()));

  //// Move a link
  //if (MathAlgo::rand100() < 1 && nLinks() > 0) {
  //  ModelLink *link = links_[MathAlgo::randInt(nLinks())];
  //  if (MathAlgo::randBool())
  //    link->setRegulated(products_[MathAlgo::randInt(nProducts())]->label());
  //  else
  //    link->setRegulator(products_[MathAlgo::randInt(nProducts())]->label());
  //}

  //// Remove a link
  //if (MathAlgo::rand100() < 5 && nLinks() > 0)
  //  removeLink(MathAlgo::randInt(nLinks()));

  //// Add a link
  //if (MathAlgo::rand100() < 4)
  //  addOrReplaceRandomLink();
  
  //// Swap links (changes the boolean function)
  //if (MathAlgo::rand100() < 1 && nLinks() > 2)
  //  links_.swap(MathAlgo::randInt(nLinks()), MathAlgo::randInt(nLinks()));


  // Individual param mutations
  static const int paramMutationProb = 1;

  int n = nProducts();
  for (int i = 0; i < n; ++i)
    products_.at(i)->mutateParams(paramMutationProb, nMorphogens, nInputMorphogens);

  n = nLinks();
  for (int i = 0; i < n; ++i)
    links_.at(i)->mutateParams(paramMutationProb);


  //// A single param mutation per product and link
  //static const int paramMutationProb = 5;

  //int n = nProducts();
  //for (int i = 0; i < n; ++i)
  //  if (MathAlgo::rand100() < paramMutationProb)
  //    products_.at(i)->mutateParam();

  //n = nLinks();
  //for (int i = 0; i < n; ++i)
  //  if (MathAlgo::rand100() < paramMutationProb)
  //    links_.at(i)->mutateParam();





  //// A Single mutation

  //int r = MathAlgo::rand100();

  //if (r < 90) { // Mutate 1 param
  //  int i = MathAlgo::randInt(nProducts() + nLinks());
  //  if (i < nProducts())
  //    products_[i]->mutateParam();
  //  else
  //    links_[i-nProducts()]->mutateParam();
  //} 
  //
  //else if (r < 94 && nProducts() > 4) { // Remove a product
  //  QVector<int> labels = calcProductLabels();
  //  for (int i=0; i < 4; ++i)
  //    labels.remove(i);
  //  removeProductWithLabel(labels.at(MathAlgo::randInt(labels.size())));
  //}  
  //
  //else if (r < 97) { // Add a product
  // addRandomProductWithLinks();
  //} 
  //
  //else if (r < 99 && nLinks() > 0) { // Remove a link
  //  removeLink(MathAlgo::randInt(nLinks()));
  //} 
  //
  //else { // Add a link
  //  addOrReplaceRandomLink();
  //} 





  //// Independent mutations

  //// Remove a product
  //QVector<int> labels = calcProductLabels();
  //for (int i = labels.size() - 1; i >= 0; --i)
  //  if (MathAlgo::rand100() < 1 && labels.at(i) > 3) {
  //    if (MathAlgo::randBool())
  //      removeProductWithLabel(labels.at(i));
  //    else
  //      replaceProductWithLabelByRandomLinks(labels.at(i));
  //}

  //// Add regulating/regulated product
  //for (int i = nProducts() - 1; i >= 0; --i)
  //  if (MathAlgo::rand100() < 1)
  //    addRandomProductWithLinks();
  //  
  //// Remove link
  //for (int i = nLinks() - 1; i >= 0; --i)
  //  if (MathAlgo::rand100() < 1)
  //    removeLink(i);
  //
  //// Add link
  //for (int i = nProducts() - 1; i >= 0; --i)
  //  if (MathAlgo::rand100() < 1)
  //    addOrReplaceRandomLink();

  //// Swap links (changes the boolean function)
  //if (nLinks() > 2)
  //  for (int i = nLinks() / 2; i >= 0; --i)
  //    if (MathAlgo::rand100() < 1)
  //      links_.swap(MathAlgo::randInt(nLinks()), MathAlgo::randInt(nLinks()));
  //
  //// Param mutations
  //static const int paramMutationProb = 1;

  //int n = nProducts();
  //for(int i = 0; i < n; ++i)
  //  products_.at(i)->mutateParams(paramMutationProb);

  //n = nLinks();
  //for(int i = 0; i < n; ++i)
  //  links_.at(i)->mutateParams(paramMutationProb);


#ifdef QT_DEBUG
  {
    // Check that the model is coherent
    QVector<int> labels = calcProductLabels();
    for (int i = 0; i < nLinks(); ++i) {
      ModelLink * link = links_.at(i);
      Q_ASSERT(labels.contains(link->regulatedProdLabel()) &&
               labels.contains(link->regulatedProdLabel()));
    }
  }
#endif
}

//// Removing excess of ALL products and links
//void Model::removeExcess(int nMinProducts, int nMaxProducts, int nMaxLinks) {
//  if (nMaxProducts > -1)
//    while (nProducts() > nMaxProducts)
//      removeProduct(MathAlgo::randInt(nMinProducts, nMaxProducts-1));
//
//  if (nMaxLinks > -1)
//    while (nLinks() > nMaxLinks)
//      removeLink(MathAlgo::randInt(nLinks()));
//}

// Removing excess of USED products and links
void Model::removeExcess(int nMinProducts, int nMaxProducts, int nMaxLinks) {
  if (nMaxProducts > -1) {
    QVector<int> productLabelsInUse = calcProductLabelsInUse(nMinProducts);
    int nProductsToRemove = productLabelsInUse.size() - nMaxProducts;
    if(nProductsToRemove > 0) {
      // discard non-removable products
      productLabelsInUse.remove(0, nMinProducts);

      for(int i = 0; i < nProductsToRemove; ++i)
        removeProductWithLabel(productLabelsInUse.takeAt(MathAlgo::randInt(productLabelsInUse.size())));
    }
  }

  if (nMaxLinks > -1) {
    QVector<ModelLink*> linksInUse = calcLinksInUse(nMinProducts);
    int nLinksToRemove = linksInUse.size() - nMaxLinks;

    for (int i = 0; i < nLinksToRemove; ++i)
      removeLink(linksInUse.takeAt(MathAlgo::randInt(linksInUse.size())));
  }
}

int Model::calcComplexity() const {
  int complexity = 0;

  int n = products_.size();
  for (int i=0; i < n; ++i)
    complexity += products_.at(i)->complexity();

  n = links_.size();
  for (int i=0; i < n; ++i)
    complexity += links_.at(i)->complexity();

  return complexity;
}

int Model::calcComplexityInUse(int nStrucProd) const {
  int complexity = 0;

  const QVector<int> labelsInUse = calcProductLabelsInUse(nStrucProd);

  int n = products_.size();
  for (int i=0; i < n; ++i) {
    ModelProd *prod = products_.at(i);
    if (labelsInUse.contains(prod->label()))
      complexity += prod->complexity();
  }

  n = links_.size();
  for (int i=0; i < n; ++i) {
    ModelLink *link = links_.at(i);
    if (labelsInUse.contains(link->regulatorProdLabel()) &&
        labelsInUse.contains(link->regulatedProdLabel()))
      complexity += link->complexity();
  }

  return complexity;
}

//bool Model::hasSameTopology(const Model &other) const
//{
//	bool equal = true;
//
//	const QVector<int> products = calcProductsInUse(4);
//	const QVector<int> productsOther = other.calcProductsInUse(4);
//
//	if(products != productsOther)
//		equal = false;
//	else
//	{
//		const QMap<int, QVector<int> > links = calcLinks();
//		const QMap<int, QVector<int> > otherLinks = other.calcLinks();
//
//		QVector<int>::const_iterator i = products.constBegin();
//		while(equal && i != products.constEnd())
//		{
//			int p = *i;
//			const QVector<int> connex = links.value(p);
//			const QVector<int> connexOther = otherLinks.value(p);
//			if(connex.toSet().intersect(products) ==
//			   connexOther.toSet().intersect(products))
//				++i;
//			else
//				equal = false;
//		}
//	}
//
//	return equal;
//}

// Text Serialization
void Model::loadFromString(QString &str) {
  QTextStream stream(&str, QIODevice::ReadOnly);
  stream >> *this;
}

QString Model::toString() const {
  QString str;
  QTextStream stream(&str, QIODevice::WriteOnly);
  stream << *this;

  return str;
}

QTextStream &operator<<(QTextStream &stream, const Model &model) {
  stream << '(';

  int n = model.products_.size();
  if (n > 0) {
    for (int i=0; i<n-1; ++i) {
      stream << *model.products_.at(i);
      stream << '|';
    }

    stream << *model.products_.last();
  }

  stream << '*';

  n = model.links_.size();
  if (n > 0) {
    for (int i=0; i<n-1; ++i) {
      stream << *model.links_.at(i);
      stream << '|';
    }

    stream << *model.links_.last();
  }

  stream << ')';

  Q_ASSERT(model.nProducts() > 0);

  return stream;
}

QTextStream &operator>>(QTextStream &stream, Model &model) {
  model.clear();

  char c;
  stream >> c;
  Q_ASSERT(c == '(');

  while (c != '*') {
    ModelProd *modelProd = new ModelProd();
    stream >> *modelProd;
    model.products_.append(modelProd);
    stream >> c;
  }

  stream >> c;
  if (c != ')') {
    stream.seek(stream.pos()-1);

    while (c != ')') {
      ModelLink *modelLink = new ModelLink();
      stream >> *modelLink;
      model.links_.append(modelLink);
      stream >> c;
    }
  }

  Q_ASSERT(model.nProducts() > 0);

  return stream;
}

double Model::parseDouble(QTextStream &stream) {
  QChar c;
  QString str;
  bool end = false;
  while (!end) {
    stream >> c;
    if (c == ' ' || c == '|' || c == '*' || c == ')') {
      end = true;
      stream.seek(stream.pos() - 1);
    } else if (c == 0) // end of stream
      end = true;
    else
      str.append(c);
  }

  return str.toDouble();
}

}