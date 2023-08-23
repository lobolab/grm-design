// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#pragma once

#include <QString>

namespace LoboLab {

class DB;

class DBExp {
 public:

  static bool buildDB(DB *db);
  static int getDBVersion(DB *db);
  static bool upgradeDB(DB *db, int ver);

  //void deleteAll();
  //void deleteAllExperiments();
  //void deleteAllExperimentsAndManipulations();

  //int importDB(const QString &fileName, int &nExperiments,
  //	int &nManipulations, int &nMorphologies);



 private:
  DBExp();
  virtual ~DBExp();

};

} // namespace LoboLab
