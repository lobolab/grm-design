// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "dbexp.h"
#include "db.h"
////#include "log.h"

#include <QSqlQuery>
#include <QSqlError>
#include <QStringList>
#include <QDebug>

namespace LoboLab {

DBExp::DBExp() {
}

DBExp::~DBExp() {
}
//
//void DBExp::deleteAll()
//{
//	deleteAllExperimentsAndManipulations();
//	deleteAllMorphologies();
//}
//
//void DBExp::deleteAllExperiments()
//{
//	QList<QString> tables;
//	tables.append("ExperimentDrug");
//	tables.append("ExperimentRNAi");
//	tables.append("ResultantMorphology");
//	tables.append("ResultSet");
//	tables.append("Experiment");
//	emptyTables(tables);
//}
//
//void DBExp::deleteAllExperimentsAndManipulations()
//{
//	deleteAllExperiments();
//
//	QList<QString> tables;
//	tables.append("Manipulation");
//	tables.append("RemoveActionAreaPoint");
//	tables.append("CropActionAreaPoint");
//	emptyTables(tables);
//
//	tables.clear();
//	// the rest is not foreign key compliant
//	tables.append("ManipulationAction");
//	tables.append("MorphologyAction");
//	tables.append("RemoveAction");
//	tables.append("CropAction");
//	tables.append("JoinAction");
//	emptyTablesNoKeys(tables);
//}
//
//void DBExp::deleteAllMorphologies()
//{
//	QList<QString> tables;
//	tables.append("Organ");
//	tables.append("LineOrgan");
//	tables.append("SpotOrgan");
//	tables.append("RegionsLink");
//	tables.append("RegionParam");
//	tables.append("RegionParam");
//	tables.append("Region");
//	tables.append("Morphology");
//
//	emptyTables(tables);
//}

bool DBExp::buildDB(DB *db) {
  const char *dump = "";
  QStringList sqlSts = QString(dump).split(";", QString::SkipEmptyParts);

  db->beginTransaction();
  QSqlQuery *query = db->newQuery();
  int nSts = sqlSts.size();
  bool ok = true;
  for (int i=0; i<nSts; ++i) {
    ok &= query->exec(sqlSts.at(i));
    Q_ASSERT_X(ok, QString("DBExp::buildDB: %1")
               .arg(sqlSts.at(i)).toLatin1(),
               query->lastError().text().toLatin1());
  }
  db->endTransaction();

  delete query;

  return ok;
}

int DBExp::getDBVersion(DB *db) {
  int ver;

  if (!db->exist("Meta")) // Version 1
    ver = 1;
  else {
    QSqlQuery *query = db->newQuery("SELECT DBStructVer FROM Meta");
    if (query->next())
      ver = query->value(0).toInt();
    else
      ver = -1;

    delete query;
  }

  return ver;
}

bool DBExp::upgradeDB(DB *db, int ver) {
  db->beginTransaction();
  QSqlQuery *query = db->newQuery();
  bool ok = true;
  switch (ver) {
    // No breaks make the update sequential
    case (1): // 1 -> 2
      ok &= query->exec("CREATE TABLE [Meta] ([DBStructVer] INTEGER);");
      ok &= query->exec("INSERT INTO 'Meta' VALUES(2);");
      ok &= query->exec("CREATE TABLE [MorphologyImage] ([Id] INTEGER NOT NULL PRIMARY KEY, [Morphology] INTEGER NOT NULL REFERENCES [Morphology]([Id]), [Data] BLOB NOT NULL);");

      //case(2): // 2 -> 3
  }

  Q_ASSERT_X(ok, QString("DBExp::upgradeDB: ver=%1")
             .arg(ver).toLatin1(), query->lastError().text().toLatin1());

  db->endTransaction();
  delete query;

  return ok;
}


}
