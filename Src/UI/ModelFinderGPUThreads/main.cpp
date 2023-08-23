// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#include "maincmd.h"

#include <QCoreApplication>
#include <iostream>

#ifdef QT_DEBUG
#ifndef Q_WS_X11
  #include <vld.h>
#endif
#endif

void myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
  QByteArray localMsg = msg.toLocal8Bit();
  switch (type) {
  case QtDebugMsg:
    fprintf(stderr, "Debug: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
    break;
  case QtInfoMsg:
    fprintf(stderr, "Info: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
    break;
  case QtWarningMsg:
    fprintf(stderr, "Warning: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
    break;
  case QtCriticalMsg:
    fprintf(stderr, "Critical: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
    break;
  case QtFatalMsg:
    fprintf(stderr, "Fatal: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
    abort();
  }
}

int main(int argc, char *argv[]) {
  int ret;

  {
    // QCoreApplication doesn't work with morphologytranslator
    QCoreApplication app(argc, argv);


#ifdef QT_DEBUG
#ifndef Q_WS_X11
    VLD_UINT options = VLDGetOptions();
    QString filename = QString("memory_leaks.txt");
    VLDSetReportOptions(VLD_OPT_REPORT_TO_FILE, (const wchar_t *)filename.utf16());
#endif
#endif

    // Q_ASSERT handler
    qInstallMessageHandler(myMessageOutput);

    LoboLab::MainCmd mc;
    ret = app.exec();
  }

#ifdef QT_DEBUG
  std::cout << std::endl;
  std::cout << "**** main finished (ret=" << ret << ")..." << std::endl;
#endif

  return ret;
}
