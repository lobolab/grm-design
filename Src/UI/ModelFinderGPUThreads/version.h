// Copyright (c) Lobo Lab (lobolab.umbc.edu)
// All rights reserved.

#pragma once

#define V_MAYORVERSION		0
#define V_MINORVERSION		1
#define V_BUGFIXVERSION		0

#define V_COMPANYNAME		"LoboLab\0"

#define V_FILEDESCRIPTION	"Model Finder\0"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define V_PRODUCTVERSION	"v" TOSTRING(V_MAYORVERSION) "." TOSTRING(V_MINORVERSION) "." TOSTRING(V_BUGFIXVERSION)
#define V_LEGALCOPYRIGHT	"Lobo Lab (lobolab.umbc.edu). All rights reserved.\0"
#define V_ORIGINALFILENAME	"ModelFinder.exe\0"
#define V_PRODUCTNAME		"Model Finder\0"
