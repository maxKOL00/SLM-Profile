// created by Mark O. Brown
#pragma once
// This header file includes all of the extern declarations of all of my external (global) variables.
#include "afxWin.h"
#include <string>
#include <vector>
#include <unordered_map>





extern CWnd* eMainWindowHwnd;

extern std::unordered_map<std::string, COLORREF> _myRGBs;
extern std::unordered_map<std::string, CBrush*> _myBrushes;
void initMyColors();
/// some globals for niawg stuff, only for niawg stuff so I keep it here...?
extern const std::array<int, 2> AXES;
// the following is used to receive the index of whatever axis is not your current axis.
extern const std::array<int, 2> ALT_AXES;
extern const std::array<std::string, 2> AXES_NAMES;

extern bool eWaitError;

// stuff for syntax coloring

/// Global Options Variables

extern bool eAbortNiawgFlag;

/// Other Global Handles
extern HANDLE eWaitingForNIAWGEvent;
extern HANDLE eNIAWGWaitThreadHandle;

