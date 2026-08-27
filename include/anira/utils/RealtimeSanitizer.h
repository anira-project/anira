#ifndef ANIRA_REALTIMESANITIZER_H
#define ANIRA_REALTIMESANITIZER_H

#ifdef ANIRA_WITH_RTSAN
#define ANIRA_REALTIME [[clang::nonblocking]]
// Marks a function that intentionally blocks (sleeps, waits). RealtimeSanitizer
// reports the moment such a function is CALLED from a [[clang::nonblocking]]
// context — deterministically, even on runs where the blocking branch would not
// have executed — instead of only when a blocking syscall actually fires.
#define ANIRA_BLOCKING [[clang::blocking]]
#else
#define ANIRA_REALTIME
#define ANIRA_BLOCKING
#endif

#endif  // ANIRA_REALTIMESANITIZER_H