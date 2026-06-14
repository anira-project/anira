#!/bin/bash
# Run clang-tidy on C/C++/ObjC++ source files after Write|Edit.
# Only runs on files that have an entry in the desktop build's compile_commands.json —
# files built exclusively by Xcode (iOS) or Gradle/CMake (Android RN) are skipped,
# otherwise clang-tidy falls back to a header-less parse and produces cascading
# false positives from misc-include-cleaner.

FILE=$(jq -r '.tool_input.file_path' 2>/dev/null)
[ -z "$FILE" ] || [ ! -f "$FILE" ] && exit 0

case "$FILE" in
  *.cpp|*.mm)
    CCJSON="build/compile_commands.json"
    [ -f "$CCJSON" ] || exit 0
    if ! jq -e --arg f "$FILE" 'any(.[]; .file == $f)' "$CCJSON" >/dev/null 2>&1; then
      exit 0
    fi
    OUTPUT=$(clang-tidy --warnings-as-errors='*' -p build/ "$FILE" 2>&1)
    if [ $? -ne 0 ]; then
      echo "$OUTPUT" >&2
      exit 2
    fi
    ;;
esac

exit 0
