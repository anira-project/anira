#!/bin/bash
# Format C/C++/ObjC++ files after Write|Edit

FILE=$(jq -r '.tool_input.file_path' 2>/dev/null)
[ -z "$FILE" ] || [ ! -f "$FILE" ] && exit 0

case "$FILE" in
  *.cpp|*.h|*.mm) clang-format -i "$FILE" ;;
esac

exit 0
