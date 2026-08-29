# SPDX-License-Identifier: Apache-2.0
# bin2cpp.cmake
# Variables: SRC, DST, HEADER, NAMESPACE, VAR

# Strip possible quotes from paths
string(REPLACE "\"" "" SRC "${SRC}")
string(REPLACE "\"" "" DST "${DST}")

file(READ "${SRC}" data HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " data "${data}")
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f], ){16})" "\\1\n    " data "${data}")

file(WRITE "${DST}" "// Auto-generated binary data file\n")
file(APPEND "${DST}" "#include \"${HEADER}\"\n\n")
file(APPEND "${DST}" "namespace ${NAMESPACE}\n{\n")
file(APPEND "${DST}" "static const unsigned char ${VAR}_data[] = {\n    ${data}};\n\n")
file(APPEND "${DST}" "const char* ${VAR} = reinterpret_cast<const char*>(${VAR}_data);\n\n")
file(APPEND "${DST}" "}\n")
