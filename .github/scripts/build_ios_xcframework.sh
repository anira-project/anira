#!/usr/bin/env bash
# Build anira as a static iOS xcframework (device + simulator arm64 slices) and sign
# it. This is the iOS distribution artifact for CMake *and* Xcode consumers: an
# .xcframework links from both. The backends are linked in as static archives, so
# the slices are self-contained anira + the selected engines.
#
# Usage: build_ios_xcframework.sh <output-dir> [backend cmake flags...]
# Env:
#   ANIRA_CODESIGN_IDENTITY  codesign identity (e.g. "Developer ID Application: …").
#                            Falls back to ad-hoc ("-") when unset, so local/CI runs
#                            without a cert still produce a signed bundle.
set -euo pipefail

OUT_DIR="${1:?usage: build_ios_xcframework.sh <output-dir> [cmake backend flags]}"
shift || true
BACKENDS=("$@")
if [ ${#BACKENDS[@]} -eq 0 ]; then
    BACKENDS=(-DANIRA_WITH_ONNXRUNTIME=ON -DANIRA_WITH_LITERT=ON -DANIRA_WITH_TFLITE=OFF)
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

build_slice() {  # <name> <sysroot> <archs>
    cmake -S "$ROOT" -B "$WORK/$1" -G Ninja \
        -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT="$2" -DCMAKE_OSX_ARCHITECTURES="$3" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF -DANIRA_WITH_LIBTORCH=OFF "${BACKENDS[@]}" >/dev/null
    cmake --build "$WORK/$1" --parallel 4 >/dev/null
}

echo "Building device + simulator slices…"
build_slice device iphoneos arm64
build_slice sim iphonesimulator arm64

mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR/anira.xcframework"
xcodebuild -create-xcframework \
    -library "$WORK/device/libanira.a" -headers "$ROOT/include" \
    -library "$WORK/sim/libanira.a" -headers "$ROOT/include" \
    -output "$OUT_DIR/anira.xcframework"

IDENTITY="${ANIRA_CODESIGN_IDENTITY:--}"  # "-" = ad-hoc
echo "Signing anira.xcframework with identity: ${IDENTITY}"
if [ "$IDENTITY" = "-" ]; then
    # Ad-hoc (local/dev, no cert): no secure timestamp available.
    codesign --sign - --force --timestamp=none "$OUT_DIR/anira.xcframework"
else
    # Real Developer ID: require the secure timestamp — do NOT silently fall back to
    # an un-timestamped signature, which would weaken a published artifact.
    codesign --sign "$IDENTITY" --force --timestamp "$OUT_DIR/anira.xcframework"
fi
codesign --verify --verbose "$OUT_DIR/anira.xcframework"
echo "anira.xcframework ready at $OUT_DIR/anira.xcframework"
