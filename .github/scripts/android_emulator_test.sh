#!/usr/bin/env bash
# Stage the cross-built test binary, its shared-library dependencies and the model
# tree onto a booted Android emulator, then run the whole bundled gtest suite in a
# single launch and assert a zero exit. Invoked from build_test_mobile.yml inside
# reactivecircus/android-emulator-runner, which executes each line of an inline
# `script:` in its own shell — so the orchestration lives here as one process where
# the variables persist. Expects: build-x86_64/ (configured with
# ANIRA_EXTRAS_MODELS_DIR=$DEV/models), extras/models/, ANDROID_NDK_LATEST_HOME.
set -euo pipefail

DEV=/data/local/tmp/anira
NDK_SYSROOT_LIB="$ANDROID_NDK_LATEST_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/x86_64-linux-android"
ONNX_SO=$(find modules -path '*x86_64/libonnxruntime.so' | head -1)
LITERT_SO=$(find modules -path '*x86_64/libLiteRt.so' | head -1)

adb shell "rm -rf $DEV && mkdir -p $DEV"

# binary + every shared dependency next to it, reached via LD_LIBRARY_PATH
adb push build-x86_64/test/tests "$DEV/tests"
adb push build-x86_64/libanira.so "$DEV/"
adb push build-x86_64/lib/libgtest.so build-x86_64/lib/libgtest_main.so "$DEV/"
adb push "$ONNX_SO" "$DEV/"
adb push "$LITERT_SO" "$DEV/"
adb push "$NDK_SYSROOT_LIB/libc++_shared.so" "$DEV/"

# model tree staged to the path baked into the build (ANIRA_EXTRAS_MODELS_DIR)
adb push extras/models "$DEV/models"
adb shell "chmod 755 $DEV/tests"

# one launch runs the whole bundled suite; assert the device-side exit code is 0
adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV ./tests --gtest_brief=1; echo ANIRA_EXIT=\$?" | tee /tmp/out.txt
grep -q "ANIRA_EXIT=0" /tmp/out.txt
