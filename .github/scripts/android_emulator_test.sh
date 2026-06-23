#!/usr/bin/env bash
# Stage the cross-built test binary, its shared-library dependencies and the model
# tree onto a booted Android emulator, then run the whole bundled gtest suite in a
# single launch and assert a zero exit. Invoked from build_test_mobile.yml inside
# reactivecircus/android-emulator-runner, which executes each line of an inline
# `script:` in its own shell — so the orchestration lives here as one process where
# the variables persist.
#
# Backend/linkage-agnostic: it pushes whatever .so the build actually produced (a
# shared build yields libanira.so + libgtest*.so + the enabled backend .so; a static
# build yields none, the binary is self-contained bar the C++ runtime). Expects
# build-x86_64/ (configured with ANIRA_EXTRAS_MODELS_DIR=$DEV/models), extras/models/,
# and ANDROID_NDK_LATEST_HOME.
set -euo pipefail

DEV=/data/local/tmp/anira
NDK_SYSROOT_LIB="$ANDROID_NDK_LATEST_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/x86_64-linux-android"

adb shell "rm -rf $DEV && mkdir -p $DEV"
adb push build-x86_64/test/tests "$DEV/tests"

# Shared libs the build emitted (libanira.so, libgtest*.so for a shared build; none
# for a static build) plus the enabled backend libs for this ABI (none when static,
# and modules/ is absent entirely for a no-backend build).
find build-x86_64 -name '*.so' -exec adb push {} "$DEV/" \;
[ -d modules ] && find modules -path '*x86_64/*.so' -exec adb push {} "$DEV/" \; || true

# The NDK C++ runtime the binary links against (needed even for static anira builds).
adb push "$NDK_SYSROOT_LIB/libc++_shared.so" "$DEV/"

# Model tree staged to the path baked into the build (ANIRA_EXTRAS_MODELS_DIR).
adb push extras/models "$DEV/models"
adb shell "chmod 755 $DEV/tests"

# One launch runs the whole bundled suite; assert the device-side exit code is 0.
adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV ./tests --gtest_brief=1; echo ANIRA_EXIT=\$?" | tee /tmp/out.txt
grep -q "ANIRA_EXIT=0" /tmp/out.txt
