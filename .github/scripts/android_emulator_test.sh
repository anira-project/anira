#!/usr/bin/env bash
# Stage the cross-built per-component test binaries, their shared-library
# dependencies and the model tree onto a booted Android emulator, then run each
# binary in a single launch and assert a zero exit. Invoked from
# build_test_mobile.yml inside reactivecircus/android-emulator-runner, which
# executes each line of an inline `script:` in its own shell — so the
# orchestration lives here as one process where the variables persist.
#
# Backend/linkage-agnostic: it pushes whatever .so the build actually produced (a
# shared build yields libanira.so + libgtest*.so + the enabled backend .so; a static
# build yields none, the binaries are self-contained bar the C++ runtime). Expects
# build-x86_64/ (configured with ANIRA_EXTRAS_MODELS_DIR=$DEV/models), extras/models/,
# and ANDROID_NDK_LATEST_HOME.
set -euo pipefail

DEV=/data/local/tmp/anira
NDK_SYSROOT_LIB="$ANDROID_NDK_LATEST_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/x86_64-linux-android"
TEST_BINARIES="test_utils test_scheduler test_backends test_handler"

adb shell "rm -rf $DEV && mkdir -p $DEV"
for t in $TEST_BINARIES; do
    adb push "build-x86_64/test/$t" "$DEV/$t"
done

# Shared libs the build emitted (libanira.so, libgtest*.so for a shared build; none
# for a static build) plus the enabled backend libs for this ABI (none when static,
# and modules/ is absent entirely for a no-backend build).
find build-x86_64 -name '*.so' -exec adb push {} "$DEV/" \;
[ -d modules ] && find modules -path '*x86_64/*.so' -exec adb push {} "$DEV/" \; || true

# The NDK C++ runtime the binary links against (needed even for static anira builds).
adb push "$NDK_SYSROOT_LIB/libc++_shared.so" "$DEV/"

# Model tree staged to the path baked into the build (ANIRA_EXTRAS_MODELS_DIR).
# LibTorch has no mobile build, so the (cache-restored) 75 MB RAVE model is
# dead weight on the device — drop it before staging.
rm -rf extras/models/third-party/ircam-acids/RAVE
adb push extras/models "$DEV/models"

# One launch per binary runs its whole suite; assert each device-side exit code
# is 0 (adb shell does not propagate exit codes, hence the echoed marker).
# Failures are collected, not fail-fast, so one run reports every broken suite.
FAILED=0
for t in $TEST_BINARIES; do
    adb shell "chmod 755 $DEV/$t"
    adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV ./$t --gtest_brief=1; echo ANIRA_EXIT=\$?" | tee "/tmp/out_$t.txt"
    grep -q "ANIRA_EXIT=0" "/tmp/out_$t.txt" || FAILED=1
done
exit $FAILED
