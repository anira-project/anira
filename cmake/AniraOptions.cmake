# ==============================================================================
# AniraOptions.cmake — the single place every user-facing build option lives.
# Included near the top of the root CMakeLists (before project()).
# Cross-option validation lives in cmake/AniraValidate.cmake.
# ==============================================================================

# --- Library type --------------------------------------------------------------
option(BUILD_SHARED_LIBS "Build anira as a shared library" ON)

# --- Components ----------------------------------------------------------------
option(ANIRA_WITH_BENCHMARK "Build the library with benchmarking capabilities" OFF)
option(ANIRA_WITH_EXAMPLES  "Add example targets (juce plugin, benchmarks, minimal inference and model examples)" OFF)
option(ANIRA_WITH_INSTALL   "Add install targets" OFF)
option(ANIRA_WITH_TESTS     "Add build tests" OFF)
option(ANIRA_WITH_DOCS      "Build documentation" OFF)
option(ANIRA_WITH_LOGGING   "Enable logging printouts" ON)
option(ANIRA_WITH_RTSAN     "Enable RealtimeSanitizer (rtsan) checks (requires clang 20)" OFF)
option(ANIRA_BUILD_WASM     "Build WebAssembly module (requires Emscripten toolchain)" OFF)

# --- Inference backends --------------------------------------------------------
# Multiple backends can be enabled. LiteRT (LiteRt* C API) is the default
# TensorFlow-family backend; TFLite (legacy TfLite* C API) is the SAME underlying
# runtime through the older API, so the two are mutually exclusive (their static
# libraries export the same TfLite* symbols). Enforced in cmake/AniraValidate.cmake.
option(ANIRA_WITH_LIBTORCH    "Build with the LibTorch backend" ON)
option(ANIRA_WITH_ONNXRUNTIME "Build with the ONNX Runtime backend" ON)
option(ANIRA_WITH_LITERT      "Build with the LiteRT backend (LiteRt* C API; runs .tflite via the CompiledModel runtime)" ON)
option(ANIRA_WITH_TFLITE      "Build with the legacy TensorFlow Lite backend (TfLite* C API); mutually exclusive with ANIRA_WITH_LITERT" OFF)

# --- Pre-built backend download ------------------------------------------------
# Backends are downloaded from the anira-project/backends release with this tag.
# Integrity is checked live: anira fetches each asset's published sha256 from GitHub
# at configure (when reachable) and re-downloads anything that changed upstream or
# downloaded incompletely — nothing is pinned in-repo. Empty -> built-in default tag.
set(ANIRA_BACKENDS_VERSION "" CACHE STRING "anira-project/backends release tag to download (default: built-in)")

# Skip the live integrity check (no GitHub query at configure). For fully offline /
# reproducible builds; a backend already in modules/ is then reused as-is.
option(ANIRA_BACKENDS_SKIP_REMOTE_CHECK "Don't query GitHub for backend integrity at configure" OFF)

# Backend linkage follows BUILD_SHARED_LIBS (shared anira -> shared backends, static
# anira -> static backends). Decouple a single engine, bring your own, or use a custom
# source via the per-engine variables (<ENGINE> = LIBTORCH | ONNXRUNTIME | TFLITE | LITERT):
#   -DANIRA_<ENGINE>_LINKAGE=shared|static
#   -DANIRA_<ENGINE>_ROOTDIR=/path/to/prebuilt   (a tree with include/ + lib/)
#   -DANIRA_<ENGINE>_URL=...  [-DANIRA_<ENGINE>_SHA256=...]
