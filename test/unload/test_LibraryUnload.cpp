// Host-shaped library-unload test. See CMakeLists.txt in this directory.
//
// This executable deliberately links nothing of anira and includes no anira header:
// the module loaded below is the only thing that maps anira into the process, exactly
// like a plugin in a DAW. Every entry point is resolved by name; the signatures mirror
// test/unload/module_api.h.

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdlib>  // std::_Exit
#else
#include <dlfcn.h>
#include <sys/wait.h>  // IWYU pragma: keep (WIFSIGNALED)
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#endif

namespace {

constexpr const char* k_module_path = ANIRA_UNLOAD_MODULE_PATH;
#if defined(ANIRA_UNLOAD_LIB_PATH)
// Shared build: libanira itself must be gone after the module is unloaded.
constexpr const char* k_library_path = ANIRA_UNLOAD_LIB_PATH;
#else
constexpr const char* k_library_path = nullptr;
#endif

// How long a surviving thread gets to run into unmapped memory before the test ends.
constexpr auto k_post_unload_grace = std::chrono::milliseconds(300);

std::string basename_of(const std::string& path) {
    const auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::vector<std::string> pinned_runtimes() {
    std::vector<std::string> result;
    const std::string joined = ANIRA_UNLOAD_PINNED_LIBS;
    std::string::size_type start = 0;
    while (start <= joined.size()) {
        const auto end = joined.find('|', start);
        const std::string item =
            joined.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!item.empty()) { result.push_back(item); }
        if (end == std::string::npos) { break; }
        start = end + 1;
    }
    return result;
}

// Keep the shared backend runtimes (libtorch, ONNX Runtime, ...) mapped for the life of
// the process so that unloading the module never unloads them: their global destructors
// are not ours to test, and some are not unloadable at all. The handles are leaked on
// purpose. A missing runtime is a hard failure — silently not pinning would let the
// unmapped assertion pass or fail for the wrong reasons.
void pin_backend_runtimes() {
    for (const std::string& path : pinned_runtimes()) {
#if defined(_WIN32)
        HMODULE handle = LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        ASSERT_NE(handle, nullptr)
            << "could not pin backend runtime " << path << " (error " << GetLastError() << ")";
#else
        const void* const handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
        ASSERT_NE(handle, nullptr) << "could not pin backend runtime " << path << ": " << dlerror();
#endif
    }
}

// Is a library with this path currently mapped in the process?
bool is_mapped(const char* path) {
#if defined(_WIN32)
    HMODULE module = nullptr;
    return GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              basename_of(path).c_str(),
                              &module) != 0;
#else
    // RTLD_NOLOAD: succeeds only if the object is already loaded (glibc matches by
    // device/inode, dyld by path).
    if (void* handle = dlopen(path, RTLD_NOW | RTLD_NOLOAD)) {
        dlclose(handle);
        return true;
    }
#if defined(__APPLE__)
    // Belt and braces: dyld may know the image under an @rpath name.
    const std::string name = basename_of(path);
    for (uint32_t i = 0; i < _dyld_image_count(); ++i) {
        const std::string image = _dyld_get_image_name(i);
        if (image.size() >= name.size() &&
            image.compare(image.size() - name.size(), name.size(), name) == 0) {
            return true;
        }
    }
#endif
    return false;
#endif
}

using CreateFn = void* (*)();
using PrepareFn = void (*)(void*);
using ProcessFn = void (*)(void*, int);
using DestroyFn = void (*)(void*);
using IntFn = int (*)();
using UIntFn = unsigned int (*)();
using VoidFn = void (*)();

struct Api {
    CreateFn m_create = nullptr;
    PrepareFn m_prepare = nullptr;
    ProcessFn m_process = nullptr;
    DestroyFn m_destroy = nullptr;
    IntFn m_create_throwing = nullptr;
    UIntFn m_num_inference_threads = nullptr;
    IntFn m_has_inference_threads = nullptr;
    IntFn m_num_sessions = nullptr;
    IntFn m_has_core = nullptr;
    VoidFn m_shutdown = nullptr;
    VoidFn m_leak_thread = nullptr;
};

// The loaded module, plus the resolved API. Mirrors what a host does with a plugin.
class Module {
public:
    bool load() {
#if defined(_WIN32)
        m_handle = LoadLibraryExA(k_module_path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (m_handle == nullptr) {
            m_error = "LoadLibrary failed with error " + std::to_string(GetLastError());
            return false;
        }
#else
        m_handle = dlopen(k_module_path, RTLD_NOW | RTLD_LOCAL);
        if (m_handle == nullptr) {
            const char* error = dlerror();
            m_error = error != nullptr ? error : "dlopen failed";
            return false;
        }
#endif
        return resolve(m_api.m_create, "anira_test_create") &&
               resolve(m_api.m_prepare, "anira_test_prepare") &&
               resolve(m_api.m_process, "anira_test_process") &&
               resolve(m_api.m_destroy, "anira_test_destroy") &&
               resolve(m_api.m_create_throwing, "anira_test_create_throwing") &&
               resolve(m_api.m_num_inference_threads, "anira_test_num_inference_threads") &&
               resolve(m_api.m_has_inference_threads, "anira_test_has_inference_threads") &&
               resolve(m_api.m_num_sessions, "anira_test_num_sessions") &&
               resolve(m_api.m_has_core, "anira_test_has_core") &&
               resolve(m_api.m_shutdown, "anira_test_shutdown") &&
               resolve(m_api.m_leak_thread, "anira_test_leak_thread");
    }

    void unload() {
        if (m_handle == nullptr) { return; }
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(m_handle));
#else
        dlclose(m_handle);
#endif
        m_handle = nullptr;
        m_api = Api{};
    }

    const Api& api() const { return m_api; }
    const std::string& error() const { return m_error; }

private:
    template <typename Fn>
    bool resolve(Fn& out, const char* name) {
#if defined(_WIN32)
        auto* symbol =
            reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(m_handle), name));
#else
        void* symbol = dlsym(m_handle, name);
#endif
        if (symbol == nullptr) {
            m_error = std::string("missing entry point ") + name;
            return false;
        }
        out = reinterpret_cast<Fn>(symbol);
        return true;
    }

    void* m_handle = nullptr;
    Api m_api;
    std::string m_error;
};

// Can this module — and, in shared builds, libanira — be unloaded at all? Probed once
// per process by loading and unloading the module without touching anira. dyld (macOS)
// never unloads some images — in practice any image carrying thread-local variables,
// which the statically linked backend runtimes (ONNX Runtime, LiteRT, ExecuTorch) do —
// so such an image stays mapped forever there. It cannot crash on unload either (there
// is no unload), so the unmapped assertions and the death test have nothing to prove
// and are skipped on Apple platforms. On Linux/Windows an image that stays mapped is a
// real finding (e.g. a NODELETE object from STB_GNU_UNIQUE symbols under GCC) and fails.
const std::string& pinned_by_loader() {
    static const std::string k_pinned = [] {
        Module probe;
        if (!probe.load()) { return std::string(); }  // let the real tests report the load error
        probe.unload();
        if (is_mapped(k_module_path)) { return std::string("the test module"); }
        if (k_library_path != nullptr && is_mapped(k_library_path)) {
            return std::string("libanira");
        }
        return std::string();
    }();
    return k_pinned;
}

#if defined(__APPLE__)
constexpr bool k_skip_when_not_unloadable = true;
#else
constexpr bool k_skip_when_not_unloadable = false;
#endif

void expect_unmapped() {
    if (k_skip_when_not_unloadable && !pinned_by_loader().empty()) {
        GTEST_SKIP() << "dyld never unloads " << pinned_by_loader()
                     << " (thread-local variables from the statically linked backend "
                        "runtimes), so there is no unload to survive";
    }
    EXPECT_FALSE(is_mapped(k_module_path))
        << "the module is still mapped after unload — the loader refused to delete it "
           "(a NODELETE object, e.g. from STB_GNU_UNIQUE symbols under GCC?); the test "
           "cannot prove anything while it stays mapped";
    if (k_library_path != nullptr) {
        EXPECT_FALSE(is_mapped(k_library_path))
            << "libanira is still mapped after the module was unloaded";
    }
}

// The pool's threads count themselves as active when they enter their loop, i.e.
// asynchronously after prepare(); only the join side (count 0) is synchronous.
bool wait_for_num_inference_threads(const Api& api, unsigned int expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (api.m_num_inference_threads() != expected) {
        if (std::chrono::steady_clock::now() > deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// Give a thread that survived the unload the chance to run into unmapped memory: if one
// exists, the process dies here, which is the visible failure this test is for.
void grace_period() {
    std::this_thread::sleep_for(k_post_unload_grace);
}

class LibraryUnload : public ::testing::Test {
protected:
    void SetUp() override { pin_backend_runtimes(); }
};

}  // namespace

// The default lifecycle: once the last handler is destroyed no anira thread exists, so
// the host may unload right away.
TEST_F(LibraryUnload, DefaultPolicyLeavesNoThreadBehind) {
    Module module;
    ASSERT_TRUE(module.load()) << module.error();
    const Api& api = module.api();

    void* instance = api.m_create();
    ASSERT_NE(instance, nullptr);
    api.m_prepare(instance);
    api.m_process(instance, 50);
    EXPECT_TRUE(wait_for_num_inference_threads(api, 2));
    EXPECT_EQ(api.m_num_sessions(), 1);

    api.m_destroy(instance);
    // Joined synchronously by the last release — no waiting, no polling.
    EXPECT_EQ(api.m_num_inference_threads(), 0u);
    EXPECT_EQ(api.m_has_inference_threads(), 0);
    EXPECT_EQ(api.m_num_sessions(), 0);
    // The core outlives the sessions while the library is loaded (immortal) …
    EXPECT_EQ(api.m_has_core(), 1);

    // … and is reclaimed by the unload hook when the library goes.
    module.unload();
    expect_unmapped();
    grace_period();
}

// A host that unloads while an instance is still alive: the library-unload hook joins
// the pool (POSIX). On Windows there is no hook that may join; the plugin's module-exit
// entry point calls shutdown() instead, which is mirrored here.
TEST_F(LibraryUnload, UnloadWithLiveSessionIsJoinedByHook) {
    Module module;
    ASSERT_TRUE(module.load()) << module.error();
    const Api& api = module.api();

    void* instance = api.m_create();
    ASSERT_NE(instance, nullptr);
    api.m_prepare(instance);
    api.m_process(instance, 10);
    ASSERT_TRUE(wait_for_num_inference_threads(api, 2));

#if defined(_WIN32)
    api.m_shutdown();
    EXPECT_EQ(api.m_num_inference_threads(), 0u);
#endif

    // No destroy: the instance (and its session) is leaked, as a careless host would.
    module.unload();
    expect_unmapped();
    grace_period();
}

// Issue #106 from the host's angle: a failed construction leaves nothing behind, so the
// host may unload right after it.
TEST_F(LibraryUnload, FailedCreateLeavesNoState) {
    Module module;
    ASSERT_TRUE(module.load()) << module.error();
    const Api& api = module.api();

    EXPECT_EQ(api.m_create_throwing(), 1);
    EXPECT_EQ(api.m_num_sessions(), 0);
    EXPECT_EQ(api.m_num_inference_threads(), 0u);
    EXPECT_EQ(api.m_has_inference_threads(), 0);

    module.unload();
    expect_unmapped();
    grace_period();
}

// Two load/unload cycles in one process — a host scanning, then using, a plugin.
TEST_F(LibraryUnload, ReloadAfterUnloadWorks) {
    for (int cycle = 0; cycle < 2; ++cycle) {
        Module module;
        ASSERT_TRUE(module.load()) << module.error();
        const Api& api = module.api();
        if (cycle == 1) {
            void* instance = api.m_create();
            ASSERT_NE(instance, nullptr);
            api.m_prepare(instance);
            api.m_process(instance, 10);
            api.m_destroy(instance);
            EXPECT_EQ(api.m_num_inference_threads(), 0u);
        }
        module.unload();
        expect_unmapped();
    }
    grace_period();
}

// The negative control that proves the harness is sensitive: a thread that is still
// alive at unload must crash the process. If the child reaches _exit(0), the thread
// survived — meaning the unmapping was not real (a NODELETE or otherwise pinned
// library), which would have turned every test above into a false pass. The
// "threadsafe" death-test style is passed on the command line by CMake.
TEST(LibraryUnloadDeathTest, LeakedThreadCrashesOnUnload) {
    // Pin in the parent as well: the unloadability probe below loads and unloads the
    // module, and the backend runtimes must not be unloaded along with it.
    pin_backend_runtimes();
    if (k_skip_when_not_unloadable && !pinned_by_loader().empty()) {
        GTEST_SKIP() << "dyld never unloads " << pinned_by_loader()
                     << ", so a leaked thread cannot crash";
    }
#if defined(_WIN32)
    EXPECT_DEATH(
        {
            SetErrorMode(SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS);
            pin_backend_runtimes();
            Module module;
            if (!module.load()) { std::_Exit(3); }
            module.api().m_leak_thread();
            module.unload();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::_Exit(0);
        },
        "");
#else
    EXPECT_EXIT(
        {
            pin_backend_runtimes();
            Module module;
            if (!module.load()) { _exit(3); }
            module.api().m_leak_thread();
            module.unload();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            _exit(0);
        },
        [](int status) {
            // NOLINTNEXTLINE(misc-include-cleaner) — WIFSIGNALED comes from <sys/wait.h>
            return WIFSIGNALED(status);
        },
        "");
#endif
}
