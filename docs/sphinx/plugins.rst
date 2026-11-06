Plugins and hosts
=================

What a plugin that embeds anira must know about the host it runs in: engine symbols, shared
engine libraries, a host's own tanh-lib, and unloading.

Keeping engine symbols private
------------------------------

**Symptom.** A plugin crashes, or fails with "OrtGetApiBase resolved to an ONNX Runtime that does
not support the API version", inside one host (Ableton Live 12 bundles its own ONNX Runtime) but
works standalone and elsewhere. **Cause:** engine symbols the plugin exports bind across modules
(ELF interposition, weak-symbol coalescing on macOS such as ``Ort::Global<void>::api_``), so the
plugin's engine calls reach the host's runtime.

1. **Use anira's build.** It links one copy of each engine, privately, and exports none of it.
   Engine linkage follows ``BUILD_SHARED_LIBS``: a shared ``libanira`` links the shared engines, a
   static one the static archives, and an engine without that linkage is disabled (LibTorch is
   shared-only, ExecuTorch static-only).

   - Static archives are linked on demand and hidden (``-load_hidden`` on macOS,
     ``--exclude-libs`` on ELF), and anira compiles with hidden visibility
     (``tanh_apply_symbol_policy``).
   - A static ``libanira`` has no export decoration (``ANIRA_API`` is empty under ``ANIRA_STATIC``,
     which the CMake package defines).
   - A shared ``libanira`` pins its export table to the ``anira_*`` entries
     (``tanh_set_export_allowlist(anira SYMBOL "anira_*")``, checked by ``anira_symbol_baseline``),
     since visibility alone cannot hide what a header stamps default-visible.
   - The ONNX Runtime adapter refuses a runtime older than the API anira was built against, at the
     engine's init (the first prepare that loads one of its models, or a context's capability
     probe).
2. **Your own engine code:** a translation unit that includes an engine header links that engine's
   target, ``target_link_libraries(your_plugin PRIVATE anira::anira anira::onnxruntime)``, and
   compiles with hidden visibility (``CXX_VISIBILITY_PRESET hidden``,
   ``VISIBILITY_INLINES_HIDDEN ON``). Weak symbols an engine header instantiates in your objects are
   hidden only by your own setting.
3. **Export the entry points alone.** With anira's CMake package, two lines do it for a CLAP plugin
   (``examples/clap-audio-plugin/CMakeLists.txt``):

   .. code-block:: cmake

       tanh_apply_symbol_policy(my_plugin)
       tanh_set_export_allowlist(my_plugin SYMBOL clap_entry)

   By hand: ``-Wl,-exported_symbols_list`` on macOS (``_bundleEntry``, ``_bundleExit``,
   ``_GetPluginFactory`` for a VST3) or a version script on Linux that lists the same names under
   ``global:`` and ``local: *;`` (a CLAP plugin's one name is ``clap_entry``). This is
   **required** on macOS for a static anira with ExecuTorch, whose force-loaded archives ld64
   cannot hide. Verify with ``nm -gU`` (macOS) or ``nm -D --defined-only`` (Linux): no engine
   symbol may appear. ``tanh_add_export_check`` (``cmake/tanh/check-exports.cmake``) runs the same
   check in CTest.
4. **Strip dead code** (optional): anira compiles with one section per function, and a shared
   ``libanira`` links with ``--gc-sections`` / ``-dead_strip`` / ``/OPT:REF``. A static one has no
   link of its own: pass the flag in your plugin's link. ``examples/clap-audio-plugin`` has the
   whole set, ``-fno-gnu-unique`` under GCC included.

Shared engine libraries are chosen by name
------------------------------------------

The steps above keep the *symbols* private. A shared engine library can still meet a host's copy,
because the loader picks it by name:

- **Linux:** one library per soname. Every ONNX Runtime 1.x is ``libonnxruntime.so.1`` (the PyPI
  wheel's, Firefox's), so a copy the process loaded first serves ``libanira`` too; ``libtorch.so``
  and ``libLiteRt.so`` likewise.
- **Windows:** a loaded module name is reused, whatever its folder. Dependencies are searched in
  the *application's* folder, then ``System32``, which on recent Windows holds an
  ``onnxruntime.dll`` (Windows ML), then ``PATH``.
- **macOS:** by install name (``otool -L``); ``DYLD_PRINT_LIBRARIES=1`` shows which copy loads.

anira's ONNX Runtime refuses an older runtime at its init (a newer one serves older API versions).
LibTorch, LiteRT and TFLite have no such check, and another version there is undefined behaviour.

**Solution:** build a plugin, or any module loaded into another process, against a static anira
(``BUILD_SHARED_LIBS=OFF``): the engines are then hidden archives inside it. LibTorch has no static
build, so a plugin that needs it carries this risk. With a shared anira, ship the engine libraries
next to the executable you control, in the versions anira pins. Separating them by name is
planned (`#232 <https://github.com/anira-project/anira/issues/232>`_).

.. _host-tanh-lib:

A host that uses tanh-lib itself
--------------------------------

Nothing to do: the two copies never meet, with a shared or a static anira, on every platform.

- anira builds tanh-lib's core from its own pin (``cmake/tanh-lib-pin.cmake``) under its own
  FetchContent name (``anira_tanh_lib``), compiles it hidden into ``libanira`` and installs nothing
  of it, in the inline namespace ``thl::anira``. Your tanh-lib is in its default namespace
  ``thl::v<major>_<minor>`` or one you chose.
- Code spells both ``thl::…``, but no symbol shares a mangled name (``_ZN3thl5anira…`` against
  ``_ZN3thl4v0_5…``), templates and inline functions included. So your logger, level, callback and
  threads stay yours, your tanh-lib may come at any version, before or after anira, and the link
  order does not matter.
- anira's concurrentqueue is renamed the same way (``anira_moodycamel``).
- CTest proves both: ``anira_abi_namespaces`` fails on any ``thl`` symbol outside ``thl::anira``
  and any ``moodycamel`` one. ``consumer_tanh_first`` links a host's own shared tanh-lib before
  ``anira::anira`` and checks that the loggers stay apart.

.. _plugin-library-unload:

Unloading a plugin
------------------

A host unloads a plugin (``dlclose`` / ``FreeLibrary``) once its last instance is gone, and a
thread still running inside it would then execute unmapped code. anira holds the invariant *no
anira thread exists once the last handler and context are destroyed*: the last handler's destructor
stops and joins the inference pool before it returns, and the log drain thread stops with the last
context or handler. The core's state is never destroyed while the library is loaded, so calling
into anira stays valid even from late static destructors.

- **Tear down from the instance or the module-exit entry point** (CLAP ``deinit``, VST3
  ``ExitDll``): every handler, then every context (:ref:`usage-teardown`). Never from the
  plugin's static destructors or ``DllMain``, which run inside the unload under the loader lock.
  On Windows call ``anira_shutdown()`` there after the handlers and contexts, as
  ``examples/clap-audio-plugin`` does: it is idempotent and refuses with
  ``ANIRA_ERROR_INVALID_STATE`` while a context or handler still exists.
- **The backstop on Linux and macOS:** a library-unload hook joins an idle pool. An inference in
  flight gets up to two seconds. One still running then can never finish during the unload, so the
  hook writes ``anira: the library is being unloaded while an inference is still running ...`` to
  stderr and aborts instead of hanging the host.
- **At process exit** the core's exit handler stops the pool earlier and waits up to two seconds,
  before the engines' static state is torn down. The hook then joins without a bound, and the
  process exits normally. Neither waits for the core's lifecycle lock when the exiting thread holds
  it, so an engine that ends the process from inside a prepare exits with its own status. On
  Windows nothing at ``DLL_PROCESS_DETACH`` may join a thread.
- **Your own inference threads** (``context.threads(0)`` and :cpp:class:`anira::InferenceThread`)
  are yours to stop before the library can unload.
- **A library that never unloads under GCC/Linux:** glibc keeps any object with an
  ``STB_GNU_UNIQUE`` symbol. Compile your plugin with ``-fno-gnu-unique`` and hidden visibility, as
  anira does. ``nm -DC your_plugin.so | grep ' u '`` must print nothing.
- **macOS never unloads a plugin with thread-local storage**, which the static ONNX Runtime, LiteRT
  and ExecuTorch archives have. That is harmless: it stays mapped until the host quits.

``test/contracts/unload`` covers the unload: it loads and unloads a plugin-shaped module from an
executable that does not link anira, checks that it is really unmapped (skipped where macOS keeps
the image mapped), and death-tests an unload and an exit with an inference in flight.
