The C ABI registry and its gates
================================

The registry
------------

The headers under ``include/anira/abi/`` are generated, never edited by hand. The one source is
``abi/anira.yml``:

.. code-block:: bash

    pip install pyyaml
    python3 -B tools/abi/gen.py --repo . --write     # or the anira_abi_regen build target
    python3 -B tools/abi/gen.py --repo . --check     # exit 2 on any drift
    python3 -B tools/abi/gen.py --repo . --diff-against v3.0.0-alpha.2   # additions and BREAKING lines

``--write`` validates the registry against the header conventions of
``docs/anira-v3-architecture.md`` and rewrites:

- the headers, and ``include/anira/anira_em.h``;
- ``web/src/abi/enums.ts``, ``layout.ts`` and ``bindings.ts``;
- the symbol lists (``abi/symbols-<major>.txt``, ``symbols-draft.txt``, ``symbols-emscripten.txt``)
  and the wasm export list;
- ``src/capi/generated/`` (the status texts, ``struct_sizes.inc`` behind ``anira_sizeof``) and
  ``test/abi/generated/``;
- ``abi/layout-<major>.txt`` and the enum pages under ``docs/sphinx/api/enum/``.

Commit the registry with every regenerated file. The ``anira_abi_generate`` test and the
``build_web`` workflow run ``--check``. Exit codes: 0 clean, 1 a registry error, 2 drift, 3 usage.

The generated files carry a ``.clang-format`` with ``DisableFormat`` and a
``NOLINTBEGIN(readability-identifier-naming)`` block, so the pinned root configs stay untouched.
``layout.ts`` is written the way prettier prints it, and ``web/.prettierignore`` lists
``enums.ts`` and ``bindings.ts``.

``bindings.ts`` is the TypeScript face of the C ABI: the class ``AniraBindings``, one method per
entry under its C name, derived from the registry's types. A new entry needs no TypeScript.

- the ``anira_error*`` is made on the instance's stack, and a failed status throws an
  ``AniraError`` with the entry's message (or the status text);
- a ``const char*`` is a JS string (with its ``size_t len`` computed when one follows);
- what an entry writes through a non-const pointer to a scalar, an enum, a handle or a
  ``const char*`` is returned (one value, or an object by parameter name); a ``_to_json`` returns its
  text;
- ``anira_bool`` is a ``boolean``, a 64-bit integer a ``bigint``, a pointer or a size a ``number``;
  an enumeration entry and a record the caller passes in keep their pointers.

Writing a registry entry
------------------------

- **Kinds:** ``verbatim``, ``typedef``, ``define``, ``enum``, ``handles``, ``struct``, ``union``,
  ``callback``, ``function``.
- **A struct field or union arm** has exactly one shape:

  - ``type`` (a C type, optional ``array`` of 1 or more; in a Tier-1 record a fixed-width scalar or
    an earlier Tier-1 record);
  - ``ptr`` (an ``ANIRA_PTR`` slot);
  - ``fields`` (a named member of an unnamed struct type, arms only);
  - ``arms`` (a named member of an unnamed union type, Tier-1 structs only).

  No truly anonymous struct, which ``-pedantic`` C++ refuses. The generator refuses a member name
  repeated within one C scope (``duplicate field <name> in <scope>``).
- **A union** is a Tier-1 record (``tier: 1`` with ``size`` and ``align``); the layout table has a
  row per member, written as its C designator (``field anira_sync_token.u.vk.value offset 16
  size 8``).
- **Record attributes:**

  - ``struct_id: ANIRA_STRUCT_<X>`` ties a record to ``anira_sizeof``, and every Tier-1 record
    carries one;
  - ``function_type: true`` makes a callback a function type, for an ``ANIRA_PTR`` slot only;
  - ``forward: true`` pre-declares a struct;
  - ``padding_free: true`` on a Tier-2 struct makes ``test_layout.c`` assert ``sizeof`` equals the
    sum of its members on every target (a tail slot that would leave a hole is followed by an
    explicit ``reserved``).

- **Header kinds:**

  - ``draft: true`` puts a header under ``include/anira/abi/draft/`` with ``status: draft``
    functions only. Their names go to ``abi/symbols-draft.txt``, and no umbrella but
    ``anira_all.h`` includes it. Promotion moves a name between the symbol lists without renaming
    it. ``--diff-against`` lists a changed or promoted draft function as an addition, while a
    changed frozen union arm, signature, thread tag or ``nonblocking``, and a removed
    ``callback_safe``, are breaks.
  - ``platform: emscripten`` makes a companion for one target (``file: anira_<name>.h``, its own
    ``guard:``). Its body sits inside ``#if defined(__EMSCRIPTEN__) || defined(DOXYGEN)``, and it
    declares functions, callbacks, typedefs and defines only. No header of the registry includes
    it. Its names go to ``abi/symbols-emscripten.txt`` and the wasm export list.

- **Thread tags:** every function carries one from the architecture document's vocabulary
  (``main-thread``, ``driver-thread``, ``inference-thread``, ``thread-safe``, with qualifiers),
  ``callback_safe`` where it applies and ``nonblocking: true`` where the body is real-time. The tag
  becomes the ``@par Thread contract`` line.

  - The generator refuses a nonblocking entry with a 64-bit argument or an ``anira_error*``.
  - The dual tag ``driver-thread | inference-thread`` (a processor callback, the context accessors,
    the default bodies) requires ``nonblocking: true`` on an entry, but no attribute on a callback
    typedef: whether a processor or an engine body is real-time is its own promise (the
    ``flags``).

- **The 64-bit allowlist** is closed: ``anira_now_ns``, ``anira_tensor_init_vulkan``,
  ``anira_tensor_init_opaque_fd``, ``anira_tensor_init_wgpu_buffer``, ``anira_tensor_init_dmabuf``
  and ``anira_tensor_init_iosurface``, whose parameters are vendor handles at their wire width. A
  seventh needs the same argument.
- **Names:** the two descriptors share one scheme.

  - ``processor`` / ``engine`` on every level that belongs to one side (``anira_processor_desc`` /
    ``anira_engine_desc``, ``_ctx``, the ``_fn`` typedefs, ``anira::Processor`` /
    ``anira::Engine``);
  - no side on what both lifecycles share (``anira_init_info``, ``anira_prepare_info``,
    ``anira_phase``);
  - ``FLAG`` in every flag define;
  - ``Prepared`` / ``Loaded`` for what ``prepare`` / ``load`` hand back;
  - callbacks shaped ``init(info, user_data)``, ``load(info, user_data, &loaded)``,
    ``prepare(info, [loaded,] user_data, &prepared)`` and per chunk ``(ctx, prepared, user_data)``.

  The provider is the engine's twin wherever the backend pair travels (``provider`` /
  ``provider_id`` beside ``engine`` / ``engine_id``, the pair rule). The JSON words of both enums
  live in ``src/capi/words.h``.
- **Layout:** while ``ANIRA_ABI_MAJOR`` is 0, ``--write`` rewrites ``abi/layout-0.txt`` with the
  headers. From the v3.0.0 freeze on, a Tier-1 layout changes only with the ABI major.

Every entry needs its body in the same pull request, on Emscripten too: three presence gates link
every promised and draft name on every leg (below).

The gates
---------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - target
     - what it proves
   * - ``anira_symbol_baseline``, ``anira_abi_link``, the AniraWeb link
     - The shared library exports exactly ``abi/symbols-0.txt`` and ``abi/symbols-draft.txt`` (``nm``
       or ``dumpbin``; only linker-defined names and the exports a PE engine archive forces are
       tolerated). ``anira_abi_link`` (``test/abi/generated/link_probe.c``) takes the address of
       every name on static and shared legs, and the wasm link exports
       ``web/src/abi/exports_wasm.txt``.
   * - ``anira_exports``, ``anira_abi_namespaces``
     - No engine runtime and no private dependency (``thl``, ``moodycamel``, ``nlohmann``) in the
       library's or a plugin-shaped module's export table; no ``thl`` symbol outside ``thl::anira``
       and no ``moodycamel`` one anywhere in the library (``cmake/abi-namespaces.cmake``).
   * - ``anira_abi_layout``
     - Every Tier-1 ``sizeof`` / ``offsetof`` and enum width, natively and under node, against
       ``abi/layout-0.txt``.
   * - ``anira_header_c11``, ``_cxx17``, ``_cxx20``, ``_compat``, ``_compat_with_hpp``,
       ``anira_header_isolation``
     - Every public header alone and the umbrellas together, as C11 and C++17 with no anira define
       under strict flags, ``anira.hpp`` and the shim as C++20; isolation fails when anything
       private resolves through ``anira::anira``'s usage requirements. A CTest on desktop, a member
       of ALL on Android, iOS and WebAssembly.
   * - ``InstallConsumer`` (``test/contracts/install``)
     - The installed package, shared and static: ``consumer`` (``anira.hpp``), ``consumer_c`` (pure
       C11 over a registered engine), ``consumer_compat`` (the 2.x shim), ``consumer_engine`` (an
       engine through ``anira::onnxruntime``), ``consumer_tanh_first`` (a host's own tanh-lib linked
       first never sees anira's records) and a plugin-shaped module that exports no engine symbol.
   * - the RTSan leg, ``anira_abi_rt_contract``
     - The suite under RealtimeSanitizer (``desktop-tests-rtsan``, the unload harness excepted), and
       ``-Werror=function-effects`` on the consumer-shaped C targets and the C++20 header gates
       (clang 20 or later).
   * - ``anira_abi_wasm_gates``
     - Under node, on the ``wasm-abi`` preset:
       ``anira_abi_layout`` (the wasm32 column of the layout), ``anira_em_smoke`` (a JavaScript
       processor and engine through ``anira_em.h``), ``anira_em_smoke_thread`` (a host-driven loop
       on a Wasm Worker), ``anira_em_mutex`` (two threads building and tearing down handlers),
       ``anira_em_pool`` (a context of two Worker threads with a worker module),
       ``anira_em_console`` (one warning reaches ``console.warn`` once), ``anira_em_runtime`` (a
       library module keeps its runtime alive) and ``anira_em_oom`` (a full heap refuses a handler
       with ``ANIRA_ERROR_OUT_OF_MEMORY``). ``build_web`` then checks that ``web/wasm/AniraWeb.d.ts``
       exports exactly the three symbol lists plus ``_malloc`` and ``_free``.

An ``abidiff`` against the last tag joins the gates at the v3.0.0 freeze.
