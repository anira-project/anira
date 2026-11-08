State, sharing, capabilities and edges
======================================

What a registered engine needs beyond its lifecycle (:doc:`../custom_engines`): the declared state
it is fed, when anira shares one load between handlers, what it reports it can serve, and how it
states its edges.

.. _engines-state:

Declared state and the two buffers
----------------------------------

A model with a declared State pair (:ref:`usage-state`) is fed its state by anira without a copy:
the handler owns two buffers per pair, binds the model's State input to the one the last inference
wrote and its State output to the other, and flips the pair behind every successful inference. For
the engine this is the adapter rule and nothing more: the two descriptors name the pair's buffers in
turn from one call to the next, so an engine that read the data pointer at load or prepare would read
the wrong half every other inference. An engine that binds caller memory (an ``Ort::Value`` over the
descriptor, a ``torch::from_blob`` view) runs the pair without a copy; one that copies its inputs
copies the pair as it copies every other slot.

- An engine whose runtime wants the addresses to stay put, a captured graph above all (CUDA graphs,
  WebGPU replay), sets ``ANIRA_ENGINE_FLAG_STATE_ALIAS``: for every call of a plan of that engine
  anira binds **one** buffer as both halves and never flips, and the engine reads the state before it
  writes it (in place) or copies for itself. A failed call leaves the buffer as the engine left it
  (the last-good rule is the engine's to keep here), and a chunk of a plan without the bit in between
  flips the pair, so the addresses stay put across the chunks of aliasing plans alone (re-capture
  after such a switch).
- The pair is the model's, not a plan's: every plan of the variant runs it, in host memory in this
  pre-release (a ``host_domain`` on the State input overrides it), so a plan switch keeps the state
  whether the plans flip or alias, and the first inference of a new stream reads zeros either way.
- A model with a State pair is stateful: it is loaded once with no shared slot, and every handler of
  it is exclusive, so an engine with state of its own beside the pair builds it at ``prepare`` and
  re-initialises it at ``reset``.
- The two buffers take the spec's dtype, whatever the ring dtype is. A registered engine binds what
  its ``load`` accepts; a built-in one refuses at load, with ``ANIRA_ERROR_NOT_SUPPORTED`` naming the
  engine and the tensor, what its runtime cannot carry.

.. _engines-sharing:

Sharing and lifetime
--------------------

anira loads once per loaded model it **pools**. Two handlers share one loaded model and its shared
slots, and ``load`` runs once for both, when

- they run **the same engine object**, whichever pipelines they came from (two objects never share,
  even under one id over the same callbacks and ``user_data``);
- their **model configurations are equal**, the whole variant (every entry, spec and extension, since
  ``load`` may read any of it);
- their plans run on **the same provider** with the same provider options of their contexts; and
- their tensors **resolve to the same shapes**, with the same shared-slot count (the contract's
  warm-up is no part of it).

Whether a handler is exclusive is the handler's property, not part of the key: a stateful model is
loaded once too, with ``instances`` 0, and its handlers are prepared on it one by one, each with
``ANIRA_PREPARE_EXCLUSIVE`` and its own executor; they are the only handlers ``reset`` runs on.

The order for one engine serving two handlers of one model: ``init`` (once), ``load`` (once), the
``prepare`` of each handler (with an ``unprepare`` and a second ``prepare`` inside it when a
measured warm-up moved the entry count), the ``process`` calls of both, the ``unprepare`` of each as it is
re-prepared or destroyed, ``unload`` with the last of them, ``release`` once the handle, the
pipelines and the handlers are all gone. An ``unprepare`` always precedes the ``unload`` of the model
it ran on; for a handler alone a re-prepare unloads and loads again, since the old session is
released before the new one is created.

What a refusal owes, at every level: a refused ``init`` leaves the object uninitialised (the next
prepare calls it again); a refused ``load`` owes no ``unload``; a refused ``prepare`` owes no
``unprepare``, and the loaded model goes back (``unload`` when the handler was its last holder).
Whatever an earlier plan of the same prepare loaded and prepared goes back on the way out, through
its ``unprepare`` and ``unload``, and the handler is left unprepared. A refused
``anira_custom_engine_create`` owes no ``release``.

.. _engines-plugin-instances:

Sharing an engine across plugin instances
-----------------------------------------

The built-in engines share models between plugin instances without help: two instances with equal
settings load a model once. A custom engine does the same when the instances hand anira **the same
engine object**, kept once per plugin binary:

.. code-block:: c

    /* One engine object per plugin binary, created by the first instance. */
    static anira_custom_engine* shared_engine(void) {
        static anira_custom_engine* engine = NULL;   /* guard with a lock if instances are
                                                        created on several threads */
        if (engine == NULL) {
            anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
            desc.process = my_process;
            desc.load = my_load;
            desc.unload = my_unload;
            desc.prepare = my_prepare;
            desc.unprepare = my_unprepare;
            anira_custom_engine_create("com.example.myengine", &desc, &engine, NULL);
        }
        return engine;
    }

    /* Every plugin instance: */
    anira_pipeline_add_engine(pipe, shared_engine(), &err);

The static handle keeps the engine alive for the life of the binary: destroy it at the plugin's
unload, after the last instance. In C++ a function-local ``std::weak_ptr`` lets the engine go with
the last instance:

.. code-block:: cpp

    std::shared_ptr<anira::Engine> shared_gain() {
        static std::mutex mutex;
        static std::weak_ptr<anira::Engine> cache;       // one per plugin binary
        const std::lock_guard<std::mutex> lock(mutex);
        if (std::shared_ptr<anira::Engine> engine = cache.lock()) { return engine; }
        auto engine = std::make_shared<Gain>(0.5F);
        cache = engine;
        return engine;
    }

    // every plugin instance, each with its own Pipeline:
    pipe.add_engine(shared_gain());

Every instance's ``Pipeline`` holds the object's C engine, so two instances with equal configurations
load the model once. Keep the ``Pipeline`` while the instance runs: a registration made after every
``Pipeline`` holding the object is gone creates a new C engine, which does not share with handlers of
the old one. Two different plugin binaries never share a custom engine, and a plugin's engine code
never outlives the plugin. Two engine objects of one id load separately but share the model's
measured figure (:ref:`usage-budget-warmup`; ``anira.v2.custom`` excepted).

.. _engines-capabilities:

What a custom engine can serve here
-----------------------------------

A built-in engine's capabilities are the context's (:ref:`usage-capabilities`). A custom engine has
the same three parts, on the pipeline it belongs to: its **declared** vocabulary (``providers``), a
**measurement** (the ``query`` slot) and a **visible report**, the pipeline's capabilities.

``query`` (``anira_engine_query_fn``, a tail field) is the engine's own ``GetAvailableProviders``:
which of ``providers`` are usable here, now, as a bitmask over the list (bit *i*: ``providers[i]``;
a descriptor without a list serves the CPU path without a bit). It runs before ``init`` and any number
of times, on the main thread, at ``anira_handler_create`` and at the pipeline's capabilities entries,
with an ``anira_init_info``; it may log and never calls an entry that takes the lifecycle lock.
``anira_handler_prepare`` never calls it: the handler keeps its create's answer and checks the plans
against it again. A status other than ``ANIRA_OK`` fails the calling entry, naming the engine; a
declared provider the query cleared is refused for a candidate as "declares provider 'x' but its
query reports it unavailable here". In C++: ``Engine::query(const InitInfo&)``.

.. code-block:: c

    /* The rows a handler of this pipeline sees on this context: the context's, then one per
       added engine and provider usable here (engine ANIRA_ENGINE_CUSTOM, engine_id the
       engine's id): the listed providers the query keeps, in list order, or the CPU path
       alone for an engine without a list. Every call runs the engines' queries. */
    uint32_t count = 0;
    anira_pipeline_capabilities_backends(pipe, context, sizeof(anira_backend_id), &count, NULL);
    anira_backend_id* rows = malloc(count * sizeof *rows);
    anira_pipeline_capabilities_backends(pipe, context, sizeof(anira_backend_id), &count, rows);

    /* One edge, by its two ends: host memory to the engine on a provider (zero-copy to
       ANIRA_PROVIDER_CPU and XNNPACK; to every other a move the engine makes itself,
       ANIRA_MOVER_ENGINE, whose class is the memory kind's: ANIRA_EDGE_UNIFIED_TRANSFER on
       unified memory, ANIRA_EDGE_DEDICATED_TRANSFER on dedicated memory, ANIRA_EDGE_UNKNOWN on
       unknown memory, edge.memory_kind; Core ML's device is the Metal default device). */
    anira_edge_info edge = ANIRA_EDGE_INFO_INIT;
    edge.from_domain = ANIRA_DOMAIN_HOST;
    edge.to_engine = ANIRA_ENGINE_CUSTOM;
    edge.to_engine_id = "com.example.myengine";
    edge.to_provider = ANIRA_PROVIDER_COREML;
    anira_pipeline_capabilities_edge(pipe, context, &edge);

    /* The way back, from the engine to host memory: the same class and mover for the
       provider rule, the reason naming the readback where the other names the upload. */
    anira_edge_info back = ANIRA_EDGE_INFO_INIT;
    back.from_engine = ANIRA_ENGINE_CUSTOM;
    back.from_engine_id = "com.example.myengine";
    back.from_provider = ANIRA_PROVIDER_COREML;
    back.to_domain = ANIRA_DOMAIN_HOST;
    anira_pipeline_capabilities_edge(pipe, context, &back);

The rows' strings point into the pipeline's engines and the context's store: valid until the pipeline
is destroyed and until the context's next probe, a reason while the context lives. The backends list
asks no device's memory kind; the edge entry asks it for the row it reads, once per device API, and
the context keeps the answer. ``anira_handler_create`` asks the same question for every candidate's
provider of the engine, so what the report lists is what a handler runs on. In C++:
``pipe.capabilities(context).backends()`` / ``.edge(from, to)``
(:cpp:class:`anira::PipelineCapabilities`).

.. _engines-edges:

How the engine passes the slots
-------------------------------

The plan report says per slot where the edge takes the bytes and who moves them
(``anira_plan_slot.edge_class`` and ``moved_by``, :ref:`usage-edges`), and the contract's edge cost
drops a plan above its limit (:ref:`usage-contract-options`). anira hands a custom engine the chunk's
buffer itself, so by default its rows follow the **provider rule**: zero-copy on
``ANIRA_PROVIDER_CPU`` and XNNPACK; on any other provider the engine moves an input to its device and
an output back itself, ``ANIRA_EDGE_UNIFIED_TRANSFER`` where the device computes in unified memory,
``ANIRA_EDGE_DEDICATED_TRANSFER`` where it has its own and ``ANIRA_EDGE_UNKNOWN`` where no probe
established which (the plan row's ``memory_kind``); every edge-cost limit refuses an unknown edge. On
``ANIRA_PROVIDER_CUDA`` the device asked is CUDA device 0 (a custom engine's provider options are its
own vocabulary); a custom provider's name maps to no device, so its rows are ``ANIRA_EDGE_UNKNOWN``.

An engine that does something else says so in the descriptor's ``edge`` slot
(``anira_engine_edge_fn``): anira asks it once per slot right after each load, inputs first, on the
thread and under the lock of the load, with an ``anira_engine_edge`` already filled with the provider
rule's edge for the slot's direction (the same class both ways, the reason naming an input's upload
or an output's readback), and the engine overwrites what differs. The mover follows from the class:
whatever moves, the engine's code moves it (``ANIRA_MOVER_ENGINE``; ``ANIRA_MOVER_NONE`` on a
zero-copy edge). anira copies the reason before the call returns.

.. code-block:: c

    /* The engine copies its outputs out of its runtime's own tensors; its inputs it binds. */
    static anira_status ANIRA_CALL my_edge(void* loaded, uint32_t is_input, uint32_t slot,
                                           anira_engine_edge* out, void* user_data) {
        (void)loaded; (void)slot; (void)user_data;
        if (!is_input) {
            out->edge_class = ANIRA_EDGE_COPY;
            out->reason = "com.example.myengine: the runtime returns tensors of its own";
        }
        return ANIRA_OK;
    }

    desc.edge = my_edge;

A status other than ``ANIRA_OK`` fails the prepare with it, a class ``anira_edge_class`` does not
name with ``ANIRA_ERROR_ENGINE``; either way the load's ``unload`` follows. In C++ the statement is a
member of the loaded model, ``Engine::Loaded::edge(is_input, slot, provider_rule)``, whose base
returns the rule:

.. code-block:: cpp

    anira::EngineEdge edge(bool is_input, uint32_t, const anira::EngineEdge& rule) const override {
        if (is_input) { return rule; }
        return {.edge_class = ANIRA_EDGE_COPY,
                .reason = "com.example.myengine: the runtime returns tensors of its own"};
    }

On the web a JavaScript engine runs in the inference Workers, so the main thread states its edges
per direction, once for every input slot and once for every output slot, in ``Pipeline.addEngine``'s
``EngineOptions.edge`` (``{ inputs, outputs }``; per direction since the engine is added before the
model is known), and a direction left out keeps the provider rule's
(:doc:`../web-api/custom_inference_backends`).
