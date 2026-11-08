.. _usage-c-api:

The C API
=========

Everything of the guide is reachable from C11 through ``anira/abi/*.h`` (``<anira/anira.h>``
includes them all), the binary promise that ``anira/anira.hpp`` and the TypeScript package are
written over. ``anira.hpp`` is C++20 and header-only: compiled into your binary, it is not part of
the binary promise, needs no anira define, and can be included beside ``<anira/anira.h>``. Each C++ method is one C function named ``anira_<handle>_<verb>``:
``ModelConfig::tensor_layout`` is ``anira_model_config_set_tensor_layout``, ``TensorSpec::axis``
``anira_tensor_spec_set_axis``, ``ContextConfig::threads`` ``anira_context_config_set_threads``.

.. code-block:: c

    #include <anira/anira.h>

    anira_error err = ANIRA_ERROR_INIT;
    anira_model_config* cfg = NULL;
    anira_tensor_spec* in = NULL;
    uint32_t i = 0;
    if (ANIRA_FAILED(anira_model_config_create(&cfg, &err)) ||
        ANIRA_FAILED(anira_model_config_add_model_path(cfg, ANIRA_ENGINE_ONNXRUNTIME, NULL,
                                                       "model.onnx", &i, &err)) ||
        ANIRA_FAILED(anira_tensor_spec_create("audio_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED,
                                              &in, &err))) {
        fprintf(stderr, "%s: %s\n", anira_status_string(err.status), err.message);
        return 1;
    }
    anira_tensor_spec_set_axis(in, 0, ANIRA_AXIS_TIME, ANIRA_DYNAMIC);
    anira_model_config_add_input(cfg, in);   /* copied */
    anira_tensor_spec_destroy(in);
    /* ... anira_model_config_destroy(cfg) when done */

The conventions every entry follows:

- **Statuses:** an entry returns an ``anira_status``. Negative values are failures; ``ANIRA_OK`` and
  the positive values (``ANIRA_SUCCESS_UPGRADED``, ``ANIRA_MISSED``) are successes, so test with
  ``ANIRA_FAILED(status)``, never ``!= ANIRA_OK``: a minor release may add positive statuses.
- **Messages:** an entry that can fail for more than one reason takes a caller-owned
  ``anira_error`` last (initialise it with ``ANIRA_ERROR_INIT``; ``NULL`` when you want no message)
  and writes the status and a message into it (:doc:`../logging`). Out-parameters are left
  untouched on failure.
- **Handles** are opaque and single-owner: every ``*_create`` has a ``*_destroy`` (NULL-safe), and
  what you pass into another handle is copied.
- **Getters** are named after the field (``anira_tensor_spec_axis``, ``anira_model_config_input``,
  ``anira_contract_hard_budget``, ``anira_context_config_threads``). A count, a pointer or an enum is
  the return value, with a no-value answer for a ``NULL`` handle (0, ``NULL``, ``ANIRA_ENGINE_NONE``,
  a ``_FORCE32`` constant); anything else comes through out-parameters behind a status, written on
  success only (the Hard getters answer ``ANIRA_ERROR_WRONG_CONTRACT`` on an Async contract). What
  comes back is the handle's: a spec pointer from ``anira_model_config_input`` lives until the
  config's next ``add_input`` or ``add_output``.
- **Enumerations** (the plan report's rows, the capabilities, the latencies) are stride-explicit:
  ``out == NULL`` asks for the count, a buffer shorter than the count returns ``ANIRA_INCOMPLETE``,
  and records are written at the caller's ``element_size``.
- **Thread contracts:** every entry's documentation names where it may run (``[main-thread]``,
  ``[driver-thread]``, ``[inference-thread]``, ``[thread-safe]``, ``[callback-safe]``), and the
  real-time entries are ``ANIRA_NONBLOCKING`` (clang's ``nonblocking``), so a real-time body built
  of them can be checked at compile time.
- ``anira_check_abi(ANIRA_ABI_VERSION)`` says whether the library matches the headers; while the
  ABI major is 0 it demands the exact version.

A whole host
------------

The handler, prepared with the defaults (a measured budget, the ``UNTIL_STABLE`` warm-up) and run on
one planar block (:ref:`usage-handler`, :ref:`usage-host-tensors`):

.. code-block:: c

    anira_context* context; anira_pipeline* pipe; anira_handler* h; anira_contract* c;
    /* cfg: the model config above; mc: an anira_context_config (anira_context_config_create) */
    if (ANIRA_FAILED(anira_context_create(mc, &context, &err))) { return fail(&err); }
    anira_pipeline_create(&pipe, &err);
    const anira_model_config* variants[] = { cfg };
    anira_backend_id candidates[] = {
        { sizeof(anira_backend_id), ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CPU, NULL },
        { sizeof(anira_backend_id), ANIRA_ENGINE_LITERT,      ANIRA_PROVIDER_CPU, NULL },
    };
    anira_pipeline_add_inference(pipe, variants, 1, candidates, 2, &err);   /* NULL, 0: the default set */
    anira_status st = anira_handler_create(context, pipe, &h, &err);         /* copies everything */
    anira_pipeline_destroy(pipe); anira_model_config_destroy(cfg); anira_context_config_destroy(mc);
    if (ANIRA_FAILED(st)) { return fail(&err); }

    anira_contract_create_hard(block_size, block_size, sample_rate, &c, &err);
    /* an explicit budget instead: anira_contract_hard_set_budget(c, ANIRA_BUDGET_EXPLICIT, 5.0); */
    st = anira_handler_prepare(h, c, &err);                    /* err names the tensor or field */
    anira_contract_destroy(c);
    if (ANIRA_FAILED(st)) { return fail(&err); }
    host_set_latency(anira_handler_get_latency(h, 0));         /* constant until the next prepare */

    /* once: one planar tensor over the host's channel pointers (float** channels) */
    const int64_t shape[2] = { num_channels, block_size };
    anira_tensor block;
    anira_tensor_init_host_planar(&block, channels, (uint32_t)num_channels, ANIRA_DTYPE_F32, 2, shape);

    /* the process callback: in place is the same tensor twice, each side with its slot */
    block.shape[1] = (int64_t)num_samples;
    st = anira_handler_process(h, &block, 0, &block, 0, NULL);
    if (st == ANIRA_MISSED) { /* the buffers hold what the miss policy chose */ }
    if (ANIRA_FAILED(st)) { /* a refusal, recorded in anira_handler_rt_error */ }

    /* off the driver thread, at teardown */
    anira_handler_destroy(h);
    anira_context_destroy(context);

``test/contracts/install/main_c.c`` is a complete C host: a registered engine, a handler prepared
under a Hard contract and one block through ``anira_handler_process``.
