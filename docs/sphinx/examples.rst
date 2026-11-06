Examples
========

The examples build with ``-DANIRA_WITH_EXAMPLES=ON`` (the benchmarks with
``-DANIRA_WITH_BENCHMARK=ON``); the web examples live in ``web/examples/``.

.. _examples-models:

The bundled models
------------------

Every bundled model ships a model file and a contract file next to its model directory
(``<model>.model.json``, ``<model>.contract.json``; :ref:`usage-json`), named for the examples and
the tests by ``extras/models/model_files.h``:

.. list-table::
   :header-rows: 1
   :widths: 30 42 28

   * - Files (``extras/models/``)
     - Model, tensors
     - Runs with
   * - ``cnn/cnn``, ``medium_cnn``, ``small_cnn``
     - the steerable-nafx CNN in three sizes; one Streamed tensor each way
     - anira's default fill
   * - ``hybrid-nn/hybridnn``
     - GuitarLSTM; overlapping windows
     - ``HybridNNProcessor``
   * - ``stateful-rnn/rnn``
     - a stateful LSTM (``ANIRA_MODEL_STATEFUL``)
     - the default fill
   * - ``model-pool/gain``, ``stereo_gain``
     - SimpleGainNetwork, mono and stereo; the gain is a Static input
     - the default fill
   * - ``model-pool/stateful_accumulator``
     - StatefulAccumulatorNetwork, stereo, with a declared State pair (:ref:`usage-state`)
     - the default fill
   * - ``model-pool/typed_gain_<dtype>``
     - TypedGainNetwork at ``int16``, ``int8`` and ``float16`` on every engine; the ``_unplanned``
       files run an ExecuTorch export whose I/O the memory plan leaves to the caller, bound in place
     - the default fill (model files only)
   * - ``third-party/ircam-acids/rave_funk_drum``, ``_encoder``, ``_decoder``
     - RAVE funk drum whole, as an encoder and as a decoder (LibTorch only; the encoder's and the
       decoder's contracts add ``"on_miss": "zeros"``)
     - the default fill

For the tests, ``model-pool/`` also holds an ONNX model per dtype anira names and a TFLite model with
quantised int8 I/O. The contract files run the defaults (a measured budget, ``UNTIL_STABLE``), so a
model's latency depends on the machine (:ref:`latency-measured`). Every example loads a model in a
few lines:

.. code-block:: cpp

    anira::ModelConfig model_config = anira::ModelConfig::from_file(k_cnn_model_json);
    anira::ContractHandle contract = anira::ContractHandle::from_file(k_cnn_contract_json);
    anira::Pipeline pipeline;
    pipeline.add_inference(model_config);
    anira::InferenceHandler handler(context, pipeline);

The files set no instance ceiling (the pool's count, ``ANIRA_INSTANCES_AUTO``), and the plugins pass
no context configuration (the default, half the hardware threads). Every plugin also carries the
bypass engine of ``extras/models/BypassEngine.h`` (``org.anira.examples.bypass``, a custom engine
that needs no model file and delays the input by the latency anira reports) behind its "BYPASS"
backend choice ("Bypassed" in the CLAP plugin). The one place a model is built in code is the benchmark sweeps: ``CNNConfig.h``,
``HybridNNConfig.h`` and ``StatefulRNNConfig.h`` build the configurations with the hop, the batch
count or the chunk following the host buffer; a test keeps each builder equal to its file at the
default size.

.. _examples-plugins:

The plugins
-----------

Both plugins declare the host's real block range on the Hard contract, since neither host promises a
fixed block: JUCE's ``prepareToPlay(sampleRate, samplesPerBlock)`` names the largest block only
(``hard_host_blocks(1, samplesPerBlock, sampleRate)``), CLAP's ``activate(sampleRate,
minFrameCount, maxFrameCount)`` both ends (``hard_host_blocks(max(1, minFrameCount), maxFrameCount,
sampleRate)``). anira sizes the latency for every block size of the range, which the plugins report
to the host (``setLatencySamples``, the CLAP latency extension) for compensation; a fixed contract
under a host that changes its block size would miss blocks instead. A host that changes the size
between calls can still miss a block, rarely (`#228 <https://github.com/anira-project/anira/issues/228>`_).

Both offer as backends the engines this build compiled in (``anira::enabled_engines()``) that the
model file has an entry for, then the bypass, last. The default is ONNX Runtime where the build has
it; JUCE variant 3 defaults to LibTorch, since the stateful LSTM's ONNX export starts every
inference from a zero state. A backend whose engine cannot run the model on the machine (no plan in
the report) runs the bypass plan and says so.

Both act on the verdict of the plan they run (:ref:`usage-plan-report`): a ``NOT_REALTIME``
backend stays on the bypass plan until the user chooses "Try nonetheless", and an offline render is
never held.

**JUCE** (``examples/juce-audio-plugin/``, target ``anira-juce-plugin-example_VST3``):

- ``MODEL_TO_USE`` picks the model (``-DMODEL_TO_USE=2``, default 1):

  .. list-table::
     :header-rows: 1
     :widths: 10 90

     * - Value
       - Model
     * - 0
       - the steerable-nafx CNN
     * - 1
       - the same CNN with every file compiled into the plugin: the model file, the contract file
         and the four exports in JUCE's ``BinaryData``, the config from the embedded text
         (``ModelConfig::from_json``) and each entry's source swapped for the embedded file of its
         path's name (``set_model_bytes``, ``ANIRA_BYTES_BORROW``), so the plugin reads nothing
         from disk
     * - 2
       - GuitarLSTM with ``HybridNNProcessor``
     * - 3
       - the stateful LSTM
     * - 4, 5
       - SimpleGainNetwork, mono and stereo; the gain parameter is the Static input
         (``set_static_input``)
     * - 6
       - RAVE funk drum
     * - 7
       - RAVE as an encoder and a decoder handler; the decoder anchors on its audio output, so both
         handlers prepare with the host's block range and rate, and the latent frames pass from the encoder's ``pop_data`` to the decoder's ``push_data`` as one
         planar tensor

- The "Backend Type" parameter stores its choice in an atomic and ``processBlock`` applies it: the
  plan that runs the chosen engine (``PlanReport::find``, then ``set_plan``).
- ``PluginEditor`` holds the parameters' controls, a verdict line and a "Try nonetheless" button,
  shown while the chosen backend waits, polling the processor at 4 Hz. ``prepareToPlay`` reads the
  row of each backend's plan and writes it to ``juce::Logger`` (variant 7 both handlers' rows,
  keeping the worse verdict)
  ("anira: ONNX: budget X ms (measured X ms, median X ms, N warm-up inferences, stable); real-time
  factor F of n (load L): comfortable"). A ``NOT_REALTIME`` backend runs the BYPASS plan until the
  click, unless ``isNonRealtime()`` (a bounce runs it whatever the verdict); a marginal one plays,
  its line saying so. Per backend the processor keeps the worst verdict the user accepted, so a
  re-prepare asks again only when the verdict got worse.

**CLAP** (``examples/clap-audio-plugin/``, target ``anira-clap-demo``; ``anira-clap-demo.h/cpp``,
the entry point ``anira-clap-demo-pluginentry.cpp``):

- A third parameter, "Try nonetheless" (stepped, default off), kept as the user set it across
  activations.
- The ``clap.render`` extension: ``CLAP_RENDER_OFFLINE`` runs the chosen backend whatever its
  verdict.
- ``activate`` prints each backend's row and, while the parameter is off in a real-time render,
  that a ``NOT_REALTIME`` backend "stays Bypassed until 'Try nonetheless' is on".

.. _examples-benchmarks:

Benchmarks, minimal inference, C
--------------------------------

- ``examples/benchmark/`` (:doc:`benchmarking`), on the fixture of ``examples/benchmark/fixture/``,
  run as CTest tests (``ctest -R '^Benchmark' -j 1``): ``simple-benchmark`` (GuitarLSTM with
  ``HybridNNProcessor`` on ONNX Runtime), ``cnn-size-benchmark`` (the CNN's sizes on every engine of
  the build, the bypass engine as the baseline), ``advanced-benchmark`` (buffer sizes, models and
  engines, the ``ClearEngine`` of ``ClearEngine.h``, ``org.anira.examples.clear``, as the baseline of
  anira's own path). Each run's header says what prepare measured; nothing is held.
- ``examples/minimal-inference/``: the minimal code per engine. Each reads the model path and the
  shapes from anira's model files (``minimal_model.h``, over the read-back getters) and drives the
  engine's own API, without the anira runtime. The ExecuTorch example does not link anira (anira
  embeds its own ExecuTorch runtime, which a second copy must stay isolated from) and spells the
  path and the shapes out.
- ``test/contracts/install/main_c.c`` (``consumer_c``): a C host against the installed package, a
  registered engine, a handler prepared on the defaults, every plan's figure, factor and verdict
  printed, one block through ``anira_handler_process``.

.. _examples-web:

Web
---

``web/examples/model-file`` runs a bundled model file in the browser,
``web/examples/processor-and-engine`` a processor and an engine written in TypeScript
(:doc:`web-api/index`). After each prepare
both log every plan's row; on a live context a ``NOT_REALTIME`` plan holds the node
(``audioNode.hold(true)``) until a "Try nonetheless" button releases it, a marginal one shows a note,
and an offline render never holds.

Elsewhere
---------

`nn-inference-template <https://github.com/Torsion-Audio/nn-inference-template>`_ is a more elaborate
JUCE / VST3 plugin on anira, with a full GUI, parameter and state handling: a starting point for a
plugin of your own.
