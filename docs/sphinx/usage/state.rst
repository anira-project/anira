.. _usage-state:

Declared state
==============

A model whose state is explicit, an output that is the next inference's input (an exported GRU or
LSTM hidden state, a streaming convolution's cache), declares the pair and anira runs the feedback:
no processor is needed. Both halves have the role ``state``, and the input names the output it is
fed from. The bundled StatefulAccumulatorNetwork
(``extras/models/model-pool/stateful_accumulator.model.json``) is the smallest instance: a stereo
stream of 64 samples per inference and a ``[1, 2, 2]`` state, per channel a running sum and a
block counter. Trimmed to one engine, and without the windows (the fixed Time extent is the window):

.. code-block:: json

    {
      "models": [
        { "engine": "onnxruntime",
          "path": "example-models/StatefulAccumulatorNetwork/models/stateful_accumulator_network_stereo.onnx" }
      ],
      "inputs": [
        { "name": "state_in", "dtype": "float32", "role": "state",
          "axes": [["batch", 1], ["channel", 2], ["any", 2]], "state_source": "state_out" },
        { "name": "data", "dtype": "float32", "role": "streamed",
          "axes": [["batch", 1], ["channel", 2], ["time", 64]] }
      ],
      "outputs": [
        { "name": "processed_data", "dtype": "float32", "role": "streamed",
          "axes": [["batch", 1], ["channel", 2], ["time", 64]] },
        { "name": "state_out", "dtype": "float32", "role": "state",
          "axes": [["batch", 1], ["channel", 2], ["any", 2]] }
      ]
    }

In code: ``spec.state_source("state_out")``; in C ``anira_tensor_spec_set_state_source(spec,
"state_out")``.

What anira does
---------------

- The handler owns **two buffers per pair**, in the spec's shape and dtype, on a 64-byte boundary,
  zeroed at create and at every prepare. On the inference thread, first in every inference and
  ahead of ``before_inference``, the State input is bound to one buffer (the state the last
  inference produced) and the State output to the other; the engine reads and writes them like
  any slot; last, behind ``after_inference``, the pair flips. Nothing is copied by anira: both
  steps are field fills outside the timed phase (``ANIRA_PHASE_INFERENCE`` stays the engine call
  alone), and each engine binds the buffers as it binds any slot (:ref:`usage-edges`).
- Under a plan whose engine sets ``ANIRA_ENGINE_FLAG_STATE_ALIAS`` both halves are one stable buffer
  that the engine updates in place; addresses stay put across such plans (a chunk of a flipping plan
  in between moves the pair, so a captured graph is captured again after a switch).
- The pair is the model's: every plan of the variant runs it, in host memory in this pre-release
  (a ``host_domain`` on the State input overrides it), so **a plan switch keeps the state**.
- A failed inference does not flip, so the read side keeps the last good state (an aliasing
  engine's buffer holds what the failed call left); a dropped chunk never runs and does not advance
  it; a missed block does not affect it.
- ``reset`` and ``prepare`` re-initialise it to zeros, the reset on the inference thread at the
  first inference of the new stream (:ref:`usage-reset`). The warm-up pass feeds the pair like any
  inference; the session sized again after it zeroes both buffers, so the pass reaches neither the
  state nor the stream.
- A model with a pair runs as ``ANIRA_MODEL_STATEFUL`` whatever its ``state`` says: an exclusive
  handler, one inference at a time and in submission order, which makes the feedback well defined
  and lets the buffers go without a latch.

What the host and a processor see
---------------------------------

A State spec may stand anywhere in its list and has a slot like every other tensor (the
accumulator's state is first in, last out, so its stream runs as ``process(in, 1, out, 0)``). No
Hard entry carries it: a single form naming its slot is ``ANIRA_ERROR_INVALID_ARGUMENT``, in a form
over every slot its position holds the empty tensor (a rank of 1 or more with an extent of 0) and
its ``delivered`` count is 0 (a zeroed record has rank 0 and is refused there: it is not the empty
tensor), and the latency and the available samples of a State output read 0.
The state is not host-readable in this pre-release, and it starts at zeros.

A processor's two hooks see the state at its slot: ``anira_processor_input_tensor`` in
``before_inference`` is what the engine will get, ``anira_processor_output_tensor`` in
``after_inference`` what the next inference reads, and the processor may read or alter either. In
``pre_process`` and ``post_process`` a State slot answers ``ANIRA_ERROR_INVALID_STATE``.

Validation
----------

At create and again at prepare, ``ANIRA_ERROR_CONFIG`` naming the tensor:

- a State input without a source; a source that names no output, or an output of another role; a
  State output named by no input, or by two;
- a source on an output spec or on a spec of another role (``anira_tensor_spec_set_state_source``
  refuses the latter with ``ANIRA_ERROR_INVALID_ARGUMENT``, the loader ``state_source`` on an output
  with ``ANIRA_ERROR_JSON``);
- two halves of unequal dtype or shape; a Time axis, a window, a time ratio or a latency on a State
  spec; a ring dtype or the anchor naming one.

``ANIRA_ERROR_NOT_SUPPORTED``: a transposed layout on a State spec (a view layout prepares). A pair
takes any dtype anira names; whether an engine takes it is the engine's (a built-in adapter refuses
at load what its runtime cannot carry, with ``ANIRA_ERROR_NOT_SUPPORTED``). A Channel axis of any
extent is legal.

.. note::
    The accumulator's TFLite export orders the outputs ``[state_out, processed_data]`` while its
    ``serving_default`` signature keys them ``output_0`` / ``output_1``: the tflite row binds by
    position through the signature runner (key order), the litert row names ``processed_data`` as
    ``output_0`` and ``state_out`` as ``output_1``, since LiteRT lists a signature's outputs in the
    file's order. Either way prepare's shape check would refuse a swapped pair (``[1, 2, 64]``
    against ``[1, 2, 2]``).
