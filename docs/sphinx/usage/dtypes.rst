.. _usage-dtypes:

Dtypes
======

A model tensor is of one of the ten dtypes anira names and keeps it end to end: its ring (unless
the contract names a ring dtype), the chunk's buffer, its Static store, its State pair and every
engine's descriptors carry it. Nothing in anira converts.

.. list-table::
   :header-rows: 1

   * - Constant
     - Model-file word
     - Element
   * - ``ANIRA_DTYPE_F32``
     - ``"float32"`` (the default)
     - IEEE 754 binary32
   * - ``ANIRA_DTYPE_F64``
     - ``"float64"``
     - IEEE 754 binary64
   * - ``ANIRA_DTYPE_F16``
     - ``"float16"``
     - binary16, as its 16-bit words (C and C++ have no type for it)
   * - ``ANIRA_DTYPE_BF16``
     - ``"bfloat16"``
     - the upper 16 bits of a binary32, as its 16-bit words
   * - ``ANIRA_DTYPE_I8``, ``U8``
     - ``"int8"``, ``"uint8"``
     - 8-bit integers
   * - ``ANIRA_DTYPE_I16``, ``I32``, ``I64``
     - ``"int16"``, ``"int32"``, ``"int64"``
     - signed integers
   * - ``ANIRA_DTYPE_BOOL8``
     - ``"bool"``
     - one byte, 0 or 1

A dtype anira does not name (several lanes, an unsigned integer wider than 8 bits) is
``ANIRA_ERROR_NOT_SUPPORTED`` at ``anira_handler_create`` and has no word (``to_json`` refuses it);
``anira_tensor_spec_create`` refuses a dtype of 0. Every built-in engine carries all ten: the bundled
TypedGainNetwork fixtures (int16, int8, float16) run exactly on each, and the ONNX dtype matrix runs
all ten through ONNX Runtime and ONNX Runtime Web.

.. _usage-ring-dtype:

The ring dtype
--------------

The **ring** of a Streamed tensor, its host end, holds the spec's dtype unless the contract names
another (``ring_dtypes``, :ref:`usage-contract-options`), so an int16 model takes int16 host tensors
with no setting. A host of another sample type names its type and adds a processor that converts:
in ``pre_process`` for an input, in ``post_process`` for an output, with the host's own scaling. A
ring dtype that differs from the spec's without a processor that fills the phase moving that ring
is ``ANIRA_ERROR_CONFIG`` at prepare (anira checks the phase, not that it converts), and so is a
name that is no Streamed tensor.

A host tensor whose dtype is not its slot's ring dtype is refused per call with
``ANIRA_ERROR_CONFIG``, recorded in ``rt_error()``, nothing delivered or staged, and the log line
names both remedies:

.. code-block:: text

    anira_handler_process: the tensor of input slot 0 'audio_in' is float32, the slot's ring
    dtype is int16; nothing converts: pass a tensor of the ring dtype, or name float32 in the
    contract's ring_dtypes and add a processor that converts in pre_process

The plan report names both dtypes of every slot (``anira_plan_slot.dtype`` and ``ring_dtype``, 0
for a slot without a ring), so a host can compare its sample type with the ring's before its first
call.

Converting processors
---------------------

- an int16 PCM host on a float32 model: ``ring_dtypes = {{"audio_in", ANIRA_DTYPE_I16}}`` and a
  ``pre_process`` that writes ``x / 32768.0f``;
- a float32 host on a float16 model: float32 for both rings, a ``pre_process`` that writes the
  model's float16 and a ``post_process`` that reads it back;
- a float32 host on an int8 model: the same with the model's own convention, which the host knows and
  anira does not (``clamp(round(x * 127), -128, 127)`` in, ``q / 127`` out, for a model trained so);
- a host whose type is the model's: no setting, no processor.

The int16 and float16 processors are written out in :doc:`../custom_preprocessing/examples`.

Quantised models
----------------

A model whose quantisation lives inside the graph and whose I/O is float (ONNX QDQ, a TFLite model
with float I/O around quantised operators, PT2E-quantised internals) runs as any float model. A
tensor with quantisation parameters at the boundary (an int8 tensor meaning
``scale * (q - zero_point)``, a fully integer TFLite export) is refused at load with
``ANIRA_ERROR_NOT_SUPPORTED`` and never handed on as plain integers: TFLite and LiteRT report the
parameters, LibTorch and ExecuTorch a quantised scalar type. ONNX carries no quantisation at a
graph's boundary, so an ONNX model's int8 I/O is plain integers. Until a later pre-release adds the
parameters to the spec, a host that has them converts as in the int8 case.

On the web an audio node streams float32: a model of another dtype behind one needs float32 ring
dtypes and a converting processor (:doc:`../web-api/audio_nodes`).
