.. _usage-processing:

Real-time processing
====================

In the host's audio callback every value crosses as an :cpp:class:`anira::Tensor`, a plain
descriptor of your memory: the audio blocks as planar tensors over the host's channel pointers,
built once at prepare, the Static values whole. The entries are ``noexcept`` and
``ANIRA_NONBLOCKING``: they never wait, allocate or throw.

.. code-block:: cpp

    // At prepare: the pointer array the tensor refers to, and the tensor over it
    std::array<float*, 2> channels{};
    anira::Tensor audio = anira::Tensor::from_host_planar<float>(
        std::span<float* const>(channels), std::array<int64_t, 2>{2, max_block});

    // The callback: store the pointers and the sample count, then process in place
    void process_block(float* const* audio_data, size_t num_samples) noexcept {
        channels = {audio_data[0], audio_data[1]};
        audio.shape[1] = static_cast<int64_t>(num_samples);
        size_t delivered = 0;
        const anira_status st = inference_handler.process(audio, 0, audio, 0, &delivered);
        // ANIRA_OK: num_samples delivered; ANIRA_MISSED: the miss policy filled the block
    }

.. _usage-slots:

Slots
-----

A tensor is addressed by its **slot**: its position in the model config's ``inputs`` or
``outputs`` list, counted from 0, whatever its role, State tensors included. Every entry, the
latency vector, the plan report and a processor's accessors use that one number. The two lists are
unrelated, so the single forms name a slot per side: a model with ``data`` at input 1 and
``processed_data`` at output 0 calls ``process(in, 1, out, 0)``. The role decides which entries take
a slot: a Streamed slot the block entries, a Static slot the two Static entries, a State slot none
(anira feeds it), a Buffer slot the Async contract's. A host that does not want to hard-code a slot
resolves it by name at setup, and reads its role from the plan report (``anira_plan_slot.role``).

.. _usage-host-tensors:

Host tensors
------------

A block is logically ``[channels, samples]`` (``[1, samples]`` for a spec without a Channel axis),
its sample count is ``shape[1]``, and its dtype must be the slot's ring dtype (the model's unless
the contract names another, :ref:`usage-dtypes`). The memory is host memory (``ANIRA_DOMAIN_HOST``
or ``ANIRA_DOMAIN_HOST_PINNED``), described one of three ways, and the two sides of a call may
differ:

- **Planar**, one pointer per channel (what JUCE and CLAP hand out): ``Tensor::from_host_planar``
  (``anira_tensor_init_host_planar``). The tensor borrows the pointer *array*, read at each call, so
  a host that rewrites the pointers in it per callback needs nothing else; a host whose array itself
  moves (``juce::AudioBuffer::getArrayOfWritePointers``) fills the tensor in the callback, which is
  legal there (a field fill, ``ANIRA_NONBLOCKING``):

  .. code-block:: cpp

      const int64_t shape[2] = {buffer.getNumChannels(), buffer.getNumSamples()};
      anira_tensor block;
      anira_tensor_init_host_planar(&block, buffer.getArrayOfWritePointers(),
                                    (uint32_t)buffer.getNumChannels(), ANIRA_DTYPE_F32, 2, shape);
      anira_handler_process(handler, &block, 0, &block, 0, NULL);

- **One block read by strides**, in elements: contiguous channels are ``{samples, 1}``, an
  interleaved block (miniaudio, most device callbacks) ``{1, channels}``, assigned on the record:

  .. code-block:: c

      const int64_t shape[2] = {2, (int64_t)frame_count};
      anira_tensor in, out;
      anira_tensor_init_host(&in, (void*)frames_in, ANIRA_DTYPE_F32, 2, shape);
      in.flags |= ANIRA_TENSOR_READ_ONLY;
      anira_tensor_init_host(&out, frames_out, ANIRA_DTYPE_F32, 2, shape);
      in.strides[0] = out.strides[0] = 1;
      in.strides[1] = out.strides[1] = 2;   /* the channel count */
      anira_handler_process(handler, &in, 0, &out, 0, &delivered);

- **One packed block**, all-zero strides (read as ``{samples, 1}``).

In place, for a model whose input and output blocks have the same shape, is the same tensor on both
sides; separate input and output are two tensors, the input's
over ``const float* const*`` with ``ANIRA_TENSOR_READ_ONLY``. anira writes the memory an output
names, never a descriptor, so the tensors are built once and reused. Every tensor of a call is
validated before anything moves, in the order rank, domain, flags, shape, dtype, memory, strides,
then the count against the block, the input before the output. A refusal writes no ring:

- ``ANIRA_ERROR_INVALID_ARGUMENT``: a rank other than 2 (which catches the zeroed record a refused
  factory leaves), a domain other than those two, ``READ_ONLY`` on an output, ``shape[0]`` other
  than the slot's channel count, a negative ``shape[1]``, and with ``shape[1]`` above 0: NULL memory,
  a plane count other than ``shape[0]``, a NULL plane, a negative stride, a stride of 0 on an axis longer than 1 (unless all are 0), a channel off
  the element size's alignment, or a ``shape[1]`` above one host block on the slot's stream (``ceil``
  of the block: a call carries one declared block at most);
- ``ANIRA_ERROR_NOT_SUPPORTED``: a flag bit this library does not know;
- ``ANIRA_ERROR_CONFIG``: a dtype other than the ring's (nothing converts; the log line names both
  remedies, :ref:`usage-dtypes`).

The memory is borrowed until the call returns; ``release``, ``manager_ctx`` and ``acquire`` are not
read. Under ``ANIRA_MISS_BYPASS``, input and output memory
that overlap under two different descriptions are undefined on a miss; the same tensor on both sides
is left where it is.

.. _usage-entries:

The entries
-----------

.. code-block:: cpp

    inference_handler.process(input, 0, output, 0, &delivered);   // one block in, one out
    inference_handler.push_data(input, 0);                       // decoupled: push ...
    inference_handler.pop_data(output, 0, &received);            // ... and pop, at most shape[1]
    inference_handler.process(inputs, outputs, delivered.data()); // one tensor per slot of each side

The block entries are ``[driver-thread]`` and ``ANIRA_NONBLOCKING``.

- ``process(in, in_slot, out, out_slot, &delivered)`` moves one block in and one out, of any size up
  to ``host_block_max``. The output tensor's ``shape[1]`` is the request.
- ``push_data`` and ``pop_data`` split the halves for hosts that produce and consume at different
  rates; call both from the same thread. ``push_data`` also collects finished inferences while the
  receive rings have room, so push-only use works for a model whose results leave through Static
  outputs; a Streamed output must still be popped (unread, its samples stay, anira stops collecting
  into the full ring and logs "Output stream not consumed").
- **The forms over every slot** (``process(inputs, outputs, delivered)``, ``push_data(inputs)``,
  ``pop_data(outputs, delivered)``; C ``_multi``) take exactly one tensor per slot of the model
  config's lists (``num_inputs`` / ``num_outputs`` are the lists' lengths): a Streamed slot's block,
  a Static slot's whole tensor, and an empty tensor (rank 1 or more, an extent of 0; its memory is
  not read, so NULL is legal) for a slot the call leaves out and at every State position, where
  anything else is ``ANIRA_ERROR_INVALID_ARGUMENT``. They are the sequence they abbreviate:
  ``set_static_input`` for every non-empty Static input, the stream over the Streamed slots,
  ``get_static_output`` for every non-empty Static output; every element is validated first.
- ``delivered`` is a nullable pure out parameter, written on every return: zeroed, then on
  ``ANIRA_OK`` the ``shape[1]`` of each Streamed output, and 0 on ``ANIRA_MISSED`` and on a failure.
  A carried Static output reports its element count on ``ANIRA_OK`` and on ``ANIRA_MISSED`` alike;
  a State slot always 0.

.. _usage-status:

Statuses and the real-time error
--------------------------------

Every entry returns an ``anira_status``: ``ANIRA_OK`` when the block was delivered in full;
``ANIRA_MISSED`` when its inference had not completed, a success whose buffers hold what the miss
policy chose (:ref:`usage-miss-policy`); or a failure, which pushes nothing and leaves every
delivered count 0. A refusal carries no
``anira_error``: the entry records the status in ``rt_error()`` (``anira_handler_rt_error``,
readable from any thread and any callback; cleared by ``prepare`` and ``reset``) and logs once
through the real-time queue (:ref:`logging-realtime`); a miss is not recorded.
``get_available_samples(slot, channel, &count)`` collects completed inferences and reports what
waits in an output ring (right after prepare, the latency).

.. _usage-static:

Static tensors
--------------

A Static tensor travels whole, in the spec's shape and dtype, any dtype, a Channel axis of any
extent included; one ``Tensor::from_host`` over the spec's extents builds it (all-zero strides are
packed row-major; other strides are read and written as given). The example's model has a
Streamed audio tensor at slot 0 of each side, a Static ``control`` input ``[1, 1, 4]`` at input slot
1 and a Static ``confidence`` output ``[1, 1, 1]`` at output slot 1:

.. code-block:: json

    {
      "inputs": [
        { "name": "audio_in", "axes": [["batch", 1], ["channel", 1], ["time", 2048]] },
        { "name": "control", "role": "static", "axes": [["batch", 1], ["channel", 1], ["any", 4]] }
      ],
      "outputs": [
        { "name": "audio_out", "axes": [["batch", 1], ["channel", 1], ["time", 2048]] },
        { "name": "confidence", "role": "static", "axes": [["batch", 1], ["channel", 1], ["any", 1]] }
      ]
    }

.. code-block:: cpp

    std::array<float, 4> control{};
    float confidence = 0.0F;
    const anira::Tensor control_tensor =
        anira::Tensor::from_host(control.data(), ANIRA_DTYPE_F32, std::array<int64_t, 3>{1, 1, 4});
    const anira::Tensor confidence_tensor =
        anira::Tensor::from_host(&confidence, ANIRA_DTYPE_F32, std::array<int64_t, 3>{1, 1, 1});

    void process_block(size_t num_samples, float gain, float threshold) noexcept {
        control = {gain, threshold, 0.0F, 0.0F};
        inference_handler.set_static_input(1, control_tensor);   // every inference from now on sees it
        audio.shape[1] = static_cast<int64_t>(num_samples);
        inference_handler.process(audio, 0, audio, 0);
        inference_handler.get_static_output(1, confidence_tensor); // the latest collected inference's
    }

- ``set_static_input(slot, tensor)`` stores an input in the handler's own store (sized and zeroed at
  create, untouched by prepare and reset); every inference submitted afterwards sees it whole (copied
  under a latch, materialised when the chunk is formed), so a value set between two calls never
  tears.
- ``get_static_output(slot, tensor)`` reads the value the latest collected inference produced, all
  zeros before the first; a chunk that completed as zeros (dropped, or failed) captures nothing.
- Both are ``[driver-thread]`` and ``ANIRA_NONBLOCKING``, legal from create on (a value set before
  prepare survives it) and from a processor's ``prepare``. A value set from another thread (a UI) is
  stored in an atomic and set from the audio thread, as the JUCE example does with its gain.
- "Fits" is a check, never a clamp: another rank or extent is ``ANIRA_ERROR_INVALID_ARGUMENT``,
  another dtype ``ANIRA_ERROR_CONFIG``, ``ANIRA_TENSOR_PLANAR`` or an unknown flag
  ``ANIRA_ERROR_NOT_SUPPORTED``; a slot out of range or not Static, NULL or misaligned memory, a
  negative stride and ``READ_ONLY`` on the output are ``ANIRA_ERROR_INVALID_ARGUMENT``. Every refusal
  is recorded in ``rt_error()``.
- A single block form never names a Static slot (``ANIRA_ERROR_INVALID_ARGUMENT``). On a missed block
  a carried Static output holds the stored value under every policy; under ``ANIRA_MISS_CALLBACK``
  it is filled with the stored value first, and what the function writes reaches the caller's memory
  only, never the store. A Static output reports a latency of 0.

.. _usage-one-sided:

Generators and analysers
------------------------

Streams may sit on one side only, and the anchor resolves to that side.

- **A generator** has no Streamed input: each ``process`` or ``pop_data`` adds the requested sample
  count on the anchored output to the demand, and one inference is submitted per output hop
  demanded, capturing the Static values current at that call; ``push_data`` never submits.
  ``get_latency`` counts from the first pull after prepare or reset.

  .. code-block:: cpp

      parameter = frequency;
      inference_handler.set_static_input(0, parameter_tensor);   // captured at submission
      output.shape[1] = static_cast<int64_t>(num_samples);
      inference_handler.pop_data(output, 0);                      // submits on demand

- **An analyser** has no Streamed output: push the stream and read the latest result
  (``get_static_output``), which updates whenever results are collected (by any ``process``,
  ``push_data``, ``pop_data`` or ``get_available_samples``) and reads 0 before the first.

.. _usage-wait:

The ``_wait`` twins
-------------------

``process_wait`` and ``pop_data_wait`` (each in both forms) wait for the block's inference:
``timeout_ms >= 0`` explicitly, ``ANIRA_WAIT_CONTRACT`` for the contract's ``wait_ratio`` times the
block duration (the call's block on the process forms, ``host_block_max`` on the pop forms), or
``ANIRA_WAIT_FOREVER``. In C++ the timeout comes before the optional ``delivered``; the C twins
(``anira_handler_process_wait``, ``_process_multi_wait``, ``_pop_data_wait``,
``_pop_data_multi_wait``) append it as the last parameter. They wait on the completion semaphore
when ``wait_ratio`` is above 0, else by polling every millisecond.

They are ``[any-thread, blocking]``, not ``ANIRA_NONBLOCKING``: a wait on the real-time thread is
the host's decision, trading
real-time safety for a smaller latency (on WebAssembly every wait spins). A block not completed at
the timeout is ``ANIRA_MISSED``. Without an inference thread inside its loop (the pool, or a host
thread in ``anira_inference_thread_run_loop``; a host pumping ``execute`` does not count) a twin
does what the nonblocking entry does and returns ``ANIRA_ERROR_INVALID_STATE`` with the counts that
delivered. A push never waits.

.. _usage-reset:

Resetting the stream
--------------------

``reset()`` (``anira_handler_reset``) re-anchors the stream on a transport jump or a restart: it
clears the rings, re-seeds the latency's zeros, and invalidates every inference dispatched so far
(their results are discarded and their structures reclaimed when they complete). It is wait-free,
``ANIRA_NONBLOCKING`` for every configuration, a stateful one included, and clears ``rt_error()``.
Call it from the thread that drives the handler, never concurrently with ``prepare`` or destruction.
At the first chunk of the new stream, keyed on its generation:

- the declared state is zeroed (on the inference thread, so an inference in flight across the reset
  cannot seed the new stream);
- an exclusive handler's engine state is reset through the engine's ``reset``, on the inference
  thread right before the first ``process`` of the new stream (TensorFlow Lite's variable tensors, a
  registered engine's per-stream state); a stateless handler keeps nothing and is never reset;
- the processor's state is reset through its ``reset``, on the thread that runs ``pre_process``,
  before that chunk's ``pre_process``.

``reset`` does not wait for in-flight inferences: one may still run, a processor's hooks or a custom
engine included, after it returns. ``prepare`` drains them, for code that needs that guarantee. Until
they finish (at most one inference duration) their structures stay captive; fresh data that exhausts the rest (likely on a stateful
handler, whose pool is small) drops hops under the miss policy, and the stream recovers by itself.
