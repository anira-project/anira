Latency
=======

Overview
--------

Latency is a critical factor in real-time audio processing applications. The anira framework implements a sophisticated latency calculation system to ensure proper synchronization between audio processing and neural network inference.

How Latency is Calculated
-------------------------

The latency calculation in anira is performed by the :cpp:class:`anira::SessionElement` class. The total system latency consists of several components:

Buffer Adaptation Latency
~~~~~~~~~~~~~~~~~~~~~~~~~~

Accounts for mismatches between the host buffer size and the model's expected input/output sizes. When the host provides audio in different chunk sizes than what the model expects, additional buffering is required to accumulate or split the data appropriately.

.. note::
    When the host buffer size is a fractional (floating-point) value, this indicates that the host and model process buffers at non-integer ratios. The latency calculation in anira accounts for this by assuming the worst-case scenario: a sample is pushed to the :cpp:class:`anira::InferenceHandler` only when the host buffer accumulates a full sample. For example, if the host buffer size is 0.25f, the :cpp:class:`anira::InferenceHandler` receives one sample every four host buffer cycles, and latency is calculated as if the sample is delivered during the fourth cycle. If your system always sends the sample at the first host buffer cycle, a lower latency is possible—in such cases, consider configuring :cpp:class:`anira::InferenceHandler` with a custom latency value.


Inference-Caused Latency
~~~~~~~~~~~~~~~~~~~~~~~~~

Represents the delay introduced by the neural network inference process itself. This includes:

* Maximum inference time of the model
* Number of parallel inferences that can be processed
* Time spent waiting for inference completion
* Relationship between host buffer timing and inference completion

Wait Time Calculation
~~~~~~~~~~~~~~~~~~~~~

When using controlled blocking (``blocking_ratio > 0``), the system may wait for inference to complete before continuing. This wait time is calculated based on the host buffer duration and the configured blocking ratio.

Internal Model Latency
~~~~~~~~~~~~~~~~~~~~~~~

Additional latency that may be inherent to the model itself, such as look-ahead requirements or internal buffering.

Latency Synchronization
~~~~~~~~~~~~~~~~~~~~~~~

When multiple outputs are present, the system synchronizes latencies across all outputs to ensure coherent processing. This is achieved by calculating a latency ratio and applying it uniformly across all output channels.

The latency vector returned by :cpp:func:`anira::InferenceHandler::get_latency_vector` is index-aligned with the output tensor list. Non-streamable outputs (``postprocess_output_size == 0``) carry no stream latency and always report ``0``.

Adaptive Buffer Handling
~~~~~~~~~~~~~~~~~~~~~~~~~

For hosts that support variable buffer sizes (``allow_smaller_buffers``), the system performs additional calculations to handle worst-case scenarios across different buffer sizes, ensuring stable latency regardless of the actual buffer size used. The reported latency and the number of inference slots are the maximum over every buffer size up to the stated one. That maximum is not found by trying each size: the inference-caused latency of a buffer size is a sawtooth (it jumps whenever the inference time crosses a whole number of buffers), so only its peaks, the points where a blocking wait stops covering the remainder, and the sizes where the number of inferences per buffer changes are evaluated. The result is identical to an exhaustive walk, at a cost that grows with the square root of the inference time in samples rather than with the buffer size.

These calculations count buffer sizes in samples of the *reference stream* selected by the :cpp:struct:`anira::HostConfig` — a streamable input for effects and analysers, the streamable output for a generator model with no streamable input (see the usage guide, section 4.1).

Output Behavior
---------------

The final latency value represents the total delay (in samples) between when input data enters the system and when the processed output data becomes available.

.. important::
    Before the first valid output is produced, the :cpp:func:`anira::InferenceHandler::process` and :cpp:func:`anira::InferenceHandler::pop_data` methods will return zeroed data. This ensures real-time audio processing without introducing unexpected delays or artifacts in the output signal.