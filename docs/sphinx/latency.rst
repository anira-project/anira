Latency
=======

Latency Considerations
----------------------

Latency is a critical factor in real-time audio processing applications. The anira framework implements a sophisticated latency calculation system to ensure proper synchronization between audio processing and neural network inference.

Latency Calculation Overview
----------------------------

The latency calculation in anira is performed by the :cpp:class:`anira::SessionElement` class and consists of several components that work together to determine the total system latency:

**1. Buffer Adaptation Latency**
   This component accounts for mismatches between the host buffer size and the model's expected input/output sizes. When the host provides audio in different chunk sizes than what the model expects, additional buffering is required to accumulate or split the data appropriately.

**2. Inference-Caused Latency** 
   This represents the delay introduced by the neural network inference process itself. It considers:
   - The maximum inference time of the model
   - The number of parallel inferences that can be processed
   - The time spent waiting for inference completion
   - The relationship between host buffer timing and inference completion

**3. Wait Time Calculation**
   When using controlled blocking (blocking_ratio > 0), the system may wait for inference to complete before continuing. This wait time is calculated based on the host buffer duration and the configured blocking ratio.

**4. Internal Model Latency**
   Additional latency that may be inherent to the model itself, such as look-ahead requirements or internal buffering.

**5. Latency Synchronization**
   When multiple outputs are present, the system synchronizes latencies across all outputs to ensure coherent processing. This is done by calculating a latency ratio and applying it uniformly across all output channels.

**6. Adaptive Buffer Handling**
   For hosts that support variable buffer sizes (`allow_smaller_buffers`), the system performs additional calculations to handle the worst-case scenarios across different buffer sizes, ensuring stable latency regardless of the actual buffer size used.

The final latency value represents the total delay (in samples) between when input data enters the system and when the processed output data becomes available at the output.
