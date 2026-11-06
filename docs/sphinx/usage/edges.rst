.. _usage-edges:

Edges, movers and memory
========================

The plan report says per slot how the plan's engine hands the slot's memory over: ``edge_class``
says where the bytes go, ``moved_by`` who moves them. The capability rows say the same of a
provider before any model exists (:ref:`usage-capabilities`), and a contract's edge cost drops the
plans above a limit (:ref:`usage-contract-options`).

.. code-block:: cpp

    for (const anira_plan_slot& slot : report.slots(plan, /*inputs=*/true)) {
        // slot.edge_class: ANIRA_EDGE_ZERO_COPY, _COPY, _UNIFIED_TRANSFER, _DEDICATED_TRANSFER, _UNKNOWN
        // slot.moved_by:   ANIRA_MOVER_NONE, or the bits ANIRA_MOVER_ANIRA | ANIRA_MOVER_ENGINE
    }
    const anira_memory_kind kind = static_cast<anira_memory_kind>(report.plans()[plan].memory_kind);

The edge classes
----------------

Four classes ordered by cost, each its own sort of move, which follow the domains and the physical
memory, never an API's names for them (a layout transform inside a runtime's graph is no edge), and
one class outside the scale:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - ``edge_class``
     - ``moved_by``
     - the move
   * - ``ANIRA_EDGE_ZERO_COPY``
     - ``NONE``
     - Nothing moves: the engine computes on the slot's own memory (bound in place on a provider over
       host memory: the CPU path, XNNPACK, ONNX Runtime's CPU-side execution providers; or a device
       runtime wrapping it on unified memory, which only a registered engine that knows its runtime
       says).
   * - ``ANIRA_EDGE_COPY``
     - ``ANIRA`` or ``ENGINE``
     - A copy within one domain, the CPU's or one device's, with no hand-over between processors: an
       adapter's memcpy (``ANIRA``, the copies anira makes and counts) or the runtime's own API
       (``ENGINE``); in a device's memory a relayout or a device copy.
   * - ``ANIRA_EDGE_UNIFIED_TRANSFER``
     - ``ENGINE``
     - A crossing between the CPU's domain and a device's within one unified physical memory (an
       integrated GPU, Apple silicon, a CPU device such as Mesa's lavapipe): a copy plus a hand-over
       (a submission, a fence, maybe cache maintenance), no bus.
   * - ``ANIRA_EDGE_DEDICATED_TRANSFER``
     - ``ENGINE``
     - A crossing over a bus to or from dedicated memory (a discrete GPU).
   * - ``ANIRA_EDGE_UNKNOWN`` (255)
     - ``ENGINE``
     - A device crossing whose memory kind no probe established (an API missing or failed, an NPU, a
       custom provider, every device provider on the web); the reason says "memory kind not probed".
       It lies above the scale: every limit of the edge cost refuses it, only no limit admits it.

``moved_by`` is a set of bits: both where anira and the runtime each move the bytes (a memory-planned
ExecuTorch output on a device delegate), and the edge then has the worse move's class. The rule is
per direction: an input slot is a device runtime's upload, an output its readback, with the same
class and mover both ways for a built-in provider and each its own reason; a registered engine
states its own per slot (:ref:`engines-edges`).

The memory kind
---------------

A device provider's class follows its device's memory, reported as itself in ``memory_kind``
(``anira_memory_kind``: ``ANIRA_MEMORY_UNIFIED``, ``ANIRA_MEMORY_DEDICATED``, ``ANIRA_MEMORY_UNKNOWN``)
on every plan row and capability row; a provider over host memory is ``UNIFIED``. anira asks the
device's own API, loaded at run time (no link dependency): Metal's ``hasUnifiedMemory`` (Apple), the
Vulkan device type cross-checked with its heaps, the CUDA driver's ``CU_DEVICE_ATTRIBUTE_INTEGRATED``,
HIP's ``hipDeviceAttributeIntegrated``, OpenCL's ``CL_DEVICE_HOST_UNIFIED_MEMORY``, D3D12's
``ARCHITECTURE.UMA``. A browser exposes none.

.. list-table::
   :header-rows: 1

   * - Provider
     - Device asked
   * - CUDA (any engine), ONNX Runtime's TensorRT and NvTensorRTRTX
     - the CUDA device of ONNX Runtime's ``device_id`` option (0 without one, and for another engine)
   * - ONNX Runtime's ROCm and MIGraphX
     - the HIP device of ``device_id``
   * - DirectML
     - the DXGI adapter of ``device_id`` (D3D12)
   * - Core ML, ExecuTorch's MPSBackend
     - the Metal system default device
   * - WebGPU (through Dawn), LiteRT's ``gpu`` accelerator, Vulkan (ExecuTorch's VulkanBackend)
     - every device of the API the runtime picks among (Metal on Apple, D3D12 on Windows, Vulkan
       elsewhere, OpenCL for LiteRT on Android): their kind when all agree, else unknown, naming each
   * - an NPU, any other custom provider
     - none: unknown

The probe runs lazily, the first time a row needs a device provider's kind (a read of either of its
backend's two capability rows, or a prepare with a plan on it), never at context creation and never
on the audio thread. Each API is asked once on a thread of its own, waited for 5 s at most, and the
context keeps the answer until ``anira_context_probe``; a Debug record of ``anira.system`` logs each
answer.

The built-in engines
--------------------

On the CPU path:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Engine
     - Inputs
     - Outputs
   * - ONNX Runtime
     - bound in place (a value over the slot's memory per call)
     - bound in place
   * - LibTorch
     - bound in place (``from_blob``)
     - one copy, the adapter's: a TorchScript method returns tensors of its own
   * - LiteRT
     - bound in place (a tensor buffer over the slot's memory, kept per memory), on an accelerator too
       (the accelerator moves the slot itself: a unified or dedicated transfer by the engine)
     - the same
   * - TFLite
     - bound in place (the slot's memory is the tensor's custom allocation per call)
     - the same
   * - ExecuTorch
     - bound in place where the memory plan leaves the input out (``set_input`` aliases it); else one
       copy, ``set_input``'s (moved by the engine), into a planned one
     - bound in place where the plan leaves the output out (``set_output_data_ptr``); else one copy,
       the adapter's, out of a planned one
   * - ONNX Runtime Web
     - one copy, the engine's: anira's module and onnxruntime-web's are two wasm memories
     - one copy
   * - a registered engine
     - the chunk's buffer as it is; the engine states what it does with it
     - likewise

A chunk's buffers start on the largest boundary any engine of the plans asks for (64 bytes for every
built-in one), so the in-place paths apply to anira's own memory; a descriptor off the boundary takes
the copy. A State pair's buffers are slots like any other. An engine that binds an input in place
may write it: nothing reads a chunk's inputs after the inference but an ``after_inference`` hook.

On a device provider (a LiteRT GPU accelerator, a GPU execution provider of ONNX Runtime) anira copies
nothing either: the slot's host memory is bound in place and the provider's runtime moves it to the
device and back itself, a unified or a dedicated transfer by the engine, or unknown where no probe
answered. That move cannot be avoided while a chunk lives in host memory: a contract that admits no
bus traffic says ``edge_cost(ANIRA_EDGE_UNIFIED_TRANSFER)`` (the host plans and those on
unified-memory devices), one that admits no crossing ``edge_cost(ANIRA_EDGE_COPY)`` (the host plans
alone). Device chunks, which the runtime binds where it computes, come with a later pre-release.
