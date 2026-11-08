About
=====

**anira** runs neural networks inside real-time audio applications. The audio thread hands its
blocks to a handler and gets processed blocks back at a fixed latency; the inferences run on the
inference threads of a process-wide pool, so the audio thread never allocates, locks or waits.

- **Engines:** ONNX Runtime, LibTorch, LiteRT (or the legacy TensorFlow Lite runtime), ExecuTorch,
  any engine you register, and ONNX Runtime Web in the browser. A model file names an export per
  engine, and the handler switches between them at run time.
- **Any model shape:** streamed tensors windowed from host blocks of any size (a receptive field,
  another rate, a generator or an analyser), static parameters, declared recurrent state, and the
  ten dtypes anira names.
- **The latency is computed,** in closed form from the host blocks, the inference time and the
  instance count (:doc:`latency`), and a miss is filled by a policy you choose, keeping the stream
  aligned.
- **Custom pre- and post-processing** in a processor, and engines of your own, in C, C++ or
  TypeScript.
- **Benchmarks** in the same environment as the application: a Google Benchmark fixture in the
  examples (:doc:`benchmarking`).
- **One API in three faces:** a versioned C ABI (``anira/abi/*.h``), the header-only C++20 builders
  of ``anira/anira.hpp`` over it, and the ``@anira-project/anira`` WebAssembly/TypeScript package.

anira also suits other real-time uses with streamed and static data, such as robotics and
computer vision. Its design and an evaluation are in the
`paper <https://doi.org/10.1109/IS262782.2024.10704099>`_.
