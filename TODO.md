# TODOs

## Architecture

- [ ] Model and context config should be one json file
- [ ] Change the non-audio data to use ringbuffers instead of raw atomics
- [ ] Add interface for direct push and pull from m_inference_queue
- [ ] Add ability to turn off the latency compensation
- [ ] Add option to select different model functions
- [ ] Add option for dynamic input size
- [ ] Manual inference triggering
- [ ] Fix TFLite benchmark error

## How to handle parameters

- [ ] ThreadSafeStruct needs none audio parameters
- [ ] In and Out RingBuffer also for each non-audio parameters
- [ ] In PrePostProcessor if parameters updates on ringbuffers are available, then update the parameters in the inference queue
- [ ] After inference the parameters should be updated in the ringbuffers
- [ ] And also updated in the atomic variables
- [ ] Get free inference queue
- [ ] Get done inference queue
- [ ] Angabe wie viele Werte pro Vector brauch man in Tensor

## Testing

- [ ] Run the examples as tests in CI

## Packaging

- [ ] Trigger `ldconfig` in the .deb package
- [ ] Artifacts should not be .zip as symlinks are not supported
- [ ] Add qemu docker emulation for aarch64 and armv7l linux in CI
- [ ] Build the .deb package in CI