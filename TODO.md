# TODOs

## Architecture

- [ ] Model and context config should be one json file
- [ ] Change the non-audio data to use ringbuffers instead of raw atomics
- [ ] Add interface for direct push and pull from m_inference_queue
- [ ] Add ability to turn off the latency compensation

## Testing

- [ ] Run the examples as tests in CI

## Packaging

- [ ] Trigger `ldconfig` in the .deb package
- [ ] Artifacts should not be .zip as symlinks are not supported
- [ ] Add qemu docker emulation for aarch64 and armv7l linux in CI
- [ ] Build the .deb package in CI