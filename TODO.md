# TODOs

## Architecture

- [ ] Think about internal latency
- [ ] Make m_inference_counter.fetch_add(1) dependent on number of samples
- [ ] Make processBlockFixture more versatile with new shapes
- [ ] Model and context config should be one json file
- [ ] Add option to select different model functions
- [ ] Manual inference triggering
- [ ] Fix TFLite benchmark error

## Testing

- [ ] Run the examples as tests in CI

## Packaging

- [ ] Trigger `ldconfig` in the .deb package
- [ ] Artifacts should not be .zip as symlinks are not supported
- [ ] Add qemu docker emulation for aarch64 and armv7l linux in CI
- [ ] Build the .deb package in CI