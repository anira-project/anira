# TODOs

## Architecture

- [ ] Model and context config should be also available as json file
- [ ] InferenceConfig should return bool if correct
- [ ] RTSan check in CI
- [ ] Change model_path function
- [ ] Make processBlockFixture could be more versatile with new shapes
- [ ] Fix TFLite benchmark error

## Testing

- [ ] Run the examples as tests in CI

## Packaging

- [ ] Trigger `ldconfig` in the .deb package
- [ ] Artifacts should not be .zip as symlinks are not supported
- [ ] Add qemu docker emulation for aarch64 and armv7l linux in CI
- [ ] Build the .deb package in CI