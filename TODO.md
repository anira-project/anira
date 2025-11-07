# TODOs

## Architecture

- [ ] Make moodycamel producer structs with session id and delete them when the session removed
- [ ] Model and context config should be also available as json file
- [ ] InferenceConfig check
- [ ] RTSan check in CI
- [ ] Change model_path function
- [ ] Make processBlockFixture could be more versatile with new shapes
- [ ] Fix TFLite benchmark error

## Extras

- [ ] Put RAVE model on tu servers, for more stable download

## Documentation

- [ ] Update Dokumentation for JSON example
- [ ] Update Dokumentation for supporting all models now
- [ ] Add more examples to the documentation

## Testing

- [ ] More sanitizer tests
- [ ] Run the examples as tests in CI
- [ ] InferenceHandler tests with buffersizes that are not a multiple of the preprocess input size

## Bugs

- [ ] Fix noise burst at the start of the plugin with the model 6 - this is really annoying and hurts ears!
- [ ] When declaring the universal shape in HybridNNConfig.h first, tests fail on asahi linux system (tflite gets universal tensor shapes)
- [ ] Calling reset in inference handler with blocking mechanism causes freeze

## Packaging

- [ ] Trigger `ldconfig` in the .deb package
- [ ] Artifacts should not be .zip as symlinks are not supported
- [ ] Add qemu docker emulation for aarch64 and armv7l linux in CI
- [ ] Build the .deb package in CI