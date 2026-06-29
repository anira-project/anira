# TODOs

## Architecture

- [ ] Change model_path function, so that the model with the same backend can be switched on runtime.

## Documentation

- [x] Update Dokumentation for JSON example
- [x] Update Dokumentation for supporting all models now
- [ ] Add more examples to the documentation

## Testing

- [ ] Sanitizer tests in ci
- [ ] InferenceHandler tests with buffersizes that are not a multiple of the preprocess input size

## Bugs

- [ ] When declaring the universal shape in HybridNNConfig.h first, tests fail on asahi linux system (tflite gets universal tensor shapes)
- [ ] Calling reset in inference handler with blocking mechanism causes freeze

## Packaging

- [ ] Trigger `ldconfig` in the .deb package
- [ ] Artifacts should not be .zip as symlinks are not supported
- [ ] Build the .deb package in CI
- [ ] Properly declare the dependencies on the backend libraries in the .deb package