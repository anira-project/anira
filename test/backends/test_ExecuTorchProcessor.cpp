// The ExecuTorchProcessor paths test_ExecuTorchModelFunction.cpp does not take:
// a program handed over as bytes rather than a path, and a program that cannot
// be loaded at all. Driven directly, without a Context — the way
// test_BackendBase.cpp drives BackendBase.

#ifdef USE_EXECUTORCH

#include <anira/InferenceConfig.h>
#include <anira/backends/ExecuTorchProcessor.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend_test_support.h"
#include "gtest/gtest.h"

namespace {

constexpr size_t k_size = 64;

using anira_test::filled_buffers;
using anira_test::read_model_file;

std::string multifunction_model_path() {
    return std::string(SIMPLEGAIN_MODEL_PATH) + "/simple_gain_network_multifunction.pte";
}

}  // namespace

// A .pte supplied as bytes loads through the BufferDataLoader branch and runs
// the same named method as the same program supplied as a path. "gain2"
// multiplies by two, so the output identifies which program ran.
TEST(ExecuTorchProcessor, BinaryModelDataLoadsFromMemory) {
    const std::vector<char> bytes = read_model_file(multifunction_model_path());
    ASSERT_FALSE(bytes.empty()) << "fixture missing: " << multifunction_model_path();

    anira::InferenceConfig config({anira::ModelData(const_cast<char*>(bytes.data()),
                                                    bytes.size(),
                                                    anira::InferenceBackend::EXECUTORCH,
                                                    "gain2",
                                                    /*is_binary=*/true)},
                                  {anira::TensorShape({{1, 1, static_cast<int64_t>(k_size)}},
                                                      {{1, 1, static_cast<int64_t>(k_size)}})},
                                  5.0F,
                                  /*warm_up=*/1,
                                  /*session_exclusive_processor=*/true);
    ASSERT_TRUE(config.is_model_binary(anira::InferenceBackend::EXECUTORCH));

    anira::ExecuTorchProcessor processor(config);
    processor.prepare();

    std::vector<anira::BufferF> input = filled_buffers({k_size}, 0.25F);
    std::vector<anira::BufferF> output = filled_buffers({k_size}, 0.F);

    processor.process(input, output, nullptr);

    for (size_t i = 0; i < k_size; ++i) {
        EXPECT_FLOAT_EQ(output[0].get_sample(0, i), 0.5F) << "sample " << i;
    }
}

// The contract create_session() rolls back on.
TEST(ExecuTorchProcessor, UnloadableModelThrowsRuntimeError) {
    anira::InferenceConfig config(
        {anira::ModelData("this/model/does/not/exist.pte", anira::InferenceBackend::EXECUTORCH)},
        {anira::TensorShape({{1, 1, static_cast<int64_t>(k_size)}},
                            {{1, 1, static_cast<int64_t>(k_size)}})},
        5.0F,
        /*warm_up=*/0,
        /*session_exclusive_processor=*/true);

    EXPECT_THROW({ const anira::ExecuTorchProcessor processor(config); }, std::runtime_error);
}

// A method name the program does not carry must fail the same way rather than
// silently falling back to forward().
TEST(ExecuTorchProcessor, UnknownModelFunctionThrowsRuntimeError) {
    anira::InferenceConfig config({anira::ModelData(multifunction_model_path(),
                                                    anira::InferenceBackend::EXECUTORCH,
                                                    "no_such_method")},
                                  {anira::TensorShape({{1, 1, static_cast<int64_t>(k_size)}},
                                                      {{1, 1, static_cast<int64_t>(k_size)}})},
                                  5.0F,
                                  /*warm_up=*/0,
                                  /*session_exclusive_processor=*/true);

    EXPECT_THROW({ const anira::ExecuTorchProcessor processor(config); }, std::runtime_error);
}

#endif  // USE_EXECUTORCH
