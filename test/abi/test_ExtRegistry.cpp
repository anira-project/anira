#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "capi/ext_registry.h"
#include "capi/handles.h"

using anira::capi::EntryPayload;
using anira::capi::ExtBag;
using anira::capi::ExtSlot;

namespace {

anira_ext_entry make_entry(const char* name) {
    anira_ext_entry entry = ANIRA_EXT_ENTRY_INIT;
    entry.name = name;
    return entry;
}

}  // namespace

TEST(AbiExtRegistry, RegistryRowsAndKinds) {
    const std::vector<anira::capi::ExtRow>& rows = anira::capi::ext_rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_STREQ(rows[0].m_kind, "entry");
    EXPECT_EQ(rows[0].m_version, 1u);
    EXPECT_EQ(rows[0].m_struct_size, sizeof(anira_ext_entry));
    EXPECT_NE(rows[0].m_from_json, nullptr) << "entry has a JSON form";
    ASSERT_EQ(anira::capi::ext_kinds().size(), 1u);
    EXPECT_STREQ(anira::capi::ext_kinds()[0], "entry");
}

TEST(AbiExtRegistry, SetDeepCopiesAKnownKind) {
    std::string name = "encode";
    const anira_ext_entry entry = make_entry(name.c_str());
    ExtBag bag;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(bag.set(&entry.header, &err), ANIRA_OK) << err.message;
    name = "changed";  // the caller's strings may die or change after the call
    const ExtSlot* slot = bag.find("entry");
    ASSERT_NE(slot, nullptr);
    EXPECT_TRUE(slot->known());
    const auto* payload = bag.payload<EntryPayload>("entry");
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->m_name, "encode");
    EXPECT_STREQ(payload->m_hdr.name, "encode") << "the header points into the payload";
    EXPECT_STREQ(payload->m_hdr.header.kind, "entry");
    EXPECT_EQ(slot->to_json(), R"({"name":"encode"})");
    // A second set of the same kind replaces the first.
    const anira_ext_entry other = make_entry("decode");
    ASSERT_EQ(bag.set(&other.header, &err), ANIRA_OK);
    EXPECT_EQ(bag.slots().size(), 1u);
    EXPECT_EQ(bag.payload<EntryPayload>("entry")->m_name, "decode");
}

TEST(AbiExtRegistry, CopyingABagDeepCopiesItsPayloads) {
    ExtBag bag;
    anira_error err = ANIRA_ERROR_INIT;
    const anira_ext_entry entry = make_entry("forward");
    ASSERT_EQ(bag.set(&entry.header, &err), ANIRA_OK);
    const ExtBag copy = bag;  // NOLINT(performance-unnecessary-copy-initialization): the copy is
                              // the test
    const auto* a = bag.payload<EntryPayload>("entry");
    const auto* b = copy.payload<EntryPayload>("entry");
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_NE(a, b);
    EXPECT_EQ(b->m_name, "forward");
    EXPECT_STREQ(b->m_hdr.name, "forward");
}

TEST(AbiExtRegistry, RejectionsAndUnknownKinds) {
    ExtBag bag;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(bag.set(nullptr, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    const anira_ext_header short_header{.struct_size = 4, .version = 1, .kind = "entry"};
    EXPECT_EQ(bag.set(&short_header, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "struct_size"), nullptr);
    anira_ext_entry v2 = make_entry("x");
    v2.header.version = 2;
    EXPECT_EQ(bag.set(&v2.header, &err), ANIRA_ERROR_EXTENSION_VERSION);
    EXPECT_NE(std::strstr(err.message, "version 2"), nullptr);
    EXPECT_TRUE(bag.empty()) << "nothing stored on rejection";
    // An unknown kind is stored, not rejected: prepare names it.
    const anira_ext_header unknown{.struct_size = sizeof(anira_ext_header),
                                   .version = 1,
                                   .kind = "quant"};
    EXPECT_EQ(bag.set(&unknown, &err), ANIRA_OK);
    const ExtSlot* slot = bag.find("quant");
    ASSERT_NE(slot, nullptr);
    EXPECT_FALSE(slot->known());
    EXPECT_EQ(slot->raw_header().size(), sizeof(anira_ext_header));
}

TEST(AbiExtRegistry, JsonTwinParsesKnownKindsAndKeepsUnknownText) {
    ExtBag bag;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(bag.set_json("entry", R"({"name": "decode"})", &err), ANIRA_OK) << err.message;
    EXPECT_EQ(bag.payload<EntryPayload>("entry")->m_name, "decode");
    ASSERT_EQ(bag.set_json("entry", R"({"version": 1, "name": "encode"})", &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(bag.payload<EntryPayload>("entry")->m_name, "encode");
    EXPECT_EQ(bag.set_json("entry", R"({"version": 2, "name": "x"})", &err),
              ANIRA_ERROR_EXTENSION_VERSION);
    EXPECT_EQ(bag.set_json("entry", R"({"name": 3})", &err), ANIRA_ERROR_JSON);
    EXPECT_NE(std::strstr(err.message, "entry.name"), nullptr);
    EXPECT_EQ(bag.set_json("entry", "{not json", &err), ANIRA_ERROR_JSON);
    EXPECT_EQ(bag.set_json("entry", R"({"version": "one", "name": "x"})", &err), ANIRA_ERROR_JSON);
    EXPECT_EQ(bag.set_json("", R"({})", &err), ANIRA_ERROR_INVALID_ARGUMENT);
    ASSERT_EQ(bag.set_json("quant", R"({"scale": 0.5})", &err), ANIRA_OK);
    const ExtSlot* slot = bag.find("quant");
    ASSERT_NE(slot, nullptr);
    EXPECT_FALSE(slot->known());
    EXPECT_EQ(slot->to_json(), R"({"scale": 0.5})") << "the text is kept verbatim";
}

TEST(AbiExtRegistry, ConsumedOrFailWalkNamesTheOffender) {
    anira_model_config model;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira::capi::ext_check_consumed(model, nullptr, nullptr, nullptr, 0, &err), ANIRA_OK)
        << "nothing set";

    anira_tensor_spec spec;
    spec.m_name = "audio_in";
    const anira_ext_header unknown{.struct_size = sizeof(anira_ext_header),
                                   .version = 1,
                                   .kind = "quant"};
    ASSERT_EQ(spec.m_ext.set(&unknown, &err), ANIRA_OK);
    model.m_inputs.push_back(spec);
    EXPECT_EQ(anira::capi::ext_check_consumed(model, nullptr, nullptr, nullptr, 0, &err),
              ANIRA_ERROR_EXTENSION_UNKNOWN);
    EXPECT_STREQ(err.message, "extension 'quant' on tensor 'audio_in' is not known to this build");
    model.m_inputs.clear();

    // A known kind on a host no consumer reads it from.
    const anira_ext_entry entry = make_entry("forward");
    ASSERT_EQ(spec.m_ext.set(&entry.header, &err), ANIRA_OK);
    spec.m_ext = ExtBag();
    ASSERT_EQ(spec.m_ext.set(&entry.header, &err), ANIRA_OK);
    model.m_inputs.push_back(spec);
    EXPECT_EQ(anira::capi::ext_check_consumed(model, nullptr, nullptr, nullptr, 0, &err),
              ANIRA_ERROR_EXTENSION_UNCONSUMED);
    EXPECT_STREQ(
        err.message,
        "extension 'entry' on tensor 'audio_in' is not consumed by any stage in this build");
    model.m_inputs.clear();

    anira::capi::ModelEntry libtorch;
    libtorch.m_engine = ANIRA_ENGINE_LIBTORCH;
    libtorch.m_path = "model.pt";
    ASSERT_EQ(libtorch.m_ext.set(&entry.header, &err), ANIRA_OK);
    model.m_models.push_back(libtorch);
    const anira_backend_id only_onnx{.struct_size = sizeof(anira_backend_id),
                                     .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                     .provider = ANIRA_PROVIDER_DEFAULT,
                                     .engine_id = nullptr};
    EXPECT_EQ(anira::capi::ext_check_consumed(model, nullptr, nullptr, &only_onnx, 1, &err),
              ANIRA_OK)
        << "a LibTorch entry that is not a candidate is not walked";
    const anira_backend_id only_libtorch{.struct_size = sizeof(anira_backend_id),
                                         .engine = ANIRA_ENGINE_LIBTORCH,
                                         .provider = ANIRA_PROVIDER_DEFAULT,
                                         .engine_id = nullptr};
#ifdef USE_LIBTORCH
    EXPECT_EQ(anira::capi::ext_check_consumed(model, nullptr, nullptr, nullptr, 0, &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(anira::capi::ext_check_consumed(model, nullptr, nullptr, &only_libtorch, 1, &err),
              ANIRA_OK)
        << err.message;
#else
    EXPECT_EQ(anira::capi::ext_check_consumed(model, nullptr, nullptr, &only_libtorch, 1, &err),
              ANIRA_ERROR_EXTENSION_UNCONSUMED);
    EXPECT_STREQ(err.message,
                 "extension 'entry' on model 0 is not consumed by any stage in this build");
#endif

    // The same kind on an engine that has no adapter reading it fails by name.
    anira::capi::ModelEntry onnx;
    onnx.m_engine = ANIRA_ENGINE_ONNXRUNTIME;
    onnx.m_path = "model.onnx";
    ASSERT_EQ(onnx.m_ext.set(&entry.header, &err), ANIRA_OK);
    model.m_models.push_back(onnx);
    EXPECT_EQ(anira::capi::ext_check_consumed(model, nullptr, nullptr, &only_onnx, 1, &err),
              ANIRA_ERROR_EXTENSION_UNCONSUMED);
    EXPECT_STREQ(err.message,
                 "extension 'entry' on model 1 is not consumed by any stage in this build");

    // Context and contract hosts are walked when given.
    model.m_models.clear();
    anira_context_config context;
    ASSERT_EQ(context.m_ext.set(&unknown, &err), ANIRA_OK);
    EXPECT_EQ(anira::capi::ext_check_consumed(model, &context, nullptr, nullptr, 0, &err),
              ANIRA_ERROR_EXTENSION_UNKNOWN);
    EXPECT_NE(std::strstr(err.message, "on context"), nullptr);
    anira_contract contract;
    ASSERT_EQ(contract.m_ext.set(&entry.header, &err), ANIRA_OK);
    EXPECT_EQ(anira::capi::ext_check_consumed(model, nullptr, &contract, nullptr, 0, &err),
              ANIRA_ERROR_EXTENSION_UNCONSUMED);
    EXPECT_NE(std::strstr(err.message, "on contract"), nullptr);
}
