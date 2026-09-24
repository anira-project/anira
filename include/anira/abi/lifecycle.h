/*
 * anira/abi/lifecycle.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_LIFECYCLE_H
#define ANIRA_ABI_LIFECYCLE_H

/**
 * @file lifecycle.h
 * @brief The two records the stage and the engine descriptors share: what init and prepare receive.
 *
 * The lifecycle of a stage (anira/abi/stage.h) and of a custom engine (anira/abi/engine.h) uses
 * the same words on both sides, and the two callbacks both descriptors carry, init and prepare,
 * receive the same records, declared here once. anira_init_info is what the init slot receives,
 * once per carrier (a stage's registration, an engine object), at the first
 * anira_handler_prepare that reaches it: the facts of the core in effect then (the log level,
 * the size of the inference-thread pool) and the context of that handler. A refused init fails
 * that prepare and leaves the carrier uninitialised, so the next prepare calls init again;
 * release fires whether or not init ever ran. anira_prepare_info is what the prepare slot
 * receives, once per handler (and, for an engine, per loaded model the handler runs on, beside
 * the loaded pointer): the handler, its plan report, its entry count, a template of the model
 * end of every slot, the canonical names and the flags of the prepare, ANIRA_PREPARE_EXCLUSIVE
 * among them. Both records are Tier 2, struct_size first, no implicit padding, filled by anira
 * and valid for the duration of the call: the callee copies what it keeps and never keeps the
 * pointers (the handler and the report of a prepare record stay valid while the handler stays
 * prepared).
 */

#include <stddef.h>
#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief What the init slot of a stage or an engine receives (anira_stage_init_fn,
 * anira_engine_init_fn): the facts of the core in effect when the carrier is first used,
 * and the context it is used under. Tier 2, struct_size first, no implicit padding (LP64
 * 24 bytes, ILP32 20); anira fills it, so it carries no struct_id. Valid for the
 * duration of the call.
 */
typedef struct anira_init_info {
    uint32_t struct_size;  /**< sizeof(anira_init_info) of the library. */
    /**
     * anira_log_level: the level in effect in this copy of anira (the core reconciles one per
     * process, the most verbose of its contexts').
     */
    uint32_t log_level;
    /**
     * The size of the inference-thread pool this copy of anira runs, what
     * anira_num_inference_threads answers; 0 when the context brought its own threads
     * (anira_inference_thread_create) or on a target without threads.
     */
    uint32_t num_threads;
    uint32_t reserved;  /**< 0. */
    /**
     * The context of the handler whose prepare reached the carrier first: its capabilities
     * (anira_context_capabilities) are legal to query; valid until the callback returns, never
     * kept.
     */
    const anira_context* context;
} anira_init_info;
/**
 * @brief No context: what a test fills by hand.
 */
#define ANIRA_INIT_INFO_INIT ANIRA_INIT(anira_init_info, sizeof(anira_init_info), 0u, 0u, 0u, NULL)

/**
 * @brief What the prepare slot of a stage or an engine receives (anira_stage_prepare_fn,
 * anira_engine_prepare_fn): the handler being prepared and what its chunks will look
 * like. Tier 2, struct_size first, no implicit padding (LP64 72 bytes, ILP32 48); anira
 * fills it, so it carries no struct_id. Valid for the duration of the call: the callee
 * copies what it keeps (the handler pointer and the report stay valid while the handler
 * stays prepared, the arrays do not).
 */
typedef struct anira_prepare_info {
    uint32_t struct_size;  /**< sizeof(anira_prepare_info) of the library. */
    /**
     * anira_handler_num_entries of this prepare: the chunks that can be in flight at once, one
     * scratch slot per entry (anira_stage_ctx.entry and anira_engine_ctx.entry, the chunk's
     * position in the handler's inference queue).
     */
    uint32_t num_entries;
    /**
     * The handler being prepared: its getters and the two Static entries are legal, never
     * prepare, destroy or a Hard entry.
     */
    anira_handler* handler;
    /**
     * This prepare's plan report; valid while the handler stays prepared.
     */
    const anira_plan_report* report;
    /**
     * num_inputs templates in slot order, State tensors included: the MODEL end of every input,
     * the spec's dtype and shape at the pinned window, all-zero strides, the slot's host
     * domain, no memory (never dereferenced). What anira_stage_input_tensor will fill, minus
     * the data; an engine's process sees the ENGINE side of the same slots, the templates of
     * its load record.
     */
    const anira_tensor* inputs;
    const anira_tensor* outputs;  /**< num_outputs templates, likewise. */
    const char* const* input_names;  /**< num_inputs canonical names, in slot order. */
    const char* const* output_names;  /**< num_outputs canonical names. */
    uint32_t num_inputs;  /**< The tensors of the model's input list, State tensors included. */
    uint32_t num_outputs;  /**< The tensors of the model's output list, State tensors included. */
    /**
     * The flags of this prepare: ANIRA_PREPARE_EXCLUSIVE when the handler's inferences run one
     * at a time and in order (a model declared ANIRA_MODEL_STATEFUL or with a declared State
     * pair), with a reset at every stream start; 0 otherwise. No other bit is defined in this
     * pre-release.
     */
    uint32_t flags;
    uint32_t reserved;  /**< 0. */
} anira_prepare_info;
/**
 * @brief No handler: what a test fills by hand.
 */
#define ANIRA_PREPARE_INFO_INIT ANIRA_INIT(anira_prepare_info, sizeof(anira_prepare_info), 0u, NULL, NULL, NULL, NULL, NULL, NULL, 0u, 0u, 0u, 0u)

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_LIFECYCLE_H */
