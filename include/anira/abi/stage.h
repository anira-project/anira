/*
 * anira/abi/stage.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_STAGE_H
#define ANIRA_ABI_STAGE_H

/**
 * @file stage.h
 * @brief The ring accessors, the stage context and the stage descriptor (section 7).
 *
 * A stage is a descriptor, not a class: anira_stage_desc names up to four phase callbacks
 * (pre_process, post_process, before_inference, after_inference), a prepare and a release
 * function, one user_data slot and a name, and is handed once to anira_pipeline_add_stage,
 * which copies it into a refcounted carrier. The callbacks are the host's; nothing about a
 * stage is a vtable in the library. Every phase callback receives an anira_stage_ctx that anira
 * fills on its own stack for the duration of the call: the phase, the engine and provider of
 * the plan the chunk was submitted under, the tensor counts of the model's input and output
 * lists, and the arrays of that phase (rings and model tensors; an array that does not belong
 * to the phase is NULL). Indices into the arrays are tensor indices: the position in the model
 * configuration's input or output list, the order the engine binds. The tensor descriptors are
 * re-pointed on every call, so nothing about them may be cached across calls. The rings stay
 * inside anira: a stage moves elements through the ten anira_ring accessors, which state the
 * dtype they believe the ring holds and move nothing when it is another one (nothing converts,
 * in either direction). The filled slots of a phase run in chain order, in all four phases. The
 * default body of a phase (anira_stage_default_pre_process, anira_stage_default_post_process)
 * runs once per chunk and only when no stage of the chain fills that slot; a stage that fills
 * it owns the phase and calls the default itself when it wants it. When several stages fill one
 * phase they see the same chunk one after the other, so the second finds the tensors the first
 * filled and the rings the first moved: exactly one stage of the chain should move a slot's
 * ring, itself or through the default, and the others work on the tensors. anira counts: after
 * the pre_process phase every Streamed input ring must have given up exactly the slot's hop per
 * channel, after the post_process phase every Streamed output ring must have gained exactly its
 * hop. A shortfall is repaired, so the stream stays aligned (the missing elements are discarded
 * from an input ring, an output ring is topped up with zeros); an excess cannot be undone.
 * Either way anira_handler_rt_error reads ANIRA_ERROR_CONFIG and one latched record names the
 * stage, the tensor, the expected and the moved count. A phase callback that returns anything
 * but ANIRA_OK fails its chunk: later stages of that phase do not run, the status goes into
 * anira_handler_rt_error with one latched record naming the stage, and the chunk delivers zeros
 * at its stream position (a failed pre_process is not submitted to an engine; a failed
 * before_inference or after_inference skips the rest and zeroes the model outputs; after a
 * failed post_process anira tops every Streamed output ring up to the chunk's hop with zeros,
 * because a ring cannot take a push back). In this pre-release every stage is a host stage
 * (ANIRA_DOMAIN_HOST on both sides), every model tensor is host memory of ANIRA_DTYPE_F32, the
 * contract is Hard, so ticket is ANIRA_TICKET_INVALID and pre_process and post_process run on
 * the driver thread, and variant is 0.
 */

#include <stddef.h>
#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/enums.h>
#include <anira/abi/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief The element type the ring stores: the ring dtype the host declared for the slot on the
 * Hard contract (anira_contract_hard_set_ring_dtype), ANIRA_DTYPE_F32 when nothing was
 * declared. It is never inferred from the tensor spec's dtype, which may differ when a
 * stage of the chain converts; every data accessor states the dtype it believes it is
 * reading or writing.
 * @param ring A ring of the stage context, or NULL.
 * @return The ring's dtype; 0, which is no dtype, for a NULL ring.
 * @par Thread contract
 * [driver-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_dtype ANIRA_CALL anira_ring_dtype(const anira_ring* ring)
                                                  ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The channel count of the ring: the extent of the Channel axis of its Streamed tensor
 * spec, 1 without one.
 * @param ring A ring of the stage context, or NULL.
 * @return The number of channels; 0 for a NULL ring.
 * @par Thread contract
 * [driver-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_ring_num_channels(const anira_ring* ring)
                                                      ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The unread elements of one channel.
 * @param ring A ring of the stage context, or NULL.
 * @param channel The channel, below anira_ring_num_channels.
 * @return The count; 0 for a NULL ring, and 0 with ANIRA_ERROR_INVALID_ARGUMENT recorded in
 *         anira_handler_rt_error for a channel out of range.
 * @par Thread contract
 * [driver-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_ring_available(const anira_ring* ring,
                                                 uint32_t channel) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The consumed elements of one channel the ring still retains as history: what
 * anira_ring_peek_past_block can address. 0 right after prepare or reset, growing as
 * elements are popped, bounded by the ring's capacity.
 * @param ring A ring of the stage context, or NULL.
 * @param channel The channel, below anira_ring_num_channels.
 * @return The count; 0 for a NULL ring, and 0 with ANIRA_ERROR_INVALID_ARGUMENT recorded in
 *         anira_handler_rt_error for a channel out of range.
 * @par Thread contract
 * [driver-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_ring_available_past(const anira_ring* ring,
                                                      uint32_t channel) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Pops count elements of one channel into out, oldest first. Exactly count elements are
 * written: those beyond the available ones are value-initialised (zero).
 * @param ring An input ring of the stage context, or NULL.
 * @param channel The channel, below anira_ring_num_channels.
 * @param out Room for count elements of dtype, packed.
 * @param dtype The dtype of the memory at out; it must be anira_ring_dtype(ring).
 * @param count Elements to pop.
 * @return count; 0 with nothing popped and nothing written for a NULL ring, a NULL out with a
 *         count above 0 (ANIRA_ERROR_INVALID_ARGUMENT recorded), a channel out of range
 *         (ANIRA_ERROR_INVALID_ARGUMENT recorded) or a dtype that is not the ring's
 *         (ANIRA_ERROR_CONFIG recorded in anira_handler_rt_error with one latched record;
 *         nothing converts).
 * @par Thread contract
 * [driver-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_ring_pop_block(anira_ring* ring,
                                                 uint32_t channel,
                                                 void* out,
                                                 anira_dtype dtype,
                                                 size_t count) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Copies the count most recently consumed elements of one channel into out without
 * popping, oldest first, so that out[count - 1] is the element popped last: the
 * receptive field of a model whose window is longer than its hop. The range is not
 * checked against anira_ring_available_past: history the ring never held reads as zero
 * after prepare or reset.
 * @param ring An input ring of the stage context, or NULL.
 * @param channel The channel, below anira_ring_num_channels.
 * @param out Room for count elements of dtype, packed.
 * @param dtype The dtype of the memory at out; it must be anira_ring_dtype(ring).
 * @param count Elements of history to copy.
 * @return count; 0 with nothing written under the refusals of anira_ring_pop_block.
 * @par Thread contract
 * [driver-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_ring_peek_past_block(const anira_ring* ring,
                                                       uint32_t channel,
                                                       void* out,
                                                       anira_dtype dtype,
                                                       size_t count) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Pushes count elements into one channel. A full ring overwrites its oldest elements;
 * anira sizes an output ring so that one hop per submitted chunk always fits.
 * @param ring An output ring of the stage context, or NULL.
 * @param channel The channel, below anira_ring_num_channels.
 * @param in count elements of dtype, packed.
 * @param dtype The dtype of the memory at in; it must be anira_ring_dtype(ring).
 * @param count Elements to push.
 * @return count; 0 with nothing pushed for a NULL ring, a NULL in with a count above 0
 *         (ANIRA_ERROR_INVALID_ARGUMENT recorded), a channel out of range
 *         (ANIRA_ERROR_INVALID_ARGUMENT recorded) or a dtype that is not the ring's
 *         (ANIRA_ERROR_CONFIG recorded in anira_handler_rt_error with one latched record;
 *         nothing converts).
 * @par Thread contract
 * [driver-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_ring_push_block(anira_ring* ring,
                                                  uint32_t channel,
                                                  const void* in,
                                                  anira_dtype dtype,
                                                  size_t count) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Pushes count copies of one element into one channel.
 * @param ring An output ring of the stage context, or NULL.
 * @param channel The channel, below anira_ring_num_channels.
 * @param value One element of dtype.
 * @param dtype The dtype of the element at value; it must be anira_ring_dtype(ring).
 * @param count Copies to push.
 * @return count; 0 with nothing pushed under the refusals of anira_ring_push_block (a NULL
 *         value is the NULL in).
 * @par Thread contract
 * [driver-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_ring_push_fill(anira_ring* ring,
                                                 uint32_t channel,
                                                 const void* value,
                                                 anira_dtype dtype,
                                                 size_t count) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Drops up to count unread elements of one channel; they become history like popped
 * ones. No element crosses the call, so it takes no dtype and has no dtype refusal.
 * @param ring An input ring of the stage context, or NULL.
 * @param channel The channel, below anira_ring_num_channels.
 * @param count Unread elements to drop.
 * @return The elements dropped, at most the available ones; 0 for a NULL ring, and 0 with
 *         ANIRA_ERROR_INVALID_ARGUMENT recorded in anira_handler_rt_error for a channel out of
 *         range.
 * @par Thread contract
 * [driver-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_ring_discard(anira_ring* ring,
                                               uint32_t channel,
                                               size_t count) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Pops num_batches overlapping windows of one channel: window b lands at out + offset +
 * b * (num_new + num_old) and holds the num_old most recently consumed elements followed
 * by num_new freshly popped ones, the batched sliding-window layout a model with an
 * input shape [num_batches, ..., num_new + num_old] expects. It pops num_batches *
 * num_new elements in all; with num_batches 1 and offset 0 it is
 * anira_ring_peek_past_block followed by anira_ring_pop_block.
 * @param ring An input ring of the stage context, or NULL.
 * @param channel The channel, below anira_ring_num_channels.
 * @param out The base of the destination, in elements of dtype: room for offset + num_batches *
 *        (num_new + num_old) elements.
 * @param dtype The dtype of the memory at out; it must be anira_ring_dtype(ring).
 * @param num_new Freshly popped elements per window.
 * @param num_old Elements of history ahead of them per window.
 * @param offset Elements to skip at out before the first window.
 * @param num_batches The number of windows.
 * @return The elements written, num_batches * (num_new + num_old); 0 with nothing popped and
 *         nothing written under the refusals of anira_ring_pop_block.
 * @par Thread contract
 * [driver-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_ring_pop_windows(anira_ring* ring,
                                                   uint32_t channel,
                                                   void* out,
                                                   anira_dtype dtype,
                                                   size_t num_new,
                                                   size_t num_old,
                                                   size_t offset,
                                                   uint32_t num_batches) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief What a phase callback of a stage sees. Tier 1: 64 bytes, frozen, identical on every
 * target; no struct_size. anira fills one on its own stack per call, so the record and
 * the arrays it points at are valid until the callback returns and must not be kept.
 * Every scalar but phase has the same value in all four phases of a chunk. An array that
 * does not belong to the phase is NULL: pre_process sees input_rings and model_inputs,
 * before_inference model_inputs, after_inference model_outputs, post_process
 * model_outputs and output_rings. The arrays are indexed by tensor index, the position
 * in the model configuration's input or output list (the order the engine binds), and
 * have num_inputs or num_outputs entries. The tensor descriptors are re-pointed on every
 * call (an engine may swap the memory behind a tensor between two inferences), so a
 * stage reads the data pointer through anira_tensor_data on every call and caches
 * nothing of a descriptor.
 */
typedef struct anira_stage_ctx {
    uint32_t phase;  /**< anira_stage_phase: the phase this call runs in. */
    /**
     * anira_engine of the plan the chunk was submitted under; ANIRA_ENGINE_NONE for a custom
     * engine.
     */
    uint32_t engine;
    uint32_t provider;  /**< anira_provider of that plan. */
    uint32_t variant;  /**< The variant of that plan; 0 in this pre-release. */
    /**
     * The tensors of the model's input list, State tensors included: the length of input_rings
     * and model_inputs. A stage indexes by slot, the tensor's position in the model config's
     * input list: the number every entry of anira_handler and every row of the plan report
     * uses.
     */
    uint32_t num_inputs;
    /**
     * The tensors of the model's output list, State tensors included: the length of
     * model_outputs and output_rings, indexed by slot, the tensor's position in the model
     * config's output list.
     */
    uint32_t num_outputs;
    /**
     * The anira_ticket of the submitting job under an Async contract; ANIRA_TICKET_INVALID
     * under a Hard contract.
     */
    uint32_t ticket;
    uint32_t reserved;  /**< Zero. */
    /**
     * ANIRA_PHASE_PRE_PROCESS only, else NULL: one entry per input tensor, the ring of a
     * Streamed tensor and NULL for a tensor of another role.
     */
    ANIRA_PTR(anira_ring* const, input_rings);
    /**
     * ANIRA_PHASE_PRE_PROCESS and ANIRA_PHASE_BEFORE_INFERENCE, else NULL: the model's input
     * tensors in the spec's shape, host memory owned by anira, for the stage to write. In
     * pre_process every Static input is already materialised. In before_inference every State
     * input is already fed from the session's state buffer: a stage may read or alter it, and
     * what it leaves is what the engine gets.
     */
    ANIRA_PTR(anira_tensor, model_inputs);
    /**
     * ANIRA_PHASE_AFTER_INFERENCE and ANIRA_PHASE_POST_PROCESS, else NULL: the model's output
     * tensors in the spec's shape, host memory owned by anira. A chunk that completed without
     * an inference (dropped, failed) reads as zeros. In after_inference a State output is not
     * yet captured: what the last stage leaves in it is what the next inference is fed.
     */
    ANIRA_PTR(anira_tensor, model_outputs);
    /**
     * ANIRA_PHASE_POST_PROCESS only, else NULL: one entry per output tensor, the ring of a
     * Streamed tensor and NULL for a tensor of another role.
     */
    ANIRA_PTR(anira_ring* const, output_rings);
} anira_stage_ctx;

/**
 * @brief A phase callback of a stage. pre_process and post_process run on the thread that
 * drives the Hard entries (the driver thread, or the thread inside a _wait twin);
 * before_inference and after_inference run on an inference thread, around the engine
 * call. The body is real-time on both: no allocation, no lock, no system call, and of
 * anira only the [callback-safe] entries. Return ANIRA_OK, or any other status to fail
 * the chunk: later stages of that phase do not run, the status goes into
 * anira_handler_rt_error with one latched record naming the stage, and the chunk
 * delivers zeros at its stream position (the file comment has the rule per phase).
 * @param ctx The context of this call; valid until the callback returns.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [driver-thread | inference-thread] ANIRA_NONBLOCKING
 */
typedef anira_status (ANIRA_CALL* anira_stage_fn)(const anira_stage_ctx* ctx,
                                                  void* user_data) ANIRA_NONBLOCKING;

/**
 * @brief The prepare function of a stage: called by anira_handler_prepare on its caller's
 * thread after the plan report is built, once per prepare, in chain order. It may
 * allocate. Of anira it may call the [callback-safe] entries and the handler's getters
 * (the handler counts as prepared while it runs), never prepare, destroy or a Hard
 * entry. A status other than ANIRA_OK fails anira_handler_prepare with that status, the
 * message naming the stage, and leaves the handler unprepared.
 * @param handler The handler being prepared.
 * @param report The plan report of this prepare; valid while the handler stays prepared.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [main-thread]
 */
typedef anira_status (ANIRA_CALL* anira_stage_prepare_fn)(anira_handler* handler,
                                                          const anira_plan_report* report,
                                                          void* user_data);

/**
 * @brief The release function of a stage: called exactly once, when the last carrier of the
 * descriptor dies, on the thread of that destroy (anira_pipeline_destroy or
 * anira_handler_destroy, whichever comes last). No callback of the stage runs
 * afterwards.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [main-thread]
 */
typedef void (ANIRA_CALL* anira_stage_release_fn)(void* user_data);

/**
 * @brief A stage, handed once to anira_pipeline_add_stage and copied within struct_size into a
 * refcounted carrier that the pipeline and every handler created from it share. Tier 2:
 * struct_size first, user_data third (its offset never moves), growth at the tail. A
 * NULL phase slot means the stage does not take part in that phase.
 */
typedef struct anira_stage_desc {
    uint32_t struct_size;  /**< sizeof(anira_stage_desc) of the caller's header. */
    uint32_t abi_version;  /**< ANIRA_ABI_VERSION the caller compiled against. */
    void* user_data;  /**< Handed to every callback as it is; never read by anira. */
    /**
     * The stage's name, copied by anira_pipeline_add_stage (the caller's string may die when
     * the call returns). NULL or an empty string reads "stage#<index>", the position in the
     * chain. Every latched record about the stage, a prepare failure it causes and the consumer
     * column of its anira_plan_ext rows carry it. Names need not be unique.
     */
    const char* name;
    /**
     * anira_domain the stage reads; ANIRA_DOMAIN_HOST in this pre-release.
     */
    uint32_t domain_in;
    /**
     * anira_domain the stage writes; ANIRA_DOMAIN_HOST in this pre-release.
     */
    uint32_t domain_out;
    /**
     * The extensions the stage reads at prepare, as "<host>:<kind>" strings (hosts:
     * tensor_spec, model, model_config, context, contract), copied; NULL with a count of 0 for
     * none. They join the consumed-or-fail walk of anira_handler_create and
     * anira_handler_prepare, and each consumed slot is an anira_plan_ext row whose consumer is
     * the stage's name.
     */
    const char* const* consumed_kinds;
    uint32_t num_consumed_kinds;  /**< The length of consumed_kinds. */
    uint32_t reserved;  /**< Zero. */
    /**
     * ANIRA_PHASE_PRE_PROCESS: forms the model inputs from the input rings. A stage that fills
     * it owns the phase and calls anira_stage_default_pre_process itself when it wants the
     * default fill. NULL: not taking part.
     */
    anira_stage_fn pre_process;
    /**
     * ANIRA_PHASE_POST_PROCESS: pushes the model outputs into the output rings;
     * anira_stage_default_post_process is the default push. NULL: not taking part.
     */
    anira_stage_fn post_process;
    /**
     * ANIRA_PHASE_BEFORE_INFERENCE, on the inference thread ahead of the engine call. NULL: not
     * taking part.
     */
    anira_stage_fn before_inference;
    /**
     * ANIRA_PHASE_AFTER_INFERENCE, on the inference thread behind the engine call. NULL: not
     * taking part.
     */
    anira_stage_fn after_inference;
    anira_stage_prepare_fn prepare;  /**< Called at every anira_handler_prepare; NULL for none. */
    anira_stage_release_fn release;  /**< Called once, when the last carrier dies; NULL for none. */
} anira_stage_desc;
/**
 * @brief A host stage without a name and without a callback.
 */
#define ANIRA_STAGE_DESC_INIT ANIRA_INIT(anira_stage_desc, sizeof(anira_stage_desc), ANIRA_ABI_VERSION, NULL, NULL, ANIRA_DOMAIN_HOST, ANIRA_DOMAIN_HOST, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL)

/**
 * @brief The default pre_process, for a stage that wants it ("call super"): for every Streamed
 * input it pops one hop per channel from the ring into the model tensor, which it reads
 * as packed row-major memory: channel c of a tensor of n elements per channel starts at
 * element c * n. Where n is the hop, that is the plain pop. Where n is above the hop,
 * the receptive field of a sliding-window model, the n minus hop elements at the head of
 * the channel are the ring's history (anira_ring_peek_past_block) and the hop is popped
 * behind them. It never touches a tensor of another role. anira runs it itself, once per
 * chunk, when no stage of the chain fills pre_process.
 * @param ctx The context the stage's pre_process received.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL ctx or a ctx of another phase than
 *         ANIRA_PHASE_PRE_PROCESS; ANIRA_ERROR_CONFIG when the ring dtype of a Streamed input
 *         is not its tensor's dtype, with that tensor untouched and the others filled (nothing
 *         converts: a stage that declared another ring dtype pops that ring itself). The status
 *         is returned, never recorded by the default itself: a stage that returns it fails the
 *         chunk with it.
 * @par Thread contract
 * [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_stage_default_pre_process(const anira_stage_ctx* ctx)
                                                                  ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The default post_process, for a stage that wants it ("call super"): for every Streamed
 * output it pushes one hop per channel from the model tensor, read as packed row-major
 * memory, into the ring: channel c from element c * hop. It never touches a tensor of
 * another role. anira runs it itself, once per chunk, when no stage of the chain fills
 * post_process.
 * @param ctx The context the stage's post_process received.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL ctx or a ctx of another phase than
 *         ANIRA_PHASE_POST_PROCESS; ANIRA_ERROR_CONFIG when the ring dtype of a Streamed output
 *         is not its tensor's dtype, with that ring untouched and the others pushed. The status
 *         is returned, never recorded by the default itself: a stage that returns it fails the
 *         chunk with it.
 * @par Thread contract
 * [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_stage_default_post_process(const anira_stage_ctx* ctx)
                                                                   ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_STAGE_H */
