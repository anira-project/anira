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
 * lists, and an opaque frame. The tensors and the rings are not in the record: a stage asks per
 * slot, the tensor's position in the model configuration's input or output list (the order the
 * engine binds, and the one number every entry of anira_handler uses). anira_stage_input_role
 * and anira_stage_output_role answer what the tensor is (the spec's anira_role);
 * anira_stage_input_ring and anira_stage_output_ring hand out the ring of a Streamed tensor, or
 * NULL when the slot has no ring in this phase, which is an answer and not an error;
 * anira_stage_input_tensor and anira_stage_output_tensor fill a descriptor of the model end of
 * the slot. That descriptor is built on every call over the memory the tensor has right now (an
 * engine may swap it between two inferences), so nothing about it may be cached across calls.
 * Further accessors may be appended in a later version; the record's layout does not change for
 * them. The rings stay inside anira: a stage moves elements through the ten anira_ring
 * accessors, which state the dtype they believe the ring holds and move nothing when it is
 * another one (nothing converts, in either direction). The filled slots of a phase run in chain
 * order, in all four phases. The default body of a phase (anira_stage_default_pre_process,
 * anira_stage_default_post_process) runs once per chunk and only when no stage of the chain
 * fills that slot; a stage that fills it owns the phase and calls the default itself when it
 * wants it. When several stages fill one phase they see the same chunk one after the other, so
 * the second finds the tensors the first filled and the rings the first moved: exactly one
 * stage of the chain should move a slot's ring, itself or through the default, and the others
 * work on the tensors. anira counts: after the pre_process phase every Streamed input ring must
 * have given up exactly the slot's hop per channel, after the post_process phase every Streamed
 * output ring must have gained exactly its hop. A shortfall is repaired, so the stream stays
 * aligned (the missing elements are discarded from an input ring, an output ring is topped up
 * with zeros); an excess cannot be undone. Either way anira_handler_rt_error reads
 * ANIRA_ERROR_CONFIG and one latched record names the stage, the tensor, the expected and the
 * moved count. A phase callback that returns anything but ANIRA_OK fails its chunk: later
 * stages of that phase do not run, the status goes into anira_handler_rt_error with one latched
 * record naming the stage, and the chunk delivers zeros at its stream position (a failed
 * pre_process is not submitted to an engine; a failed before_inference or after_inference skips
 * the rest and zeroes the model outputs; after a failed post_process anira tops every Streamed
 * output ring up to the chunk's hop with zeros, because a ring cannot take a push back). In
 * this pre-release every stage is a host stage (ANIRA_DOMAIN_HOST on both sides), every model
 * tensor is host memory of ANIRA_DTYPE_F32, the contract is Hard, so ticket is
 * ANIRA_TICKET_INVALID and pre_process and post_process run on the driver thread, and variant
 * is 0.
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
 * target; no struct_size. anira fills one on its own stack per call: the record and the
 * frame it names are valid until the callback returns, and a record kept beyond it must
 * not be handed to an accessor. Every scalar but phase has the same value in all four
 * phases of a chunk. The tensors and the rings are not in the record: a stage asks for
 * them per slot through anira_stage_input_role, anira_stage_input_ring,
 * anira_stage_input_tensor and their output twins, handing over the ctx as it received
 * it. What a phase exposes: pre_process the input rings and the model's input tensors,
 * before_inference the input tensors, after_inference the output tensors, post_process
 * the output tensors and the output rings.
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
     * The tensors of the model's input list, State tensors included: the slots the input
     * accessors answer for. A slot is the tensor's position in the model config's input list:
     * the number every entry of anira_handler and every row of the plan report uses.
     */
    uint32_t num_inputs;
    /**
     * The tensors of the model's output list, State tensors included: the slots the output
     * accessors answer for, each the tensor's position in the model config's output list.
     */
    uint32_t num_outputs;
    /**
     * The anira_ticket of the submitting job under an Async contract; ANIRA_TICKET_INVALID
     * under a Hard contract.
     */
    uint32_t ticket;
    uint32_t reserved;  /**< Zero. */
    /**
     * anira's own: what the accessors read. Valid for the duration of the callback. A stage
     * never dereferences it and never writes it.
     */
    ANIRA_PTR(const void, frame);
    ANIRA_PTR(void, reserved_ptr0);  /**< NULL. */
    ANIRA_PTR(void, reserved_ptr1);  /**< NULL. */
    ANIRA_PTR(void, reserved_ptr2);  /**< NULL. */
} anira_stage_ctx;

/**
 * @brief What the tensor of an input slot is: the anira_role of its spec, the same answer in
 * all four phases. It is the first question of a stage about a slot: a Streamed tensor
 * has a ring in pre_process (anira_stage_input_ring) and a model end, a Static and a
 * State tensor have their model end only (anira_stage_input_tensor).
 * @param ctx The context the phase callback received.
 * @param slot The tensor's position in the model config's input list, below num_inputs.
 * @return The role; ANIRA_ROLE_FORCE32, which is no role, for a NULL ctx, a ctx without a
 *         frame, or a slot at or beyond num_inputs. The slot out of range records
 *         ANIRA_ERROR_INVALID_ARGUMENT in anira_handler_rt_error with one latched record naming
 *         the running stage; a NULL ctx and a ctx without a frame name no handler to record
 *         into.
 * @par Thread contract
 * [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_role ANIRA_CALL anira_stage_input_role(const anira_stage_ctx* ctx,
                                                       uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief What the tensor of an output slot is: the anira_role of its spec, the same answer in
 * all four phases. A Streamed tensor has a ring in post_process
 * (anira_stage_output_ring) and a model end, a Static and a State tensor have their
 * model end only (anira_stage_output_tensor).
 * @param ctx The context the phase callback received.
 * @param slot The tensor's position in the model config's output list, below num_outputs.
 * @return The role; ANIRA_ROLE_FORCE32 under the refusals of anira_stage_input_role, against
 *         num_outputs.
 * @par Thread contract
 * [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_role ANIRA_CALL anira_stage_output_role(const anira_stage_ctx* ctx,
                                                        uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The ring of an input slot, for the ten anira_ring accessors: the ring of a Streamed
 * tensor in ANIRA_PHASE_PRE_PROCESS. NULL is the answer to "has this tensor a ring here"
 * and not an error: a tensor of another role has no ring, and no input ring is handed
 * out in another phase than pre_process. Nothing is recorded for it, and every
 * anira_ring accessor takes a NULL ring. The pointer is valid until the callback
 * returns.
 * @param ctx The context the phase callback received.
 * @param slot The tensor's position in the model config's input list, below num_inputs.
 * @return The ring; NULL when the slot has no ring in this phase. NULL too for a NULL ctx, a
 *         ctx without a frame, or a slot at or beyond num_inputs, the last with
 *         ANIRA_ERROR_INVALID_ARGUMENT recorded in anira_handler_rt_error and one latched
 *         record naming the running stage.
 * @par Thread contract
 * [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_ring* ANIRA_CALL anira_stage_input_ring(const anira_stage_ctx* ctx,
                                                        uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The ring of an output slot: the ring of a Streamed tensor in ANIRA_PHASE_POST_PROCESS.
 * NULL, with nothing recorded, for a tensor of another role and in every other phase:
 * the answer to "has this tensor a ring here", not an error. The pointer is valid until
 * the callback returns.
 * @param ctx The context the phase callback received.
 * @param slot The tensor's position in the model config's output list, below num_outputs.
 * @return The ring; NULL when the slot has no ring in this phase, and under the refusals of
 *         anira_stage_input_ring, against num_outputs.
 * @par Thread contract
 * [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_ring* ANIRA_CALL anira_stage_output_ring(const anira_stage_ctx* ctx,
                                                         uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills out with the model end of an input slot: the tensor the engine binds, in the
 * spec's shape, host memory owned by anira, for the stage to write. Available in
 * ANIRA_PHASE_PRE_PROCESS and ANIRA_PHASE_BEFORE_INFERENCE, for a tensor of every role.
 * In pre_process every Static input is already materialised. In before_inference every
 * State input is already fed from the session's state buffer: a stage may read or alter
 * it, and what it leaves is what the engine gets. The descriptor is built by this call
 * over the memory the tensor has right now; an engine may swap that memory between two
 * inferences, so a stage asks again in every callback and keeps neither the descriptor
 * nor its data pointer across calls.
 * @param ctx The context the phase callback received.
 * @param slot The tensor's position in the model config's input list, below num_inputs.
 * @param out The caller's record, filled with a borrowed descriptor (release NULL); an all-zero
 *        record, which reads as dtype 0, whenever the status is not ANIRA_OK.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_STATE in a phase that does not expose the model's
 *         inputs (after_inference, post_process), with nothing recorded: the status is the
 *         report. ANIRA_ERROR_INVALID_ARGUMENT for a NULL ctx, a ctx without a frame, a NULL
 *         out, or a slot at or beyond num_inputs; the NULL out and the slot out of range are
 *         recorded in anira_handler_rt_error too, with one latched record naming the running
 *         stage.
 * @par Thread contract
 * [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_stage_input_tensor(const anira_stage_ctx* ctx,
                                                           uint32_t slot,
                                                           anira_tensor* out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills out with the model end of an output slot: the tensor the engine wrote, in the
 * spec's shape, host memory owned by anira. Available in ANIRA_PHASE_AFTER_INFERENCE and
 * ANIRA_PHASE_POST_PROCESS, for a tensor of every role. A chunk that completed without
 * an inference (dropped, failed) reads as zeros. In after_inference a State output is
 * not yet captured: what the last stage leaves in it is what the next inference is fed.
 * The descriptor is built by this call and is not kept across calls, as for
 * anira_stage_input_tensor.
 * @param ctx The context the phase callback received.
 * @param slot The tensor's position in the model config's output list, below num_outputs.
 * @param out The caller's record, filled with a borrowed descriptor (release NULL); an all-zero
 *        record, which reads as dtype 0, whenever the status is not ANIRA_OK.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_STATE in a phase that does not expose the model's
 *         outputs (pre_process, before_inference), with nothing recorded;
 *         ANIRA_ERROR_INVALID_ARGUMENT under the refusals of anira_stage_input_tensor, against
 *         num_outputs.
 * @par Thread contract
 * [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_stage_output_tensor(const anira_stage_ctx* ctx,
                                                            uint32_t slot,
                                                            anira_tensor* out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

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
