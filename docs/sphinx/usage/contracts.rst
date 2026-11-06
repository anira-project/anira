.. _usage-contracts:

Contracts
=========

A **contract** names the run. A **Hard** contract is the real-time stream: the host blocks, the
inference budget, the warm-up, what a missed block delivers, the wait ratio. An **Async** contract
is for jobs without a real-time deadline (refused at prepare in this pre-release,
:ref:`usage-pre-release`). Its file form is the contract file (:ref:`usage-json`).

.. code-block:: cpp

    // In code: an anira::Hard aggregate, the library's defaults for everything left out
    anira::Hard hard{
        .host_block_min = 1, .host_block_max = 512,   // blocks of 1..512 samples
        .host_rate = 48000.0,                         // of the anchor, per second
        .on_miss = ANIRA_MISS_BYPASS,                 // budget: measured, warm-up: until stable
    };
    anira::ContractHandle contract(hard);

    // From the contract file next to the model, with the host's blocks patched in
    anira::ContractHandle loaded = anira::ContractHandle::from_file("model.contract.json");
    loaded.hard_host_blocks(1, max_block, sample_rate);

    anira::Hard back = loaded.hard();   // every field read back, to change and mint again

``contract.hard()`` reads a Hard handle back into the aggregate (the budget rounded to the
nanosecond; an Async handle throws ``ANIRA_ERROR_WRONG_CONTRACT``), and ``contract.kind()`` tells the
two kinds apart; a Hard setter on an Async contract throws ``ANIRA_ERROR_WRONG_CONTRACT`` too.
``anira::Contract`` is the ``std::variant`` of ``anira::Hard`` and ``anira::Async``, and the handle
is minted from either.

.. _usage-host-blocks:

Host blocks
-----------

``host_block_min``, ``host_block_max`` and ``host_rate`` count in elements of the model's anchor
tensor (:ref:`usage-instances`): the block range in samples and the rate in samples per second for
an audio anchor. A contract file usually carries none; the host patches them in once it knows its
device (``hard_host_blocks``).

- ``host_block_min == host_block_max`` is a fixed block, at the lowest latency, for a host that
  never changes it.
- ``host_block_min`` below the maximum lets the host deliver any smaller block; the latency anira
  reserves covers every size of the range held from call to call. A plugin declares the range its
  host promises: JUCE's ``prepareToPlay`` names the largest block only
  (``hard_host_blocks(1, samplesPerBlock, sampleRate)``), CLAP's ``activate`` both ends. A host that
  changes the size between calls can still miss a block in rare sequences (a full block right after
  a short one that completed a hop, `#228 <https://github.com/anira-project/anira/issues/228>`_).
- The blocks are doubles, both 0 (unset) or both finite and above 0 with ``min <= max``; anything
  else is ``ANIRA_ERROR_INVALID_ARGUMENT``. Unset blocks are legal on the contract and refused at
  prepare (``ANIRA_ERROR_CONFIG``, "Hard host blocks missing"). ``host_block_max`` is also the most
  one call may carry: a Hard entry refuses a larger block.

.. _usage-flexible-windows:

Flexible windows and the anchor hop
-----------------------------------

The host blocks pin every flexible window (``window_min < window_max``, declared or derived) at
prepare. One inference consumes the same span of the stream on every tensor, so anira chooses one
**anchor hop**, the anchor elements one inference consumes: the anchor, every Streamed tensor with
a time ratio and every flexible window advance that hop times their ratio (1/1 for a flexible
window without one), and a fixed window without a ratio keeps its own hop.

- A time ratio ``num/den`` makes the anchor hop a multiple of ``den`` (reduced), and each window
  bounds it.
- Of the hops that leave every such tensor a whole hop within its window, anira takes the largest at
  or below ``host_block_max`` (one inference per block, or several when the block is larger), else
  the smallest above it (one inference every few blocks).
- A fixed anchor window is the anchor hop whatever the block.
- What no hop satisfies is ``ANIRA_ERROR_CONFIG``, naming the tensor, the hops it needs and the hops
  the tensors before it allow.

A model whose audio input is flexible and whose latent output runs at ``time_ratio(1, 2048)`` gets
an anchor hop of 2048 under a block of 3000 (one or two inferences per call), of 4096 under 4096,
and of 2048 under 512 (one inference every fourth call).

.. _usage-fractional-blocks:

Fractional blocks
-----------------

A model with no stream at the host rate (a control-rate model whose streams move one frame per 2048
audio samples) has an anchor whose block is a fraction of a sample. Anchor on a stream at the host
rate whenever the model has one; the fractional block is for the model that has none, and it is a
fixed block: a plugin, whose host promises only a largest block, anchors on a stream at the host
rate and declares its range there.

.. code-block:: cpp

    // a host block of host_block samples at sample_rate: 512 samples are 0.25 of a frame
    const double block = host_block / 2048.0;
    contract.hard_host_blocks(block, block, sample_rate / 2048.0);

    // per call k = 1, 2, ...: the whole samples accumulated since the last call
    const auto moved = static_cast<int64_t>(std::floor(k * block));
    latent_in.shape[1] = params_out.shape[1] = moved - moved_so_far;   // 0 or 1
    moved_so_far = moved;
    anira_handler_process(handler, &latent_in, 0, &params_out, 0, &delivered);

anira takes the block as the rational it recovers from it (a relative tolerance of 1e-6: 1/3 and
1/4 exactly) and assumes the host moves a sample only once a whole one has accumulated: by its
``k``-th call, ``floor(k * b)`` samples of a stream whose block is ``b``. The latency is computed for
the sample arriving on the last call of its accumulation. ``ceil(b)`` is the most one call may carry,
on every Streamed slot: a call handed more is refused with ``ANIRA_ERROR_INVALID_ARGUMENT`` before
anything moves; a pop within the bound that the stream cannot cover returns ``ANIRA_MISSED``. A
flexible window under a fractional block gets the smallest anchor hop the windows and time ratios
allow: a latent input ``[1, unbounded]`` under 0.25 runs on one frame every fourth call, and an
output at ``time_ratio(2, 1)`` beside it gets a window of 2.

.. _usage-budget-warmup:

Budget and warm-up
------------------

The warm-up pass also measures the inference time; a measured budget is the maximum of its last
four inferences; under an explicit budget the timings are only reported.

.. code-block:: cpp

    using namespace std::chrono_literals;
    anira::Hard measured{.host_block_min = 512, .host_block_max = 512, .host_rate = 48000};
    anira::Hard explicit_budget{.host_block_min = 512, .host_block_max = 512, .host_rate = 48000,
                                .budget = 5ms,                        // a reproducible latency
                                .warmup = ANIRA_WARMUP_FIXED, .warmup_iterations = 8};

- **The budget** (``anira::Hard::budget``, an ``anira::Budget``) is measured by default
  (``anira::Budget{}``, ``Budget::measured()``, ``ANIRA_BUDGET_MEASURED``, a fresh C contract's). A
  ``std::chrono`` duration is an explicit budget (``ANIRA_BUDGET_EXPLICIT``): the worst-case time per
  inference at the pinned window. ``budget.kind()`` / ``budget.value()`` read it back. An inference
  that exceeds the budget produces a dropout.
- **The pass** runs at ``prepare``, once the session is sized (for the explicit budget, or for one
  clock step under a measured one) and the registered engines and the processor are prepared:
  ordinary inferences of every plan, in
  table order, through the handler's session on the inference threads, each timed on the thread
  that runs it (the state's feed and capture, the processor's ``before_inference`` and
  ``after_inference``, the engine call; never ``pre_process``, ``post_process`` or the wait for a
  thread).
- **The warm-up modes:** ``ANIRA_WARMUP_UNTIL_STABLE`` (the default) stops at the first inference
  whose last four times lie within 20 % of their median, or within one clock step when that is more
  (a browser coarsens its clock), at most 32 (``ANIRA_WARMUP_OUTCOME_CAPPED`` then);
  ``ANIRA_WARMUP_FIXED`` runs ``warmup_iterations`` per plan (fewer than four put the cold start into
  the figure: take ``UNTIL_STABLE`` or five or more); ``ANIRA_WARMUP_NONE`` runs none and needs an
  explicit budget. A measured budget with
  ``NONE`` or ``FIXED`` 0 is ``ANIRA_ERROR_CONFIG``.
- **Per plan, the worst case:** under a measured budget every plan row takes its own figure
  (``anira_plan_info.budget_ms``, unrounded, without a margin) and the session runs the slowest
  plan's. The session is then sized again for it, which starts the stream over, so nothing of the
  pass reaches the stream or the declared state; when that changes the entry count, the registered
  engines and the processor are unprepared and prepared again with the final record.
- A measured latency depends on the machine and its load at the prepare; an explicit budget gives a
  reproducible one. A model too slow for the stream is not refused: the session is sized for one
  host block's burst, the stream misses blocks under the miss policy, and the plan's real-time
  verdict says so (:ref:`usage-plan-report`).
- The pass needs an inference thread in its loop: a context without a pool needs a host thread in
  ``anira_inference_thread_run_loop`` (on the web, ``await context.ready()`` first), else
  ``ANIRA_ERROR_INVALID_STATE``, as for an inference of the pass not complete within 10 s (a hang
  guard). An inference of the pass that fails fails the prepare with ``ANIRA_ERROR_ENGINE`` naming
  the plan. A Hard prepare never returns ``ANIRA_ERROR_BUDGET`` (the Async contract's). Apart from
  the pass, every executor of a built-in engine runs one inference over its own scratch when it is
  made (at load, or at an exclusive handler's prepare).

**The model's figure.** The inference time belongs to the model: the core keeps what every pass
measured, under either budget kind, keyed like its pool of loaded models (the engine, the provider
and its options, the model, its entry, its tensors at the pinned window, its instances), except that
a registered engine counts by its id (``anira.v2.custom`` by its engine object), plus the
processor's inference hooks (nothing for a processor without ``before_inference`` /
``after_inference``, else its id, ``anira_processor_desc.id`` / ``anira::Processor(id)``, else its
registration).

- A later prepare of the model, by another handler or the same one at another block size or rate,
  reuses the figure and runs no pass (``ANIRA_WARMUP_OUTCOME_REUSED``): identical plugins report
  identical latency, and a re-prepare is quick. A prepare whose every plan reuses its figure sizes
  the session once and needs no running inference thread. The real-time factor and verdict are
  still the contract's.
- A figure answers a warm-up at least as thorough: ``UNTIL_STABLE`` answers every one, ``FIXED n`` a
  ``FIXED m`` with ``m <= n``; a weaker one is measured again and replaced. Two prepares of one model
  on two threads measure it once; ``ANIRA_WARMUP_NONE`` neither measures nor reads it.
- The figures live as long as the core's state (until ``anira_release_core_if_idle``;
  ``anira_shutdown`` alone keeps them). An idle
  machine's figure can underestimate the time under later load (memory bandwidth, clock speed, SMT
  siblings); a marginal verdict's headroom is the margin for that, not a guarantee. Give a processor
  with inference hooks an id, so that identical plugins share the figure; registered engines of one
  id share it, however many objects the instances create (``anira.v2.custom`` excepted).

.. _usage-miss-policy:

Miss policy
-----------

A block whose inference has not completed when the host pops it is a **miss**: the entry returns
``ANIRA_MISSED`` (a success) with a delivered count of 0, and ``on_miss`` decides what the buffers
hold. The samples it stood in for are discarded when they arrive, never delivered late: a delivered
block is the stream at exactly its latency, and a host that stays ahead of the stream keeps
missing.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - ``on_miss``
     - A missed block, and a dropped hop
   * - ``ANIRA_MISS_BYPASS`` (default)
     - The block the same call pushed into the anchored input (``process`` or ``process_multi``
       carrying the anchored input's slot; a call on another slot, and a pop, deliver zeros). Needs
       an anchored input ring with the output's channel count and ring dtype: an output anchor, other
       channel counts or another ring dtype are ``ANIRA_ERROR_CONFIG`` at prepare. A dropped hop:
       the dropped input hop itself, at its position.
   * - ``ANIRA_MISS_HOLD_LAST``
     - The last block the output delivered (one block-sized buffer per channel, allocated at
       prepare). A dropped hop: the head of that block.
   * - ``ANIRA_MISS_ZEROS``
     - Silence, what the 2.x runtime did (the bundled RAVE encoder and decoder files use it).
   * - ``ANIRA_MISS_CALLBACK``
     - A function of the host fills it (below).

The policies speak about Streamed outputs; a Static output holds the handler's stored value on a
missed block too (:ref:`usage-static`).

**Dropped hops.** When every inference slot of the handler is still busy (an inference ran far
over its budget, the machine stalled), the hops a call completes get no inference: each one costs
exactly one hop of the policy's fill at its own stream position, behind the inferences submitted
before it. The stream stays on its hop grid and is back at its latency once the late inferences
are collected. ``rt_error`` reports a drop as ``ANIRA_ERROR_CAPACITY`` (:doc:`../logging`). A dropped
hop never reaches the model: a stateful model's state continues from the last hop it ran.

**The miss callback** is an ``anira_miss_fn``, set with ``anira_contract_hard_set_miss_fn(contract,
fn, user_data)`` (``Hard::miss_fn`` / ``miss_user_data``, ``ContractHandle::hard_miss_fn``); the
policy without a function is ``ANIRA_ERROR_CONFIG`` at prepare, and a contract file can say
``"on_miss": "callback"`` but not carry the function.

.. code-block:: c

    static anira_status ANIRA_CALL fade_out(anira_handler* handler,
                                            const anira_tensor* inputs, uint32_t num_inputs,
                                            const anira_tensor* outputs, uint32_t num_outputs,
                                            void* user_data) ANIRA_NONBLOCKING;

    anira_contract_hard_set_on_miss(contract, ANIRA_MISS_CALLBACK);
    anira_contract_hard_set_miss_fn(contract, fade_out, &my_state);   /* either order */

- **A missed block:** called once on the thread that called the entry, with one tensor per slot in
  slot order (the arrays the entry was handed): a slot the call did not carry, and every State slot,
  is an empty tensor (``shape[1] == 0``), a pop passes empty inputs, and a process form's input
  block is already pushed and intact. Fill every output whose ``shape[1]`` is above 0 and return
  ``ANIRA_OK``; any other status zero-fills them. The entry still returns ``ANIRA_MISSED``. Static
  output elements are filled with the stored value first, and the function may overwrite them.
- **A dropped hop:** called once per hop, at the drop, inside the entry whose call dropped it, with
  hop-shaped arrays: every Streamed input slot carries the dropped hop (``[channels, hop]`` of the
  ring dtype, packed, the handler's memory, valid for the call), every Streamed output slot a
  ``[channels, hop]`` tensor to fill, every other slot an empty tensor; a status other than
  ``ANIRA_OK`` places zeros. The entry's own return and counts are not touched by a hop; a
  function that must tell a hop from a block reads the flags (a hop's tensors are packed, without
  ``ANIRA_TENSOR_PLANAR``; a block's are the caller's own).
  The handler keeps the fills of the last ``ceil(latency / hop) + 2`` dropped hops and places an
  older one as zeros.
- The function is real-time code (no allocation, lock or system call), never calls a Hard entry,
  ``reset`` or ``prepare`` of the same handler, and outlives the prepared handler with its
  ``user_data``. Under clang it is declared ``ANIRA_NONBLOCKING`` itself, since the conversion to
  ``anira_miss_fn`` cannot add the attribute.

.. _usage-contract-options:

Wait ratio, ring dtypes, latencies, host domains, edge cost
-----------------------------------------------------------

.. code-block:: cpp

    hard.wait_ratio = 0.5;                                 // a _wait entry may wait half a block
    hard.ring_dtypes = {{"audio_in", ANIRA_DTYPE_I16}};    // the host streams int16 into a float32 model
    hard.latencies = {{"audio_out", 4096}};                // the reported latency of this output
    hard.edge_cost = ANIRA_EDGE_COPY;                      // no plan that crosses a domain
    contract.host_domain("audio_in", ANIRA_DOMAIN_HOST);   // where the host end lives (both kinds)

- **Wait ratio** (``wait_ratio``, default 0): the share of a block period a ``_wait`` entry waits
  for its inference under ``ANIRA_WAIT_CONTRACT`` (:ref:`usage-wait`); the ``ANIRA_NONBLOCKING``
  entries never wait. It selects the completion primitive at prepare (a semaphore above 0, a 1 ms
  poll otherwise), and the latency counts the wait credit whether or not the host calls a ``_wait``
  entry (:doc:`../latency`).
- **Ring dtypes** (``ring_dtypes``, ``hard_ring_dtype``): the element type of the host's samples per
  Streamed tensor, by canonical name, where it differs from the model's. Nothing converts: it needs
  a processor that converts (:ref:`usage-dtypes`).
- **Latencies** (``latencies``, ``hard_latency``): the stream latency of one output, in its samples,
  *replacing* the computed figure and priming its receive ring. ``ANIRA_ERROR_CONFIG`` at prepare
  below the model's internal latency, on an input, a Static output or an unknown name; above
  ``INT32_MAX`` the setter refuses it. Every output never named keeps the computed figure.
- **Host domains** (``host_domain``, both kinds): where the host end of one tensor lives, by
  canonical name, ``ANIRA_DOMAIN_HOST`` for every tensor never named. The ring, the model tensor, the
  Static store, the state buffers and every processor phase live there, and the planner joins it to
  the engine's domain with an edge (the plan report's slot row: the host domain is an input's
  ``from_domain`` and an output's ``to_domain``, the engine's the other end; no edge where the two
  coincide). An unknown name is ``ANIRA_ERROR_CONFIG``; any domain but ``ANIRA_DOMAIN_HOST`` is
  ``ANIRA_ERROR_NOT_SUPPORTED`` in this pre-release.
- **Edge cost** (``edge_cost``, both kinds; ``contract.edge_cost(...)``): the worst edge class a plan
  may take for any slot (:ref:`usage-edges`). ``std::nullopt``, the default, is no limit, an unknown
  edge included (in C ``anira_contract_set_edge_cost(c, ANIRA_EDGE_UNKNOWN)``, and
  ``anira_contract_edge_cost`` answers ``ANIRA_EDGE_UNKNOWN`` while no limit is set). Every limit
  refuses an ``ANIRA_EDGE_UNKNOWN`` edge. At prepare a plan above the limit is dropped once its model
  has loaded (an Info record names each slot, its class, its mover and why), and the report holds
  the eligible plans alone; with none left prepare throws ``ANIRA_ERROR_EDGE_UNREACHABLE``. The
  session is then made again over the remaining plans, so no model loads twice (an exclusive
  session's own executor is made, and warmed up, a second time).

.. _usage-async:

Async
-----

``anira::Async`` holds an optional ``deadline``, ``on_late``, ``priority``, ``lanes``,
``max_in_flight``, ``delivery`` and the ``edge_cost`` above: jobs without a real-time deadline, the
offline posture. An Async contract is refused at prepare in this pre-release.
