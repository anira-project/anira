# anira v3: errors and logs — the strategy (draft, 2026-09-03)

Status: draft for the owner's review; intended to become the amendment of decision-record
principle 7 and of the architecture document's logging section (4), real-time refusals (6a)
and thread tags. Sources: a survey of 25 libraries, tanh-lib and anira as built on the v3
branch, the Android/Apple platform documentation, and the C++/JNI/ObjC++/Emscripten/plugin-host
rules. Where this text says "measured", a reader checked the tree or the platform source.

## 1. The five rules

1. **A failure is data.** It travels as `anira_status` plus the caller-owned `anira_error`
   (control path) or as the per-handler `rt_error` (real-time path) to the one caller who
   decides, and anira logs nothing it returns as a negative status. This is Abseil's written
   rule ("low-level routines ... should typically not log status values themselves, but pass
   them up") and is already the de-facto rule of `src/capi` and of `anira.hpp`.
2. **A log record substitutes for a missing channel; it never copies one.** anira logs where no
   caller can receive the failure (anira-owned threads, void and destroy entries, sinks, the
   drain) and where the channel cannot carry a message and is presumed unread (real-time
   refusals). The second case carries `ANIRA_LOG_RECORD_CONTRACT_VIOLATION` when it is the
   caller's fault. The justification is SQLite's (`SQLITE_MISUSE` is logged "when return codes
   are not consistently checked") and Vulkan's (ERROR = "violated a valid usage condition"),
   not FFmpeg's log-and-return.
3. **Never say it twice.** No layer re-logs what the layer below reported (the kernel's
   `OOM_MESSAGE` rule): the wrapper never logs a C refusal, the C layer never re-logs an
   engine's own output, the firewall never logs a status it classified. A repeating real-time
   condition is latched: first occurrence logged, suppressed occurrences counted, the count
   reported when the condition clears or by the drain thread's slow summary.
4. **Every failure record is Error, and the compiled ceiling is pinned.** Error is the one level
   that exists in a Release build today (tanh-lib compiles Warning/Info/Debug out with
   `THL_LOG_COMPILED_MAX_LEVEL=1`), the one Android "always logs" and the one Apple persists
   and that promotes the process's recent Info records to disk. anira's private tanh copy pins
   the ceiling to 4 in every build type (the property edit the decision record already
   specifies; the tree does not do it yet), so that the runtime level in `anira_log_desc` is
   the only filter and "warnings the call survives" exist in shipped builds.
5. **Exceptions stop at the boundaries; nothing anira does aborts the host.** Every C entry is
   `noexcept` (`ANIRA_NOEXCEPT`, defined and unused today) with a function-try-block whose
   handler is the firewall; `anira.hpp` throws `anira::Error` and nothing else and never logs;
   destroy and void entries are log-only; anira installs no signal or terminate handler and
   promises nothing async-signal-safe.

## 2. Control path: returned, not logged — after widening the channel

The 2.x "log then throw" (16 sites: 14 in `src/InferenceConfig.cpp`, 2 in
`src/backends/LibTorchProcessor.cpp`) existed because the throw was narrower than the log:
the LibTorch site logs `e.what()` and throws a `std::runtime_error` without it. Retiring the
log therefore comes second; first the return channel carries the whole diagnosis:

- `StatusError` moves out of `src/capi` to a private `src/utils` header, derives from
  `std::runtime_error`, and the backends throw it with the right status:
  `MODEL_LOAD` / `ENGINE` / `NOT_SUPPORTED`, message `"<engine>: <path|memory>: <engine text>"`.
  Today every `std::runtime_error` becomes `ANIRA_ERROR_INTERNAL`, whose header meaning is
  "a bug in anira".
- The message keeps the reason before the path, and truncation into the 512-byte
  `anira_error` keeps the head (an iOS container path is ~130 bytes; ORT already repeats it).
- anira checks model files itself before any engine sees them and returns
  `ANIRA_ERROR_NO_SUCH_FILE` with the resolved absolute path and the engine name: the most
  common field failure, and today no backend produces that status (LiteRT and ExecuTorch
  report "failed with status N"; TFLite never null-checks `TfLiteModelCreateFromFile`).
- Engine failures at warm-up fail `prepare` (`ENGINE`); today the ORT adapter logs and
  constructs the session anyway. TFLite's ignored `TfLiteStatus` becomes a failure.
- The 2.x log-and-swallow loader (`JsonConfigLoader`, ~45 log-only sites, null pointer as the
  only signal) is already replaced by the v3 loaders (`ANIRA_ERROR_JSON` with the key path).

The only control-path records: (a) `ANIRA_ERROR_INTERNAL`, logged once at Error by the firewall
with the entry name — the non-fatal CHECK: not anira's message, not actionable by the caller,
and nothing below logged it; (b) a failure swallowed by a void or destroy entry, one Error
record naming the entry (today `translate_exception(nullptr)` swallows silently); (c) the
optional boundary trace, `ANIRA_LOG_FLAG_TRACE_FAILURES` on the machine config: one Error
record per failed status whose bytes are the `anira_error` message prefixed by entry and
status. Off by default; it is the switch for "the app swallowed the error and I only have
logcat". Control-path contract misuse (`INVALID_STATE`, `WRONG_CONTRACT`, `NOT_PREPARED` on a
main-thread entry) is returned only: the channel carries the message, the wrapper throws, and
the trace flag covers the C caller who ignores statuses.

## 3. Real-time path: status, `rt_error`, one latched record

An `ANIRA_NONBLOCKING` entry does at most three things on failure: returns a count or a
status; stores the status into the handler's relaxed atomic `rt_error` when it is a contract
violation (`WRONG_CONTRACT`, `NOT_PREPARED`, `CONFIG` for a dtype or axis mismatch,
`INVALID_STATE`); pushes one fixed-size record into the core's lock-free queue. `CAPACITY`
(no free ticket, a full ring) is back-pressure, not a violation: return value only, never
`rt_error`, never a record. `rt_error` is last-wins, cleared by `prepare` and `reset`, readable
from a crash handler and a core dump, which is why it exists beside the best-effort record.

The record is Error, flags `REALTIME | CONTRACT_VIOLATION`, and is **latched per handler per
kind** (a bit per status kind plus a suppressed counter): the first occurrence after `prepare`
or `reset` is logged; later ones increment the counter with one relaxed RMW. The same latch,
per site, covers the operational real-time conditions that fire every block today (missing
samples, output not consumed, engine failure per inference, dropped task): the tree has no
rate limiting at all; the harm is sink spam (one `writev` or `os_log` per block through the
drain thread, and logd drops silently under overload).

Two things keep a latch honest about persistence: the drain thread, which is not real-time,
reports counters that changed since its last summary at most every 10 s ("still failing, N
suppressed" with the handler and kind), and the re-arm logs the final count when the
condition clears. No clock is read on the real-time thread; the cost there is one relaxed
increment.

Output after a failed inference is defined: zeros, never the previous job's data (today ORT
copies stale outputs, LiteRT and ExecuTorch leave the buffer, TFLite ignores the status).
The inference thread catches per task (LibTorch's `Instance::process` has no catch today and
leaves the instance busy forever), zero-fills, sets `rt_error = ENGINE`, and emits one latched
record; a thread body never exits through tanh-lib's synchronous top-level catch.

## 4. Delivery and flush, per platform

Control path: the sinks run synchronously on the calling thread before the entry returns
(measured: platform sink, console with `fprintf`+`fflush` per record, file with `flush()` per
record, host callback). So a record emitted before a status is returned is at the sink before
the caller sees the status: in logd's kernel queue on Android (`writev` datagram to
`/dev/socket/logdw`), in logd-shared memory on Apple (`os_log`), in the kernel on stdio
(`stderr` unbuffered, a redirected `stdout` flushed per record, so glibc >= 2.27's
non-flushing `abort()` loses nothing). "Flush" is therefore not a knob on the control path.

Real-time path: the only loss window is queue -> drain (10 ms default on a Low-priority
thread; unbounded in Manual mode and on WebAssembly until the host pumps). It is closed
where it matters: **the firewall's failure path drains the real-time queue on the failing
caller's thread before returning a negative status**, for `[main-thread]` entries only (not
for `[callback-safe]` entries such as the future `ticket_error`, and not when the entry runs
inside a sink), so the real-time records that preceded a failure are in front of the host
before it acts. `Context::drain_log` is already lock-free and the queue is MPMC, so this is
safe beside the running drain thread. It changes the thread contract of real-time records
(they may arrive on the failing caller's thread as well as the drain thread); the thread-tag
table says so. The last-session release and `shutdown` already flush.

What anira does not do: no crash handler, no async-signal-safe drain, no retry of logd's
`EAGAIN`. The host owns crash-time preservation through its `anira_log_fn`, which receives
every record; the docs ship a sample sink that keeps a ring and flushes from the host's crash
handler, plus the Manual-mode rule: pump `anira_drain_log` after any failed call and once per
frame.

## 5. Sinks, identity and mobile completeness

- anira's private tanh copy emits under its own identity: Android tag `anira` (today `thl`;
  developers filter by tag and gate by `setprop log.tag.anira`), Apple subsystem reverse-DNS
  with the anira group as category (today `thl`/`logger`), `%{public}s` on every argument
  (already so; without it the message is `<private>` on a customer device). A tanh-lib change,
  filed there. Engines keep their own tags; the troubleshooting page gives the combined filter.
- The level table, documented: Error -> `ANDROID_LOG_ERROR` / `OS_LOG_TYPE_ERROR` (persisted);
  Warning -> `WARN` / `DEFAULT` (persisted); Info -> `INFO` / `INFO` (memory only);
  Debug -> `DEBUG` / `DEBUG` (only when streamed). Default runtime level `WARNING`.
- Every record reaches `anira_log_fn` with `flags` (`REALTIME`, `CONTRACT_VIOLATION`) and
  `dropped_before`. Neither has a carrier in tanh-lib today: a flags field on the record and a
  drop count beside the first record after a gap are tanh-lib changes to land and re-pin before
  the projection ships; until then the flags ride as a group suffix the trampoline strips.
- WebAssembly: the browser console is the sink; the emscripten wrappers' direct
  `emscripten_log` calls route through the logger so level, group and the JS hook apply.
- Glue is part of the contract: the JNI shim checks `ANIRA_FAILED` and `ThrowNew`s the
  `anira_error` message (never a C++ exception across JNI); the Objective-C++ face returns
  `NSError`, never `NSException`; both may additionally log at their boundary, which is the
  platform idiom and not anira's duplicate.

## 6. Exceptions

- C ABI: every entry `noexcept` plus the function-try-block; an escape is a deterministic
  `std::terminate` on every compiler instead of MSVC `/EHsc`'s undefined behaviour for
  `extern "C"`. Exception types never leave `libanira` (hidden visibility stays safe).
- Real-time entries are `noexcept` and `ANIRA_NONBLOCKING`; throwing allocates and (before
  glibc 2.35) takes a global unwinder lock; clang's function-effect analysis diagnoses a throw
  in a consumer's nonblocking TU at compile time; anira's internal RT bodies are covered by
  RTSan and review until the internal chain is annotated.
- `anira.hpp`: every control-path method throws `anira::Error{status, message}`; constructors
  throw so that a handle is never half-built; destructors, moves, `native()` are `noexcept`;
  nothing is caught or logged in the wrapper. The exception-free mode (v3.0.0-alpha.2) returns
  `Result<T>` (`std::expected<T, anira::Error>` under C++23), Vulkan-Hpp's shape.
- Plugin hosts (CLAP, VST3, AU, LV2) never catch; a throw out of a callback is terminate or
  UB. anira's Hard entries cannot throw by construction; the wrapper's throwing methods are
  configuration-time only.

## 7. Changes to the tree, in order

1. tanh-lib: identity (tag/subsystem/category) configurable per copy; `flags` on
   `RtRecord`/`LogRecord` and a flags-taking `logf`; drop count beside the first record after
   a gap; a cache option for the compiled ceiling. Re-pin.
2. anira: pin `THL_LOG_COMPILED_MAX_LEVEL=4` on the private copy (configure-time check).
3. `StatusError` to `src/utils`, derived from `std::runtime_error`; backends throw
   `MODEL_LOAD`/`ENGINE`/`NOT_SUPPORTED` with `"<engine>: <path>: <text>"`; model-file
   pre-check -> `NO_SUCH_FILE`; TFLite null checks and status checks; ORT warm-up failure fails
   prepare; retire the 16 log-then-throw sites by widening the thrown text.
4. Firewall: entry name captured by the defining macro; `ANIRA_NOEXCEPT` on every entry;
   `INTERNAL` logged once at Error; `report_void_failure` for destroy/void entries;
   failure-path drain for `[main-thread]` entries with the in-sink guard;
   `ANIRA_LOG_FLAG_TRACE_FAILURES`.
5. Real-time: `rt_error` on the handler (M2), the per-handler and per-site latches with
   suppressed counters, the drain-thread summary, zero-fill after a failed inference, the
   LibTorch catch and RAII guard, the inference-thread body wrapper.
6. Docs: principle 7 gains the rule and its two exceptions; section 4 gets the identity, the
   level table, the delivery guarantees and the sample crash-safe sink; section 6a the
   `rt_error` semantics and the latch (replacing "logs once"); the thread-tag table the
   failing-caller delivery of real-time records; troubleshooting the per-platform filters.

## 8. Tests that pin it

The firewall logs nothing for every classified status and exactly once for `INTERNAL`; a
destroy that swallows a failure logs once; a real-time refusal sets `rt_error` and emits
exactly one flagged record until re-arm, then the suppressed count; the drain summary after
a persistent condition; a failed `[main-thread]` entry delivers the queued real-time record
before returning (Manual mode); the compiled ceiling is 4 in Release; message truncation keeps
the reason; a missing model file is `NO_SUCH_FILE` with the path on every backend; a failed
inference zero-fills; the LibTorch instance is not left busy after a `c10::Error`.
