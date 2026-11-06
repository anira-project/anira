Errors and logging
==================

anira reports a failure in one of two ways, never both for the same event on a control path: a call
that can hand the failure back hands it back and stays quiet; anira logs only where there is nobody
to hand it to; the real-time entries do both, because nobody checks their return on every block. The
runtime half of this page is the handler's (:ref:`usage-status`): ``rt_error``, the per-handler latch
and the drain summary.

The rule
--------

Every fallible C entry returns an ``anira_status``: negative values are failures, ``ANIRA_OK`` and
the positive values successes. An entry that can fail for more than one reason takes a caller-owned
``anira_error`` as its last parameter and writes the status and a message into it; the C++ face
throws it as an ``anira::Error``.

1. **Returned, not logged.** A control-path entry that fails (``prepare``, a loader, a setter, a
   create) writes the diagnosis into your ``anira_error`` and returns the status; C++ throws it. No
   log record: the message is yours.
2. **Logged, because there is no caller.** Where nothing can receive a status, anira logs one Error
   record: a failure inside a destroy or a void entry, on a thread anira owns (the inference
   threads, the drain thread), a sink that misbehaves, and ``ANIRA_ERROR_INTERNAL`` (below).
3. **Both, on the real-time path.** An ``ANIRA_NONBLOCKING`` entry (``process``, ``push_data``,
   ``pop_data``, ``submit``) has no ``anira_error``, so a refusal returns its failure status (with a
   delivered count of 0), records the status on the handler and pushes one record into the
   real-time log queue (:ref:`logging-realtime`). A missed block is no refusal: ``ANIRA_MISSED`` is
   a success, the buffers hold what the miss policy chose, and nothing is recorded or logged for the
   handler (the operational "missing samples" Warning, once per prepare, is the site's).

``anira::Error`` derives from ``std::runtime_error``: ``.status`` is the ``anira_status``,
``what()`` anira's message (the entry's name and the status text when the entry carries no error
record).

.. code-block:: cpp

    #include <anira/anira.hpp>

    try {
        anira::ModelConfig cfg = anira::ModelConfig::from_file("model.json");
        anira::ContextConfig context;
        context.log_level(ANIRA_LOG_WARNING);
        if (cfg.upgraded()) {
            // a 2.x document, upgraded in memory: a success (ANIRA_SUCCESS_UPGRADED), reported
            // as a flag on the handle rather than as an exception
            std::fprintf(stderr, "model.json is a 2.x file; write it out with to_json()\n");
        }
    } catch (const anira::Error& e) {
        // e.status: ANIRA_ERROR_NO_SUCH_FILE, ANIRA_ERROR_JSON, ...; e.what(): the message
        std::fprintf(stderr, "%s: %s\n", anira_status_string(e.status), e.what());
        return 1;
    }

In C, initialise the record with ``ANIRA_ERROR_INIT`` and test with ``ANIRA_FAILED``, never with ``!=
ANIRA_OK`` (a minor release may add positive statuses; ``ANIRA_SUCCESS_UPGRADED`` is one already).
Pass ``NULL`` for the ``anira_error`` when you do not want the message; the status is returned either
way, and out-parameters are left untouched on failure.

.. code-block:: c

    #include <anira/abi/config.h>

    anira_error err = ANIRA_ERROR_INIT;
    anira_model_config* cfg = NULL;
    anira_status status = anira_model_config_from_json_file("model.json", &cfg, &err);
    if (ANIRA_FAILED(status)) {
        fprintf(stderr, "%s: %s\n", anira_status_string(err.status), err.message);
        return 1;
    }
    if (status == ANIRA_SUCCESS_UPGRADED) {
        fprintf(stderr, "model.json is a 2.x file; write it out with anira_model_config_to_json\n");
    }

What a message contains
-----------------------

The reason comes first and the place second, so the reason survives truncation:

- a model the engine refused: ``"<engine>: <path>: <engine text>"`` (``memory`` in place of the path
  for a model loaded from bytes), ``ANIRA_ERROR_MODEL_LOAD``; an
  engine that failed at run time or in the warm-up, ``ANIRA_ERROR_ENGINE``, the same shape; an engine
  not in the build, ``ANIRA_ERROR_NOT_SUPPORTED``;
- a model file that does not resolve to a readable file: ``ANIRA_ERROR_NO_SUCH_FILE`` with the
  resolved absolute path and the engine, checked by anira before any engine sees the path, so the
  message is the same on every engine;
- a JSON document: ``ANIRA_ERROR_JSON`` with the key path and the value (``models[0].engine: "foo"
  is not one of ...``);
- prepare-time legality: ``ANIRA_ERROR_CONFIG`` with the tensor's canonical name and what disagrees
  (a window against its axis, a name the export lacks, an extension kind nobody consumes).

``anira_error::message`` holds 512 bytes (``ANIRA_ERROR_MESSAGE_CAPACITY``), UTF-8, NUL-terminated,
cut at the end; the record is 520 bytes on every target, so it lives on the stack, in a struct or in a
WebAssembly allocation of ``anira_sizeof(ANIRA_STRUCT_ERROR)``.

``ANIRA_ERROR_INTERNAL`` is a bug in anira, an exception the firewall could not classify: the one
returned failure anira also logs, once, at Error, with the entry's name. Report it with the log line.

**The trace flag.** ``ANIRA_LOG_FLAG_TRACE_FAILURES`` (``context.log_flags(...)``,
``anira_context_config_set_log_flags``, the ``flags`` of ``anira_log_desc``) makes every failed status
also an Error record, the ``anira_error`` message prefixed by the entry and the status: for an
application that swallowed the status when all you have is the device log. Off by default; in effect
while a context that set it lives (counted across contexts).

.. _logging-realtime:

The real-time path
------------------

An ``ANIRA_NONBLOCKING`` entry that refuses does at most three things:

1. It returns the failure status; a form with a count writes 0 (the multi forms zero their
   ``delivered`` array).
2. For a contract violation, it stores the status into the handler's ``rt_error``, a relaxed atomic
   readable through ``anira_handler_rt_error(h)`` from any thread and any callback, last-wins, cleared
   by ``prepare`` and ``reset``; a plain word in the handler, readable from a crash handler and a core
   dump:

   ``ANIRA_ERROR_WRONG_CONTRACT`` (a Hard entry on an Async handler, or the reverse),
   ``ANIRA_ERROR_NOT_PREPARED``, ``ANIRA_ERROR_CONFIG`` (a submitted tensor's dtype or axis tags, a
   ring accessor's dtype, a float entry on a non-float32 ring, a host tensor whose dtype is not its
   slot's, a plan index out of range), ``ANIRA_ERROR_INVALID_STATE`` (a ``_wait`` entry without an
   inference thread inside its loop, an accessor asked outside its phase),
   ``ANIRA_ERROR_INVALID_ARGUMENT`` (a NULL buffer, a slot or channel out of range, a malformed host
   tensor), ``ANIRA_ERROR_NOT_SUPPORTED`` (a host tensor with a flag bit this library does not know).

3. It pushes one fixed-size record into the core's lock-free log queue, at Error, flagged
   ``ANIRA_LOG_RECORD_REALTIME | ANIRA_LOG_RECORD_CONTRACT_VIOLATION``.

Two statuses reach ``rt_error`` from elsewhere, each logged by its own latched record rather than
as a violation: ``ANIRA_ERROR_ENGINE`` from the inference thread (a failed inference, whose output is
zeros), and ``ANIRA_ERROR_CAPACITY`` from the scheduler, for a hop it drops because every inference
slot is busy (the entry still returns its own status, the hop is filled under the miss policy at its
position, and the site "no free inference queue" logs one Warning per prepare). The same status
*returned* by an entry (no free ticket slot, a full ring) is back-pressure: the return value only. A
chunk a queue of the core has no room for is recorded too (zeros at its position, logged once per prepare): a prepare reserves room for
every chunk of its pool, so that takes an inference thread stalled inside its dequeue while the
handler's other chunks go round, or no memory for a drain's requeue.

**The latch.** A record is latched per handler and per kind of status: the first occurrence after
``prepare`` or ``reset`` is logged, later ones increment a suppressed counter with one relaxed add.
No clock is read on the real-time thread. The drain thread reports counters that grew since its last
summary at most every 10 seconds ("still failing, N more suppressed", naming the handler or the
condition) without re-arming anything, and the re-arm (``prepare`` and ``reset`` for a handler,
``prepare`` for the operational sites) logs the final count; the summary runs on the drain thread, or
in ``anira_drain_log`` under ``ANIRA_LOG_DRAIN_MANUAL`` and on WebAssembly. The same latch, per site,
covers the operational conditions that would otherwise fire on every block: missing samples, an output
stream nobody pops, an engine without a model, an inference dropped for a full queue, a host tensor
whose dtype is not its slot's (the input is not pushed, the output is zero-filled and reports 0), a
failed inference. A failed inference (an engine exception, or an engine call that returns a
failure) zero-fills its output, never the previous job's data, sets ``ANIRA_ERROR_ENGINE`` and is one
latched record; the inference thread survives it. A processor's phase that returns a status other
than ``ANIRA_OK`` latches *that* status and its chunk delivers zeros; the hop rule latches
``ANIRA_ERROR_CONFIG``; a refused accessor latches ``ANIRA_ERROR_CONFIG`` (a dtype not the ring's),
``ANIRA_ERROR_INVALID_STATE`` (asked outside its phase) or ``ANIRA_ERROR_INVALID_ARGUMENT`` (a slot
or channel out of range), each with one record naming the entry and the slot or channel
(:ref:`processor-accessors`).

Levels
------

The four levels are ``ANIRA_LOG_DEBUG`` (0) to ``ANIRA_LOG_ERROR`` (3). Every failure record is
Error; Warning is a condition the call survived (a clamped value, a process-global setting a later
context could not change, a slow plan); Info is lifecycle (the pool started, a session prepared);
Debug additionally switches the engines' verbose output on. The context config's ``log_level``
(default ``ANIRA_LOG_WARNING``; across contexts the most verbose wins) applies to anira's logger and is
forwarded to the engines' runtimes that take one (ONNX Runtime, LiteRT, LibTorch's c10; TFLite and
ExecuTorch expose none) at each engine's init, when its first model loads, with the level in effect
then (:ref:`engines-lifecycle`).

.. list-table::
   :header-rows: 1
   :widths: 12 22 24 42

   * - anira
     - Android (logcat)
     - Apple (``os_log``)
     - What the platform keeps
   * - Error
     - ``ANDROID_LOG_ERROR``
     - ``OS_LOG_TYPE_ERROR``
     - Apple persists it to the log store, and with it the process's recent in-memory Info records.
   * - Warning
     - ``ANDROID_LOG_WARN``
     - ``OS_LOG_TYPE_DEFAULT``
     - Apple persists it.
   * - Info
     - ``ANDROID_LOG_INFO``
     - ``OS_LOG_TYPE_INFO``
     - Apple keeps it in memory only, purged as the buffer fills, unless an Error follows or a
       configuration change persists it.
   * - Debug
     - ``ANDROID_LOG_DEBUG``
     - ``OS_LOG_TYPE_DEBUG``
     - Apple captures it only while it is streamed (``log stream --level debug``, Console.app with
       debug messages on).

logcat's buffers are rings in memory; all four priorities reach them, and ``log.tag.<tag>`` raises
one tag's minimum (``adb shell setprop log.tag.anira ERROR``). On Linux, Windows and the macOS terminal
the platform sink is stdout/stderr: Error and Warning on stderr, Info and Debug on stdout.

**Release builds.** tanh-lib, whose logger anira uses, compiles Warning, Info and Debug out of its
own ``Release`` builds (``THL_LOG_COMPILED_MAX_LEVEL=1``). anira sets ``TANH_LOG_COMPILED_MAX_LEVEL``
to 4 for its private copy in every build type, so the runtime level is the only filter and a warning
exists in a shipped build.

.. _logging-prepare:

What a prepare logs
-------------------

Every successful prepare logs its measurement and then its plan report, in the group ``anira.capi``
(:ref:`usage-plan-report`): one record per plan (Info; Warning for a marginal or not real-time plan),
a Warning for a pass that did not settle and for an explicit budget below its measurement, one record
for the session (its budget, the worst verdict, the latency and the entries; a Warning when the pool
cannot keep up), then the report at Info:

.. code-block:: text

    anira_handler_prepare: plan 0 (onnxruntime, cpu): warm-up until_stable, 6 inferences, stable; measured 1.842 ms (median 1.701 ms) on the inference threads; real-time factor 0.17 of 4 parallel inferences (load 0.04): comfortable (96 % headroom)
    anira_handler_prepare: plan 0 (onnxruntime, cpu): the model's figure from an earlier prepare (6 inferences, stable, 21 s ago): 1.842 ms (median 1.701 ms); real-time factor 0.17 of 4 parallel inferences (load 0.04): comfortable (96 % headroom)
    anira_handler_prepare: plan 1 (libtorch, cpu): warm-up until_stable, 5 inferences, stable; measured 9.100 ms (median 8.870 ms) on the inference threads; real-time factor 0.85 of 1 parallel inference (load 0.85): marginal (15 % headroom); the stream may miss blocks under load
    anira_handler_prepare: plan 0 (onnxruntime, cpu): the warm-up did not settle in 32 inferences (the last four 1.701-2.380 ms); the budget is their maximum, 2.380 ms
    anira_handler_prepare: plan 0 (onnxruntime, cpu): measured 1.842 ms per inference, above the explicit budget 1.500 ms; the latency is sized for 1.500 ms
    anira_handler_prepare: plan 0 (onnxruntime, cpu): no warm-up; real-time factor 0.47 of 4 parallel inferences (load 0.12) from the explicit budget 5.000 ms: unmeasured
    anira_handler_prepare: plan 1 (libtorch, cpu): warm-up until_stable, 5 inferences, stable; measured 48.120 ms (median 47.990 ms) on the inference threads; real-time factor 4.51 of 4 parallel inferences (load 1.13): NOT real-time on this machine; the stream will miss blocks, which on_miss fills
    anira_handler_prepare: budget 48.120 ms, measured (the slowest of 2 plans): not real-time; the pool cannot keep up (every hop brings 4.51 hop periods of inference for 4 parallel inferences): latency 3072 samples and 12 entries are sized for one host block's burst, and the stream will miss blocks
    anira_handler_prepare: budget 1.842 ms, measured (the slowest of 1 plan): comfortable; latency 512 samples, 4 entries
    anira_handler_prepare: plan report: plans 1, input slots 1, output slots 1, selected plan 0
    anira_handler_prepare: plan 0: variant 0, engine onnxruntime, provider cpu, budget 1.842 ms (measured), memory unified

These lines come from different prepares; one prepare writes one measurement record per plan, then
its Warnings, the session record and the report. Set the context's
level to ``ANIRA_LOG_INFO`` to see the Info records; a default run (Warning) shows the problems alone.

Delivery
--------

**Control path.** The sinks run synchronously on the thread that logs, before the entry returns: the
platform sink (one ``__android_log_print`` / ``os_log`` / ``fprintf`` plus ``fflush`` per record), the
file sink (flushed per record) and the host callback. A record emitted before a status is returned is
at the sink before you see the status: in logd's queue on Android, in the log store's shared memory on
Apple, in the kernel on stdio (stderr is unbuffered, stdout flushed per record even when redirected,
so a non-flushing ``abort()`` loses nothing).

**Real-time path.** A real-time record waits in the queue until it is drained: by the drain thread
every ``drain_interval_ms`` (default 10 ms; a low-priority thread named ``anira-log``, so that under
contention delivery is delayed rather than competing with the audio; the core starts it with the
first context and stops and joins it when the last context and handler are destroyed, flushing the
queue on the destroying thread), or by the host through ``anira_drain_log()`` under
``ANIRA_LOG_DRAIN_MANUAL``. That window is the only place a record can be lost, and it is closed where
it matters: **when a** ``[main-thread]`` **entry fails, it drains the real-time queue on the failing
caller's thread before it returns**, so the records that preceded the failure are in front of you
before you act on it (not for ``[callback-safe]`` entries, and not inside a sink). The drain is
lock-free and the queue multi-consumer, so a real-time record may arrive on the failing caller's thread
as well as on the drain thread. Releasing the last handler and ``anira_shutdown`` (``anira::shutdown()``)
drain it too.

**Manual mode and WebAssembly.** Under ``ANIRA_LOG_DRAIN_MANUAL`` there is no thread: the host pumps
``anira_drain_log`` (``anira::drain_log()``; ``anira.drainLog()`` or ``anira.startLogPump()`` on the
web) from a timer and once more after any failed call. The queue is the process's, so one pump drains
every handler. WebAssembly has no drain thread: ``ANIRA_LOG_DRAIN_THREAD`` is coerced to manual with a
warning, and a page that never pumps never sees a real-time record.

**A full queue** drops and counts records until the next drain, which reports how many were lost.
``queue_capacity`` (``log_queue_capacity``, rounded up to a power of two and clamped to [64, 65536],
fixed once the first context created the core) against the burst rate times the drain interval is the
rule of thumb.

**On a crash** everything a control-path call logged is at its sink; what is in the real-time queue,
and whatever a host sink buffered, is lost. anira installs no signal, crash or terminate handler and
makes no async-signal-safety promise for ``anira_drain_log``: keep the last records on the host side
(below).

Sinks
-----

The host owns the sinks; the context config says which run.

**The platform sink** is on by default: logcat on Android, ``os_log`` on macOS and iOS (on macOS also
stdout/stderr, and stdout/stderr *only* while a debugger is attached), stdout/stderr on Linux and
Windows, the browser console by level on WebAssembly (``console.debug`` … ``console.error``; tanh-lib's
own stderr sink, which the glue would print as ``console.error`` for every level, stays off there).
``ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK`` switches it off while that context lives (counted across
contexts), for a host that shows its own log or a DAW that mirrors stderr; the JavaScript log hook
(``anira.setLogSink()``, ``anira_em_set_log_hook``) takes the same hold while it is set, so a record
reaches the console once, at its level.

**The host sink.** ``log_sink(callback, user_data)`` (``anira_context_config_set_log_sink``, the
``callback`` of ``anira_log_desc``) receives **every** record the platform sink would, as an
``anira_log_record``: ``level``, ``flags`` (``ANIRA_LOG_RECORD_REALTIME`` through the queue,
``ANIRA_LOG_RECORD_CONTRACT_VIOLATION`` when the C layer or the wrapper raised it), ``dropped_before``
(records the queue lost before this one), ``sequence``, ``timestamp_ms`` (UTC), ``monotonic_ns``,
``group`` (``"anira.<component>"``) and ``message``, valid until the callback returns. The sink runs on
whichever thread logs, never on the driver thread, possibly under anira's lifecycle lock, and must not
call anira. A context's sink is called only while that context lives: destroying it unregisters the
sink and waits for a call in flight.

A sink that must survive a crash keeps a ring the host's own crash handler writes out. The sink side
may allocate and lock, the crash handler may not, so the ring is fixed storage, each slot written
completely before it is published, and the handler uses ``write(2)`` only:

.. code-block:: c

    enum { k_slots = 64, k_line = 320 };
    static struct { unsigned len; char text[k_line]; } g_ring[k_slots];
    static atomic_uint g_next;      /* records ever written */
    static int g_crash_fd = -1;     /* opened once at start-up: open(path, O_WRONLY|O_CREAT|O_APPEND) */

    static void ANIRA_CALL ring_sink(const anira_log_record* r, void* user_data) {
        (void)user_data;
        unsigned slot = atomic_fetch_add(&g_next, 1u) % k_slots;
        int n = snprintf(g_ring[slot].text, k_line, "%lld %s%s %s\n",
                         (long long)r->timestamp_ms,
                         (r->flags & ANIRA_LOG_RECORD_REALTIME) ? "[rt] " : "",
                         r->group, r->message);
        g_ring[slot].len = n < 0 ? 0u : (n >= k_line ? k_line - 1u : (unsigned)n);
    }

    /* the host's crash handler; async-signal-safe: no malloc, no stdio, no locks */
    static void flush_ring_on_crash(void) {
        unsigned end = atomic_load(&g_next);
        unsigned begin = end > k_slots ? end - k_slots : 0u;
        for (unsigned i = begin; i < end; ++i) {
            write(g_crash_fd, g_ring[i % k_slots].text, g_ring[i % k_slots].len);
        }
    }

    /* context.log_sink(ring_sink) or anira_context_config_set_log_sink(config, ring_sink, NULL) */

One slot may be torn when the crash lands inside a ``snprintf``: the price of a lock-free ring, and
why the handler writes the slot's recorded length rather than scanning for a terminator.

Where to look
-------------

anira files every record under its own identity: the Android tag and the Apple subsystem and category
are ``anira``, and every line reads ``[<source>][<group>] <message>``, ``source`` being ``native`` or
``rt`` and ``group`` ``anira.<component>`` (``anira.core``, ``anira.scheduler``, ``anira.config``,
``anira.system``, ``anira.engine.<engine>``, ``anira.web``, ``anira.capi``). A host that also uses
tanh-lib has a second logger under ``thl``, which anira never touches, with a shared or a static anira
(anira's own copy lives in ``thl::anira``: :ref:`host-tanh-lib`).

- **Android:** ``adb logcat -s anira:W`` for anira's warnings and errors, ``adb logcat -s anira`` for
  everything at the runtime level, ``adb logcat -s anira onnxruntime tflite ExecuTorch`` with the
  engines; ``adb shell setprop log.tag.anira DEBUG`` lowers the tag's minimum for the session.
- **macOS and iOS:** ``log stream --predicate 'subsystem == "anira"' --level info`` while the app runs
  (``--level debug`` for Debug), ``log show --predicate 'subsystem == "anira"' --last 1h`` afterwards,
  or Console.app with the subsystem in the search field and *Include Info Messages* / *Include Debug
  Messages*. Under a debugger on macOS the lines are on stdout/stderr (the Xcode console), not in
  ``log``.
- **Linux and Windows:** stderr for Error and Warning, stdout for Info and Debug; a plugin host shows
  them where it shows its own stderr, or not at all, which is what the host sink is for.
- **WebAssembly:** the browser console by level, through the log hook or the platform sink; the page
  pumps ``anira.drainLog()`` (``startLogPump``) for real-time records.
- **The engines** keep their own tags: ONNX Runtime's ``onnxruntime``, LiteRT's and TFLite's
  ``tflite``, ExecuTorch's ``ExecuTorch``, LibTorch's c10 warnings on stderr; anira forwards its level
  to those that take one and never re-logs their output.

**Privacy.** Every argument of anira's ``os_log`` calls is ``%{public}s``, and logcat has no private
mode: anyone with the device log (a customer's sysdiagnose included) sees model paths, tensor names
and engine messages. Keep personal data out of model paths.

Exceptions and app boundaries
-----------------------------

- ``anira/anira.hpp`` throws ``anira::Error`` and nothing else for a failed call (an allocation
  failure is ``std::bad_alloc``). Constructors throw, so a handle is never half-built; destructors,
  moves and ``native()`` are ``noexcept``; the wrapper catches and logs nothing. An exception-free mode
  (``ANIRA_CXX_NO_EXCEPTIONS``, ``anira::Result<T>``) comes with a later pre-release, and
  ``anira.hpp`` refuses the macro until then.
- The C entries never let an exception out: every control-path entry's function-try-block handler is
  the firewall, which turns ``std::bad_alloc`` into ``ANIRA_ERROR_OUT_OF_MEMORY``, anira's own
  exception into its status, an engine's into ``ANIRA_ERROR_MODEL_LOAD`` / ``ANIRA_ERROR_ENGINE``, and
  anything else into ``ANIRA_ERROR_INTERNAL``. Every entry is ``ANIRA_NOEXCEPT``, so an exception that
  escaped would be a deterministic ``std::terminate``, not undefined behaviour in an ``extern "C"``
  frame; exception types never leave ``libanira``. A failure inside a destroy or a void entry is
  logged once and swallowed.
- Real-time entries are ``noexcept`` and ``ANIRA_NONBLOCKING``, and nothing inside them throws by
  contract (throwing allocates; unwinding may take a global lock). Never throw in an audio callback of
  your own either: a plugin host (CLAP, VST3, AU, LV2) does not catch it. The wrapper's throwing methods
  are configuration-time only; the Hard entries cannot throw by construction.
- **App boundaries.** A C++ exception must not cross into Java, Objective-C or JavaScript: the glue
  checks the status and translates. JNI, with the C entries:

  .. code-block:: cpp

      extern "C" JNIEXPORT jlong JNICALL
      Java_com_example_Model_load(JNIEnv* env, jobject, jstring jpath) {
          const char* path = env->GetStringUTFChars(jpath, nullptr);
          anira_error err = ANIRA_ERROR_INIT;
          anira_model_config* cfg = nullptr;
          const anira_status status = anira_model_config_from_json_file(path, &cfg, &err);
          env->ReleaseStringUTFChars(jpath, path);
          if (ANIRA_FAILED(status)) {
              env->ThrowNew(env->FindClass("java/io/IOException"), err.message);
              return 0;
          }
          return reinterpret_cast<jlong>(cfg);
      }

  Objective-C++, with the C++ face: catch ``anira::Error`` at the method boundary and return an
  ``NSError``, never an ``NSException``:

  .. code-block:: objc

      - (BOOL)loadModelAtPath:(NSString*)path error:(NSError**)error {
          try {
              _config = std::make_unique<anira::ModelConfig>(
                  anira::ModelConfig::from_file(path.fileSystemRepresentation));
              return YES;
          } catch (const anira::Error& e) {
              if (error != nil) {
                  *error = [NSError errorWithDomain:@"anira"
                                               code:e.status
                                           userInfo:@{NSLocalizedDescriptionKey: @(e.what())}];
              }
              return NO;
          }
      }

  Both may log at their boundary; that is the platform's idiom for an app-level error, not anira's
  duplicate.

The design record behind this page is ``docs/anira-v3-error-and-log-strategy.md``; recipes for a log
that stays empty are in :doc:`troubleshooting`.
