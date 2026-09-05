Errors and logging
==================

anira reports a failure in one of two ways, and never in both for the same event on a control
path: a call that can hand the failure back to you hands it back and stays quiet; anira logs
only where there is nobody to hand the failure to; real-time entries do both, because nobody
checks their return value on every block. This page says what that means for the code you
write, what a message contains, where the records go on each platform, and what anira
promises about exceptions.

.. note::
    The runtime half of this page is the C handler of ``anira/abi/handler.h`` (:doc:`usage`,
    section 3.2): ``anira_handler_rt_error``, the per-handler latch and the drain summary
    are in effect there. The 2.x :cpp:class:`anira::InferenceHandler` of :doc:`usage`,
    sections 2 to 5, runs on the same scheduler: its operational real-time conditions are
    latched per site the same way and a failed inference delivers zeros, but it has no
    ``rt_error`` word to read.

The rule
--------

Every fallible C entry returns an ``anira_status``: negative values are failures, ``ANIRA_OK``
and the positive values are successes. Entries that can fail for more than one reason take a
caller-owned ``anira_error`` as their last parameter and write the status and a message into
it. The C++ builders of ``anira/anira.hpp`` turn that into an ``anira::Error``. Three cases:

1. **Returned, not logged.** A control-path entry that fails (``prepare``, a loader, a setter,
   a create) writes the diagnosis into your ``anira_error`` and returns the status; the C++
   face throws it. anira writes no log record for it: the message is yours, and a second copy
   in the system log would say nothing you do not already hold.
2. **Logged, because there is no caller.** Where nothing can receive a status, anira logs one
   Error record instead: a failure inside a destroy or a void entry, a failure on a thread anira
   owns (the inference threads, the drain thread), a sink that misbehaves, and
   ``ANIRA_ERROR_INTERNAL`` (below).
3. **Both, on the real-time path.** An ``ANIRA_NONBLOCKING`` entry (``process``, ``push_data``,
   ``pop_data``, ``submit``) has no ``anira_error`` and its return value is checked by nobody
   on every block, so a refusal there returns a count or a status, records the status on the
   handler, and pushes one record into the real-time log queue. Section
   :ref:`logging-realtime` has the details.

The C++ face: ``anira::Error`` derives from ``std::runtime_error``, ``.status`` is the
``anira_status`` and ``what()`` is anira's message (the entry's name and the status text when
the entry carries no error record).

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

The C face: initialise the record with ``ANIRA_ERROR_INIT``, test with ``ANIRA_FAILED`` (never
with ``!= ANIRA_OK``: a minor release may add positive statuses, and ``ANIRA_SUCCESS_UPGRADED``
is one already), read ``err.status`` and ``err.message``.

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

Pass ``NULL`` for the ``anira_error`` when you do not want the message; the status is returned
either way. Out-parameters are left untouched on failure.

What a message contains
-----------------------

The message names the reason first and the place second, so that the reason survives
truncation: an engine's own text, the model path or ``"memory"`` for a bytes entry, the key
path of a JSON document.

- A model the engine refused: ``"<engine>: <path>: <engine text>"`` with
  ``ANIRA_ERROR_MODEL_LOAD``; an engine that failed at run time or at warm-up,
  ``ANIRA_ERROR_ENGINE``, same shape; an engine that is not in the build,
  ``ANIRA_ERROR_NOT_SUPPORTED``.
- A model file that does not resolve to a readable file: ``ANIRA_ERROR_NO_SUCH_FILE`` with the
  resolved absolute path and the engine name; anira checks this itself before any engine sees
  the path, so the message is the same on every backend.
- A JSON document: ``ANIRA_ERROR_JSON`` with the key path and the offending value
  (``models[0].engine: "foo" is not one of ...``).
- Prepare-time legality: ``ANIRA_ERROR_CONFIG`` with the tensor's canonical name and what
  disagrees (window against axis, a tensor name the export does not have, an extension kind
  nobody consumes).

``anira_error::message`` holds 512 bytes (``ANIRA_ERROR_MESSAGE_CAPACITY``), UTF-8,
NUL-terminated; a longer message is cut at the end, which is why the reason comes first. The
record is 520 bytes on every target, so it can live on the stack, in a struct, or in a
WebAssembly heap allocation of ``anira_sizeof(ANIRA_STRUCT_ERROR)``.

``ANIRA_ERROR_INTERNAL`` means a bug in anira: an exception the firewall could not classify.
It is the one returned failure anira also logs, once, at Error, with the entry's name, because
the message is not anira's, you cannot act on it, and nothing below the firewall reported it.
Report it with the log line.

**The trace flag.** ``ANIRA_LOG_FLAG_TRACE_FAILURES`` on the context config
(``context.log_flags(ANIRA_LOG_FLAG_TRACE_FAILURES)``, ``anira_context_config_set_log_flags``,
or the ``flags`` field of ``anira_log_desc``) makes every failed status also an Error record
whose text is the ``anira_error`` message prefixed by the entry and the status. It is off by
default on every platform and exists for one situation: the application swallowed the status
and the only thing you have is the device log. It is in effect while a context that set it
lives (counted across contexts, so a second context's destroy does not switch it off for the
first).

.. _logging-realtime:

The real-time path
------------------

An ``ANIRA_NONBLOCKING`` entry does at most three things when it refuses:

1. It returns a count (``0`` samples) or an ``anira_status``.
2. It stores the status into the handler's ``rt_error``, a relaxed atomic readable through
   ``anira_handler_rt_error(h)`` from any thread and from inside any callback, when the refusal
   is a contract violation: ``ANIRA_ERROR_WRONG_CONTRACT`` (a Hard entry on an Async handler or
   the reverse), ``ANIRA_ERROR_NOT_PREPARED``, ``ANIRA_ERROR_CONFIG`` (a submitted tensor's
   dtype or axis tags, a ring accessor's dtype, a float entry on a non-float32 ring, a plan
   index out of range), ``ANIRA_ERROR_INVALID_STATE`` (a ``_wait`` entry without an inference
   thread inside its loop), ``ANIRA_ERROR_INVALID_ARGUMENT`` (a NULL buffer, a slot or channel
   out of range) — and, from the inference thread, ``ANIRA_ERROR_ENGINE`` after a failed
   inference, whose output is zeros. ``rt_error`` is last-wins and is cleared by ``prepare``
   and ``reset``; it is a plain word in the handler, readable from a crash handler and from a
   core dump, which is why it exists beside the best-effort record.
3. It pushes one fixed-size record into the core's lock-free log queue, at Error, flagged
   ``ANIRA_LOG_RECORD_REALTIME | ANIRA_LOG_RECORD_CONTRACT_VIOLATION`` in the
   ``anira_log_record`` a sink receives.

``ANIRA_ERROR_CAPACITY`` (no free ticket slot, a full ring) is back-pressure, not a violation:
return value only, no ``rt_error``, no record.

The record is **latched** per handler and per kind of status: the first occurrence after
``prepare`` or ``reset`` is logged, later ones increment a suppressed counter with one relaxed
add and nothing else. No clock is read on the real-time thread. Two things keep the latch
honest about a condition that persists: the drain thread, which is not real-time, reports
counters that grew since its last summary at most every 10 seconds ("still failing, N more
suppressed", with the handler or the condition), without re-arming anything, and the re-arm
(``prepare`` and ``reset`` for a handler, ``prepare`` for the operational sites) logs the final
count. The same latch, per site, covers the operational real-time conditions that would
otherwise fire on every block: missing samples, an output stream nobody pops, an engine
without a model, an inference dropped because a queue was full, a failed inference. A failed
inference — an engine exception, a throwing custom processor or hook — zero-fills its output
(never the previous job's data), sets ``rt_error`` to ``ANIRA_ERROR_ENGINE`` and is one
latched record; the inference thread survives it. The summary runs on the drain thread and,
under ``ANIRA_LOG_DRAIN_MANUAL`` and on WebAssembly, in ``anira_drain_log``.

Levels
------

anira's four levels are ``ANIRA_LOG_DEBUG`` (0) to ``ANIRA_LOG_ERROR`` (3). Every failure
record anira writes is Error; Warning is a condition the call survived (a clamped value, a
process-global setting a later context could not change); Info is lifecycle (the pool started,
a session prepared); Debug additionally switches the engines' verbose output on. The runtime
level of the context config (``log_level``; default ``ANIRA_LOG_WARNING``; the most verbose
request across the contexts of a process wins) is applied to anira's own logger and forwarded
to the engines' runtimes (ONNX Runtime, LiteRT, LibTorch/c10; TFLite and ExecuTorch expose no
runtime level).

The platform sink maps the levels onto the platform's own vocabulary, which decides what the
platform keeps:

.. list-table::
   :header-rows: 1
   :widths: 14 22 30 34

   * - anira
     - Android (logcat priority)
     - Apple (``os_log`` type)
     - What the platform does with it
   * - Error
     - ``ANDROID_LOG_ERROR``
     - ``OS_LOG_TYPE_ERROR``
     - Apple persists it to the log store and, with it, the process's recent in-memory Info
       records; every failure record anira writes is this level.
   * - Warning
     - ``ANDROID_LOG_WARN``
     - ``OS_LOG_TYPE_DEFAULT``
     - Apple persists it.
   * - Info
     - ``ANDROID_LOG_INFO``
     - ``OS_LOG_TYPE_INFO``
     - Apple keeps it in memory only, purged as the buffer fills, unless an Error follows or
       a configuration change persists it.
   * - Debug
     - ``ANDROID_LOG_DEBUG``
     - ``OS_LOG_TYPE_DEBUG``
     - Apple captures it only while it is being streamed (``log stream --level debug``,
       Console.app with debug messages enabled).

logcat's buffers are ring buffers in memory; all four priorities reach them by default, and
the ``log.tag.<tag>`` system property raises the minimum priority for one tag
(``adb shell setprop log.tag.anira ERROR``). On Linux, Windows and in the terminal on macOS the
platform sink is stdout/stderr, Error and Warning on stderr, Info and Debug on stdout.

**Release builds.** tanh-lib, whose logger anira uses, compiles Warning, Info and Debug out of
its own ``Release`` builds (``THL_LOG_COMPILED_MAX_LEVEL=1``). anira sets tanh-lib's
``TANH_LOG_COMPILED_MAX_LEVEL`` option to 4 for its private copy in every build type, so the
runtime level of the context config is the only filter and a warning the call survives exists
in a shipped build.

Delivery
--------

**Control path.** The sinks run synchronously on the thread that logs, before the entry
returns: the platform sink (one ``__android_log_print`` / ``os_log`` / ``fprintf`` plus
``fflush`` per record), the file sink (flushed per record) and the host callback. A record
emitted before a status is returned is therefore at the sink before you see the status: in
logd's queue on Android, in the log store's shared memory on Apple, in the kernel on stdio
(stderr is unbuffered; stdout is flushed per record even when redirected, so a
non-flushing ``abort()`` loses nothing). "Flush" is not a knob on the control path.

**Real-time path.** A real-time record sits in the queue until it is drained: by anira's
drain thread every ``drain_interval_ms`` (default 10 ms; a low-priority thread named
``anira-log``, so that under heavy CPU contention delivery is delayed rather than competing
with the audio path), or by the host through ``anira_drain_log()`` under
``ANIRA_LOG_DRAIN_MANUAL``. That window is the only place where a record can be lost, and it
is closed where it matters: **when a** ``[main-thread]`` **entry fails, the firewall drains
the real-time queue on the failing caller's thread before returning the negative status**, so
the real-time records that preceded the failure are in front of you before you act on it.
(Not for ``[callback-safe]`` entries, and not when the entry runs inside a sink.) The drain is
lock-free and the queue is multi-consumer, so this is safe beside the running drain thread; it
means a real-time record may arrive on the failing caller's thread as well as on the drain
thread, which the sink contract states. Releasing the last handler and ``anira_shutdown`` /
:cpp:func:`anira::Core::shutdown` drain the queue as well.

**Manual mode and WebAssembly.** With ``ANIRA_LOG_DRAIN_MANUAL`` there is no thread; the host
pumps ``anira_drain_log`` (:cpp:func:`anira::InferenceHandler::drain_log` on the 2.x runtime,
``drainAniraLog(wasmInstance)`` on the web) from a timer, and once more after any failed call.
The queue is shared by every handler in the process, so pumping any one of them drains
everything. WebAssembly has no drain thread at all: ``ANIRA_LOG_DRAIN_THREAD`` is coerced to
manual with a warning, and a page that never pumps never sees a real-time record.

**A full queue** drops and counts further records until the next drain, which then reports how
many were lost. ``queue_capacity`` (``log_queue_capacity``, rounded up to a power of two and
clamped to [64, 65536], fixed once the first context created the core) against the burst rate
times the drain interval is the rule of thumb.

**On a crash.** Everything a control-path call logged is already at its sink. What is in the
real-time queue at the moment of the crash is lost, and so is whatever a host sink buffered
without flushing. anira installs no signal, crash or terminate handler and makes no
async-signal-safety promise for ``anira_drain_log``; if the last records before a crash matter
to you, keep them on the host side, as in the next section.

Sinks
-----

The host owns the sinks. anira never decides where a record ends up; the context config says
which sinks run.

**The platform sink** is on by default: logcat on Android, ``os_log`` on macOS and iOS
(on macOS additionally stdout/stderr, and stdout/stderr *only* while a debugger is attached),
stdout/stderr on Linux and Windows, the browser console on WebAssembly.
``ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK`` in ``log_flags`` switches it off while that context
lives (counted across contexts), for a host that shows its own log or a DAW that mirrors
stderr.

**The host sink.** ``log_sink(callback, user_data)`` (``anira_context_config_set_log_sink``, or
the ``callback`` field of ``anira_log_desc``) registers an ``anira_log_fn`` that receives
**every** record the platform sink would, as an ``anira_log_record``: ``level``, ``flags``
(``ANIRA_LOG_RECORD_REALTIME`` when the record came through the queue,
``ANIRA_LOG_RECORD_CONTRACT_VIOLATION`` when the C layer or the wrapper raised it),
``dropped_before`` (records the queue lost before this one), ``sequence``, ``timestamp_ms``
(UTC), ``monotonic_ns``, ``group`` (``"anira.<component>"``) and ``message``, valid until the
callback returns. The sink runs on whichever thread logs, never on the driver thread, possibly
with anira's lifecycle lock held, and must not call anira. Each context's sink is called only
while that context lives: destroying the context unregisters it and waits for a call in
flight.

A sink that must survive a crash keeps a ring and lets the host's own crash handler write it
out. The sink side may allocate and lock; the crash-handler side may not, so the ring is fixed
storage, each slot is written completely before it is published, and the handler uses
``write(2)`` only:

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

One slot may be torn when the crash lands in the middle of a ``snprintf``; that is the price
of a lock-free ring and the reason the handler writes the slot's recorded length rather than
scanning for a terminator.

**Where to look.** anira's private logger files every record under anira's own identity: the
Android tag and the Apple subsystem and category are ``anira``, and every line reads
``[<source>][<group>] <message>`` with ``source`` = ``native`` or ``rt`` and ``group`` =
``anira.<component>`` (``anira.core``, ``anira.scheduler``, ``anira.config``,
``anira.system``, ``anira.backend.<engine>``, ``anira.web``, ``anira.capi``). A host that also
uses tanh-lib has a second logger under ``thl``, which anira never touches.

- Android: ``adb logcat -s anira:W`` for anira's warnings and errors, ``adb logcat -s anira``
  for everything at the runtime level; add the engines' own tags for a combined view,
  ``adb logcat -s anira onnxruntime tflite ExecuTorch``. ``adb shell setprop log.tag.anira DEBUG``
  lowers the tag's minimum for the session.
- macOS and iOS: ``log stream --predicate 'subsystem == "anira"' --level info`` while the app
  runs (``--level debug`` for Debug records), ``log show --predicate 'subsystem == "anira"'
  --last 1h`` afterwards, or Console.app with the subsystem in the search field and *Include
  Info Messages* / *Include Debug Messages* in the Action menu. Under a debugger on macOS the
  lines are on stdout/stderr (the Xcode console) instead, not in ``log``.
- Linux and Windows: stderr for Error and Warning, stdout for Info and Debug; a plugin host
  shows them where it shows its own stderr, or not at all, which is what the host sink is for.
- WebAssembly: the browser console (``console.log`` / ``console.error``); the page must pump
  ``drainAniraLog`` for real-time records.
- The engines keep their own tags and channels: ONNX Runtime's ``onnxruntime`` logger, LiteRT's
  and TFLite's ``tflite``, ExecuTorch's ``ExecuTorch``, LibTorch's c10 warnings on stderr;
  anira forwards its level to the ones that take one and never re-logs their output.

**Privacy.** Every argument of anira's ``os_log`` calls is ``%{public}s``, and logcat has no
private mode: what anira logs is visible to anyone with the device log, including a customer's
sysdiagnose. Model paths, tensor names and engine messages are part of the records; do not put
personal data in a model path you would not want in the system log.

Exceptions
----------

- ``anira/anira.hpp`` throws ``anira::Error`` and nothing else for a failed call (an
  allocation failure is ``std::bad_alloc``, as everywhere). Constructors throw, so that a handle
  is never half-built; destructors, moves and ``native()`` are ``noexcept``; nothing is caught
  or logged in the wrapper. The exception-free mode returns ``anira::Result<T>`` instead
  (``v3.0.0-alpha.2``).
- The C entries never let an exception out: every control-path entry is a function-try-block
  whose handler is the firewall, which turns ``std::bad_alloc`` into
  ``ANIRA_ERROR_OUT_OF_MEMORY``, anira's own status-carrying exception into its status, an
  engine's exception into ``ANIRA_ERROR_MODEL_LOAD`` / ``ANIRA_ERROR_ENGINE``, and anything it
  cannot classify into ``ANIRA_ERROR_INTERNAL`` (logged once, see above). Every entry is
  ``noexcept`` (``ANIRA_NOEXCEPT``), so an exception that did escape would be a deterministic
  ``std::terminate`` on every compiler rather than undefined behaviour in an ``extern "C"``
  frame. Exception types never leave ``libanira``. Destroy and void entries have no status to
  return; a failure inside one is logged, once, and swallowed.
- Real-time entries are ``noexcept`` and ``ANIRA_NONBLOCKING``, and nothing inside them throws
  by contract: throwing allocates, and unwinding may take a global lock. Never throw in an
  audio callback of your own either: a plugin host (CLAP, VST3, AU, LV2) does not catch, and a
  throw out of a callback is ``std::terminate`` or undefined behaviour. The wrapper's throwing
  methods are configuration-time only; the Hard entries cannot throw by construction.
- **App boundaries.** A C++ exception must not cross into Java, Objective-C or JavaScript; the
  glue checks the status and translates. JNI, with the C entries (no ``try`` needed):

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

  Objective-C++, with the C++ face: catch ``anira::Error`` at the method boundary and return
  an ``NSError``, never an ``NSException``:

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

  Both may log at their boundary as well; that is the platform's idiom for an app-level error
  and not anira's duplicate.

The design record behind this page is ``docs/anira-v3-error-and-log-strategy.md`` in the
repository; the per-platform recipes for a log that stays empty are in :doc:`troubleshooting`.
