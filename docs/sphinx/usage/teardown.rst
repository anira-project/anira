.. _usage-teardown:

Teardown and unloading
======================

Destroy in reverse order: the handlers, then the contexts (and every inference thread you created,
``anira_inference_thread_destroy``). Destroying the last handler of this copy of anira drains its
in-flight inferences and joins the pool before it returns, so a host may unload a plugin that embeds
anira as soon as the plugin's last handler and context are gone.

.. code-block:: cpp

    // a plugin's module-exit entry point (CLAP deinit, VST3 ExitDll), after every instance is gone
    handler = nullptr;          // std::unique_ptr<anira::InferenceHandler>: drains its inferences;
                                // the last handler also joins the pool
    context = nullptr;          // std::unique_ptr<anira::Context>
    anira::shutdown();          // stops the core's threads; ANIRA_ERROR_INVALID_STATE while a
                                // context or a handler lives

- **Tear down from the instance's teardown or the module-exit entry point**, which run before the
  host's ``dlclose`` / ``FreeLibrary`` and outside the loader lock.
- **Never from the plugin's static destructors** (or ``DllMain``): they run inside the unload, under
  the loader lock, where a handler's destroy waits for its inferences, and an inference that needs
  the loader lock (LibTorch's first inference on a thread registers thread-local destructors under
  it; a runtime's lazy ``dlopen`` too) can never finish there.

On Linux and macOS a library-unload hook is the backstop for a host that unloads with a handler still
alive. It joins an idle pool. With an inference in flight it tells every inference thread to stop and
waits up to two seconds; one still running then can never finish during the unload, so the hook
writes

.. code-block:: text

    anira: the library is being unloaded while an inference is still running (1 in flight, 1 inference threads not stopped); destroy every anira handler, or call anira_shutdown(), before unloading. Aborting instead of hanging.

to stderr and aborts. At process exit (a return from ``main``, ``exit``) no loader lock is held, so it
waits without a bound and joins: a program that exits with an inference running exits normally. On
Windows the unload runs under the loader lock in ``DllMain``, where no thread may be waited for, so
the rule above is the only protection. :ref:`plugin-library-unload` has the rest: GCC's
``STB_GNU_UNIQUE``, macOS images that never unload, your own inference threads.
