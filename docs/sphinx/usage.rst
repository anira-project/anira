Usage Guide
===========

A host describes a deployment with three objects and runs it with a handler:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Object
     - What it says
   * - ``anira::ModelConfig`` with ``anira::TensorSpec``
     - The model: its exports per engine and its tensors. Travels with the model (the model file).
   * - ``anira::ContractHandle`` (``anira::Hard`` / ``anira::Async``)
     - How it runs: the host blocks, the budget and warm-up, the miss policy (the contract file).
   * - ``anira::ContextConfig``, ``anira::Context``
     - The process: the inference threads, logging, the devices (the context file).
   * - ``anira::Pipeline``, :cpp:class:`anira::InferenceHandler`
     - What runs and the runtime: the inference stage, a processor, custom engines; the real-time
       entries.

Each is the C++ face of a C handle of ``anira/abi/`` (:doc:`usage/c_api`). The whole flow:

.. code-block:: cpp

    #include <anira/anira.hpp>

    // Once, off the audio thread
    anira::ModelConfig model = anira::ModelConfig::from_file("model.model.json");
    anira::ContractHandle contract = anira::ContractHandle::from_file("model.contract.json");
    anira::Context context{anira::ContextConfig{}};
    anira::Pipeline pipeline;
    pipeline.add_inference(model);
    anira::InferenceHandler handler(context, pipeline);

    contract.hard_host_blocks(1, max_block, sample_rate);    // the host's block range and rate
    const anira::PlanReport report = handler.prepare(contract);
    const anira_plan_info row = report.plans()[handler.get_plan()];
    if (static_cast<anira_rt_verdict>(row.rt_verdict) == ANIRA_RT_NOT_REALTIME) {
        // too slow on this machine: hold the plan and ask the user ("Try nonetheless")
    }
    host_set_latency(handler.get_latency());

    std::array<float*, 2> channels{};
    anira::Tensor audio = anira::Tensor::from_host_planar<float>(
        std::span<float* const>(channels), std::array<int64_t, 2>{2, max_block});

    // The audio callback
    void process(float* const* data, size_t num_samples) noexcept {
        channels = {data[0], data[1]};
        audio.shape[1] = static_cast<int64_t>(num_samples);
        handler.process(audio, 0, audio, 0);
    }

.. toctree::
   :maxdepth: 1

   usage/models
   usage/contracts
   usage/files
   usage/context
   usage/handler
   usage/processing
   usage/state
   usage/dtypes
   usage/edges
   usage/tensors
   usage/c_api
   usage/teardown

Pre- and post-processing of your own is a processor (:doc:`custom_preprocessing`), an engine of
your own a registered engine (:doc:`custom_engines`).

.. _usage-pre-release:

What this pre-release does not run yet
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Not yet
     - Refused with
   * - the Async contract
     - ``ANIRA_ERROR_NOT_SUPPORTED`` at prepare
   * - a Buffer spec under a Hard contract
     - ``ANIRA_ERROR_NOT_SUPPORTED`` at prepare, naming the tensor
   * - a host domain other than ``ANIRA_DOMAIN_HOST``
     - ``ANIRA_ERROR_NOT_SUPPORTED`` at prepare, naming the tensor
   * - a tensor layout that moves an axis of an extent above 1 (a transpose)
     - ``ANIRA_ERROR_NOT_SUPPORTED`` at prepare
   * - device blocks on the context config (every context is Host-only)
     - ``ANIRA_ERROR_NOT_SUPPORTED`` at ``anira_context_create``
   * - model tensors with quantisation parameters at the boundary
     - ``ANIRA_ERROR_NOT_SUPPORTED`` at load
   * - device tensors (the GPU and draft factories are field fills no adapter consumes), device chunks
     - nothing reads their memory arms
   * - reading a declared state back from the host
     - no entry
   * - the exception-free C++ mode ``ANIRA_CXX_NO_EXCEPTIONS``
     - ``anira.hpp`` refuses the macro

.. note::
    Coming from anira 2.x? :doc:`migration` maps the 2.x classes, calls and JSON documents onto the
    3.x API.
