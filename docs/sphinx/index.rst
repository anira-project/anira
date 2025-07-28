Anira Documentation
===================

.. image:: ../img/anira-logo.png
    :alt: anira Logo
    :align: center
    :width: 800px

.. image:: https://github.com/anira-project/anira/actions/workflows/build_test.yml/badge.svg
    :target: https://github.com/anira-project/anira/actions/workflows/build_test.yml
    :alt: Build Test
    :height: 20px
    :class: badge-inline

.. image:: https://github.com/anira-project/anira/actions/workflows/build_benchmark.yml/badge.svg
    :target: https://github.com/anira-project/anira/actions/workflows/build_benchmark.yml
    :alt: Build Benchmark
    :height: 20px
    :class: badge-inline
    
.. image:: https://github.com/anira-project/anira/actions/workflows/build_examples.yml/badge.svg
    :target: https://github.com/anira-project/anira/actions/workflows/build_examples.yml
    :alt: Build Examples
    :height: 20px
    :class: badge-inline
    
.. image:: https://github.com/anira-project/anira/actions/workflows/build_docs_and_deploy.yml/badge.svg
    :target: https://github.com/anira-project/anira/actions/workflows/build_docs_and_deploy.yml
    :alt: Build Docs
    :height: 20px
    :class: badge-inline
    
.. image:: https://github.com/anira-project/anira/actions/workflows/on_tag.yml/badge.svg
    :target: https://github.com/anira-project/anira/actions/workflows/on_tag.yml
    :alt: On Tag
    :height: 20px
    :class: badge-inline

.. raw:: html

    <style>
    .badge-inline {
      display: inline-block !important;
      margin-right: 5px;
    }
    </style>


**anira** is a high-performance library designed to enable easy real-time safe integration of neural network inference within audio applications. Compatible with multiple inference backends, `LibTorch <https://github.com/pytorch/pytorch/>`_, `ONNXRuntime <https://github.com/microsoft/onnxruntime/>`_, and `Tensorflow Lite <https://github.com/tensorflow/tensorflow/>`_, anira bridges the gap between advanced neural network architectures and real-time audio processing. In the `paper <https://doi.org/10.1109/IS262782.2024.10704099>`_ you can find more information about the architecture and the design decisions of **anira**, as well as extensive performance evaluations with the built-in benchmarking capabilities.

Features
--------

- **Real-time Safe Execution**: Ensures deterministic runtimes suitable for real-time audio applications
- **Thread Pool Management**: Utilizes a static thread pool to avoid oversubscription and enables efficient parallel inference
- **Minimal Latency**: Designed to minimize latency while maintaining real-time safety
- **Built-in Benchmarking**: Includes tools for evaluating the real-time performance of neural networks
- **Comprehensive Inference Engine Support**: Integrates common inference engines, LibTorch, ONNXRuntime, and TensorFlow Lite
- **Flexible Neural Network Integration**: Supports a variety of neural network models, including stateful and stateless models
- **Cross-Platform Compatibility**: Works seamlessly on macOS, Linux, and Windows

Quick Start
-----------

The basic usage of anira is as follows:

.. include:: getting_started.rst
    :start-after: .. _basic-usage-example:
    :end-before: .. _basic-usage-example_end:

Installation
------------

.. include:: getting_started.rst
    :start-after: .. _installation:
    :end-before: .. _installation_end:

Documentation
-------------

.. toctree::
    :maxdepth: 1
    :caption: Contents:
    
    about
    getting_started
    usage
    custom_preprocessing
    custom_backends
    examples
    api/index
    architecture
    benchmarking
    latency
    troubleshooting
    changelog
    contributing

Citation
--------

If you use anira in your research or project, please cite either the `paper <https://doi.org/10.1109/IS262782.2024.10704099>`_ or the software itself:

.. code-block:: bibtex

    @inproceedings{ackvaschulz2024anira,
        author={Ackva, Valentin and Schulz, Fares},
        booktitle={2024 IEEE 5th International Symposium on the Internet of Sounds (IS2)},
        title={ANIRA: An Architecture for Neural Network Inference in Real-Time Audio Applications},
        year={2024},
        volume={},
        number={},
        pages={1-10},
        publisher={IEEE},
        doi={10.1109/IS262782.2024.10704099}
    }

    @software{ackvaschulz2024anira,
        author = {Valentin Ackva and Fares Schulz},
        title = {anira: an architecture for neural network inference in real-time audio application},
        url = {https://github.com/anira-project/anira},
        version = {x.x.x},
        year = {2024},
    }

License
-------

This project is licensed under `Apache-2.0 <https://github.com/anira-project/anira/blob/main/LICENSE>`_.

