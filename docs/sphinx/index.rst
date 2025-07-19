.. Anira documentation master file, created by
   sphinx-quickstart on Sat Jul 19 15:52:50 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Anira documentation
===================

Anira is a C++ library for real-time neural network inference in audio processing applications.
It provides a high-level interface for performing neural network inference with support for
multiple backends including LibTorch, ONNX Runtime, and TensorFlow Lite.

API Reference
=============

Core Classes
------------

.. doxygenclass:: anira::InferenceHandler
   :members:

.. doxygenclass:: anira::InferenceConfig
   :members:

.. doxygenclass:: anira::PrePostProcessor
   :members:

Configuration and Data Structures
---------------------------------

.. doxygenstruct:: anira::ContextConfig
   :members:

.. doxygenstruct:: anira::ModelData
   :members:

.. doxygenstruct:: anira::TensorShape
   :members:

.. doxygenstruct:: anira::ProcessingSpec
   :members:

Backend Processors
------------------

.. doxygenclass:: anira::BackendBase
   :members:

.. doxygenclass:: anira::LibtorchProcessor
   :members:

.. doxygenclass:: anira::TFLiteProcessor
   :members:

.. doxygenclass:: anira::OnnxRuntimeProcessor
   :members:

Scheduler and Management
------------------------

.. doxygenclass:: anira::InferenceManager
   :members:

.. doxygenclass:: anira::InferenceThread
   :members:

.. doxygenclass:: anira::Context
   :members:

.. doxygenclass:: anira::SessionElement
   :members:

Utilities
---------

.. doxygenclass:: anira::JsonConfigLoader
   :members:

.. doxygenstruct:: anira::HostAudioConfig
   :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

