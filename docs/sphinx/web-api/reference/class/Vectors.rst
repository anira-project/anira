Class Vectors
=============

Thin wrappers over ``std::vector<T>`` for the element types anira's
C++ API exposes. Each ``Vector*`` class shares the same surface
(``size``, ``push``, ``get``, ``destroy``) and inherits from
``VectorBase``.

.. note::
   Construct via the :doc:`AniraWeb` factory in most cases:
   ``aniraWeb.VectorSizeT([2, 1])`` rather than
   ``new VectorSizeT(...)``. The factory threads the WASM instance
   through for you, and most constructors accept a plain JS array as
   a convenience initializer.

Common Surface
--------------

Every ``Vector*`` class supports:

* ``size()`` → ``number`` — element count.
* ``push(value)`` — append one element. Element type depends on the
  class (see table below). Object-vector ``push`` accepts either a
  wrapper instance or its raw pointer.
* ``get(index)`` — read one element. Primitive vectors return the
  element by value; object vectors return a non-owning raw pointer
  to the underlying C++ element.
* ``destroy()`` — free the underlying C++ object. See
  :ref:`lifecycle-and-cleanup`.

Most constructors also accept a JS array of initial values that is
pushed eagerly during construction.

The Vector Family
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 28 20

   * - Class
     - Underlying C++ type
     - Element type
     - ``get`` returns
   * - ``VectorSizeT``
     - ``std::vector<size_t>``
     - ``number``
     - ``number``
   * - ``VectorInt64T``
     - ``std::vector<int64_t>``
     - ``bigint``
     - ``bigint``
   * - ``VectorFloat``
     - ``std::vector<float>``
     - ``number``
     - ``number``
   * - ``VectorUnsignedInt``
     - ``std::vector<unsigned int>``
     - ``number``
     - ``number``
   * - ``VectorVectorInt64``
     - ``std::vector<std::vector<int64_t>>``
     - ``VectorInt64T`` / ``number[]`` / pointer
     - ``number`` (raw pointer)
   * - ``TensorShapeList``
     - alias of ``VectorVectorInt64``
     - same as ``VectorVectorInt64``
     - same as ``VectorVectorInt64``
   * - ``VectorModelData``
     - ``std::vector<anira::ModelData>``
     - :doc:`ModelData` / pointer
     - n/a — push-only
   * - ``VectorTensorShape``
     - ``std::vector<anira::TensorShape>``
     - :doc:`TensorShape` / pointer
     - n/a — push-only
   * - ``VectorRingBuffer``
     - ``std::vector<anira::RingBuffer>``
     - :doc:`RingBuffer` / pointer
     - ``number`` (raw pointer to the element)
   * - ``VectorBufferF``
     - ``std::vector<anira::Buffer<float>>``
     - :doc:`BufferF` / pointer
     - ``number`` (raw pointer to the element)

``VectorVectorInt64`` accepts several convenience forms during
construction:

* ``number[][]`` — each inner array is auto-converted to
  ``bigint`` and wrapped in a temporary ``VectorInt64T`` that is
  destroyed after the inner copy is taken.
* ``bigint[][]`` — same thing, used as-is.
* ``(VectorInt64T | number)[]`` — existing wrapper instances or
  raw pointers; the C++ side copies each inner vector.

``TensorShapeList`` is identical to ``VectorVectorInt64``; it exists
purely so call sites that build tensor shapes read more naturally.

Typical Use
-----------

.. code-block:: typescript

   // Primitive vectors initialised from a JS array.
   const channels = aniraWeb.VectorSizeT([2, 1])
   const sizes = aniraWeb.VectorSizeT([512, 0])

   // Object vector: pass wrapper instances directly.
   const modelData = aniraWeb.ModelData(modelBuffer, aniraWeb.InferenceBackend.ONNX)
   const vectorModelData = aniraWeb.VectorModelData([modelData])

   // Nested-int64 vector: nested JS array, auto-converted.
   const shapes = aniraWeb.TensorShapeList([[1, 2, 512], [1]])

VectorBase
----------

.. js:autoclass:: Vectors.VectorBase
   :short-name:
   :members: size

   The abstract base class. Every ``Vector*`` extends this and
   exposes the methods listed under "Common Surface" above.
