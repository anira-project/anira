.. _usage-tensors:

Runtime tensors
===============

The unit of data between a host and anira at run time is ``anira_tensor`` (``anira/abi/tensor.h``,
:cpp:struct:`anira::Tensor` in C++): a descriptor of memory, never the memory itself, 216 bytes,
trivially copyable and laid out the same on wasm32, LP64 and LLP64.

.. code-block:: c

    #include <anira/abi/tensor.h>

    float samples[2 * 512];
    const int64_t shape[2] = { 2, 512 };                     /* [channels, samples] */
    anira_tensor t;                                          /* caller memory: the stack is fine */
    anira_tensor_init_host(&t, samples, ANIRA_DTYPE_F32, 2, shape);
    float* first = anira_tensor_data_f32(&t);                /* NULL for a device or a non-float tensor */
    size_t count = anira_tensor_num_elements(&t);            /* 1024 */
    size_t frames = anira_tensor_extent(&t, 1);              /* 512; 0 for an axis at or above ndim */

    float left[512], right[512];
    float* channels[2] = { left, right };                    /* an audio host's channel pointers */
    anira_tensor planar;
    anira_tensor_init_host_planar(&planar, channels, 2, ANIRA_DTYPE_F32, 2, shape);   /* no cast */
    float* second = anira_tensor_plane(&planar, 1, ANIRA_DTYPE_F32);                  /* right */

    anira_error err = ANIRA_ERROR_INIT;
    anira_tensor from_numpy;                                 /* managed: a DLManagedTensorVersioned* */
    if (ANIRA_FAILED(anira_tensor_init_dlpack(&from_numpy, managed, &err))) { return fail(&err); }
    /* ... off the driver thread, once no longer needed: */
    if (from_numpy.release != NULL) { from_numpy.release(&from_numpy); }   /* the producer's deleter, once */

The record: ``domain`` (where the bytes live), ``dtype``, ``ndim``, ``flags``, ``shape`` and ``strides``
(eight ``int64_t`` each; strides in elements, all-zero meaning packed row-major), ``byte_offset``, the
memory ``handle`` (a 24-byte union with one arm per domain: ``host``, ``cuda``, ``gl``, ``vk``,
``opaque``, ``mtl``, ``iosurface``, ``wgpu``, ``dmabuf``, ``ahb``, ``d3d12``, ``planes``, and
``raw[3]``), the producer's ``manager_ctx`` and ``release``, and ``acquire``, the fence after which
the data is valid (an ``anira_sync_token``; ``ANIRA_SYNC_NONE``: already visible). The three records
are Tier 1: no ``struct_size``, the ABI major is their version, their layout is committed in
``abi/layout-<major>.txt`` and mirrored in ``web/src/abi/layout.ts``; ``anira_sizeof(ANIRA_STRUCT_TENSOR)``
answers an allocator that cannot see the header (0 for an id this build does not know). The Hard and
Static entries take host tensors; only
host memory is read in this pre-release.

Factories and reads
-------------------

- ``anira_tensor_init_host`` and ``_init_pinned`` describe host memory; ``_init_cuda``,
  ``_init_gl_buffer``, ``_init_vulkan``, ``_init_opaque_fd``, ``_init_wgpu_buffer`` and
  ``_init_dmabuf`` device memory, each ending in ``dtype, ndim, shape``. They zero all 216 bytes and
  write the domain, the dtype, the rank, the extents, their handle arm and the fence: a borrowed,
  packed tensor until the producer sets more. A NULL data pointer is accepted (a field fill never
  looks at the memory; the Hard entries refuse NULL memory per call). They are ``[thread-safe]``,
  ``[callback-safe]`` and ``ANIRA_NONBLOCKING`` field fills, legal on a render thread and in a
  processor callback.
- They return nothing: a dtype of 0, a rank above ``ANIRA_MAX_RANK``, a NULL shape with a rank above 0
  or a negative extent leave the record all-zero, which reads as dtype 0. Every argument is read
  before the record is zeroed, so ``anira_tensor_init_host(&t, p, t.dtype, t.ndim, t.shape)``
  re-points a tensor. The previous content is overwritten, never released: reset an owning
  ``acquire`` and call a ``release`` first.
- A NULL event, sync or fence, a timeline semaphore of 0 and a negative ``sync_fd`` leave
  ``ANIRA_SYNC_NONE``: anira never fabricates a fence (a WebGPU producer whose work is queued passes
  ``ANIRA_SYNC_QUEUE_ORDERED``). A token passed by pointer is copied into ``acquire``, which is the
  hand-off: the caller does not reset its source, unless the call was refused, when the token was
  not copied and stays the caller's. The six device factories are field fills only in this pre-release: no adapter consumes
  their arms.
- ``anira_tensor_init_dlpack(&t, dl_managed_tensor_versioned, &err)`` is the one factory that can fail
  with a message, the one ``[main-thread]`` one. It takes a DLPack 1.x ``DLManagedTensorVersioned*`` as
  ``void*``, maps the CPU to ``ANIRA_DOMAIN_HOST`` and CUDA host memory to ``ANIRA_DOMAIN_HOST_PINNED``,
  takes the ``DLDataType`` as the ``anira_dtype`` it already is (codes 0 to 6), copies shape, strides
  (NULL: packed), pointer and offset, and maps the read-only flag to ``ANIRA_TENSOR_READ_ONLY``. Another
  device, a code above 6, a major other than 1, and all-zero strides over more than one element are
  ``ANIRA_ERROR_NOT_SUPPORTED``; a rank outside 0 to 8, a NULL shape, a negative extent and a dtype of 0
  bits or lanes ``ANIRA_ERROR_INVALID_ARGUMENT``. A refused call leaves the record untouched and never
  calls the deleter; a NULL deleter gives a borrowed tensor (``release`` stays NULL).
- ``anira_tensor_data_f32(&t)`` returns the first element (``handle.host.ptr`` plus ``byte_offset``)
  of a host or pinned float32 tensor, and NULL for every other domain or dtype and for a NULL base
  pointer; ``anira_tensor_data(&t, dtype)`` is the same read for any element type, NULL unless
  ``dtype`` is the tensor's own. Neither looks at the strides. ``anira_tensor_num_elements`` (1 at
  rank 0, 0 with a zero extent and for the all-zero record) and ``anira_tensor_extent`` are
  ``size_t`` and answer 0 for what is no count. These reads, ``anira_tensor_plane`` and
  ``anira_sizeof`` are the whole read surface, ``[thread-safe]``, ``[callback-safe]`` and
  ``ANIRA_NONBLOCKING``: legal in a processor callback.

**Planar host memory.** ``anira_tensor_init_host_planar(&t, planes, count, dtype, ndim, shape)``
describes one pointer per channel without copying: axis 0 is the plane axis (a rank of 1 or more),
``count`` must equal ``shape[0]`` (else, and at rank 0, the record is left all-zero),
``ANIRA_TENSOR_PLANAR`` is set and ``handle.planes`` holds the pointer array, borrowed
like the memory. ``strides[0]`` is ignored; the strides from axis 1 and ``byte_offset`` apply inside
each plane. The array travels as ``const void*``, so ``float**`` and ``const float* const*`` both
convert without a cast (under clang-tidy's ``bugprone-multi-level-implicit-pointer-conversion``,
write ``static_cast<const void*>(channels)`` or use the C++ spelling). OR ``ANIRA_TENSOR_READ_ONLY``
into ``flags`` over read-only planes, and assign ``ANIRA_DOMAIN_HOST_PINNED`` over page-locked ones.
``anira_tensor_plane(&t, i, dtype)`` reads one plane (NULL for a one-block tensor, a plane at or
above ``handle.planes.count``, another domain or another dtype); the two data reads return NULL for
a planar tensor. An entry accepts a planar tensor only where its documentation says so (where it
copies a host block); every other consumer, and every domain but the two host ones, refuses the
flag with ``ANIRA_ERROR_NOT_SUPPORTED``, and anira
never hands a planar tensor to a processor, an engine or JavaScript.

Ownership, release and sync tokens
----------------------------------

- ``release == NULL`` means borrowed: the memory stays the caller's and must stay valid until the call
  returns (Hard) or the ticket is terminal (Async). A non-NULL ``release`` frees what the producer attached through
  ``manager_ctx``; anira would call it exactly once per submitted copy, never on the driver thread
  (it may block and free), but no entry takes ownership in this pre-release, so the holder calls it
  once itself. A DLPack tensor's ``release`` disarms the record before it calls the deleter.

.. warning::
    A DLPack deleter is not real-time safe (it frees memory; a Python producer's takes the
    interpreter lock): never call ``release`` on the driver thread. The disarming guards one record,
    not its bitwise copies: release one copy only.

- ``ANIRA_SYNC_SYNC_FILE_FD`` and ``ANIRA_SYNC_OPAQUE_FD_SEMAPHORE`` own their file descriptor
  (``u.fd``, 0 or more; on Windows an NT handle above 0). Every hand-off is a transfer: never close an fd you
  handed to a tensor (``anira_tensor_init_dmabuf``'s ``sync_fd`` belongs to the record from the call
  on, and stays yours when the call is refused). ``anira_sync_token_reset(&token)`` closes an owned fd
  and zeroes the token (kind ``ANIRA_SYNC_NONE``: a second reset closes nothing);
  ``anira_sync_token_dup(&token, &out)`` makes a token that outlives its source, a new close-on-exec
  descriptor for the two owning kinds and a plain copy of every other kind
  (``ANIRA_ERROR_INVALID_ARGUMENT`` for a NULL or aliasing argument or a closed fd,
  ``ANIRA_ERROR_OUT_OF_MEMORY`` out of descriptors; ``*out`` untouched on failure). Both are ``[thread-safe, !audio-thread]``. After a
  reset test ``kind``, never the payload. Every other kind (a CUDA event, a Vulkan timeline semaphore,
  a ``GLsync``, a Metal shared event, a D3D12 fence) is a non-owning handle. On WebAssembly no kind
  owns an object, and ``dup`` of an owning kind with an fd is ``ANIRA_ERROR_NOT_SUPPORTED``.

The draft folder and compiling the headers
------------------------------------------

``anira/abi/draft/tensor_platform.h`` declares the factories of the arms that have no measured edge
yet: ``anira_tensor_init_metal``, ``_init_iosurface``, ``_init_ahardwarebuffer`` and ``_init_d3d12``,
field fills on every platform (``shared_event`` and ``fence`` are tokens, copied like the WebGPU
fence; ``fence_fd`` follows the ``sync_fd`` rule). Their arms are frozen with ``anira_memory_handle``; the factories are
outside the ABI promise (``abi/symbols-draft.txt``) until the platform is measured, and promotion
never renames them. ``<anira/anira.h>`` does not include the draft folder; ``<anira/anira_all.h>``
does.

Every pointer of the three records sits in an ``ANIRA_PTR(T, name)`` slot, an anonymous union with a
``uint64_t name_bits``, 8 bytes on every target. So zero a record before filling it by hand (the
factories do), since on a 32-bit target a slot's high half is otherwise undefined. MSVC at ``/W4``
reports C4201 (nameless struct/union) for every slot: compile with ``/wd4201``. The arms are unnamed
struct types *with* a member name (``t.handle.cuda.device``), valid ISO C++17 under ``-pedantic``. The
layout is the natural alignment of wasm32, LP64, LLP64, 32-bit ARM and 32-bit MSVC; it does not hold
on the i386 System V ABI, which anira does not target.

In C++
------

``anira::Tensor`` and ``anira::SyncToken`` are these structs with names on them (a ``Tensor*`` is an
``anira_tensor*``): ``Tensor::from_host(data, dtype, shape)`` and ``from_pinned``, ``from_cuda``,
``from_gl_buffer``, ``from_vulkan``, ``from_opaque_fd``, ``from_wgpu_buffer``, ``from_dmabuf`` are the
``noexcept`` field fills; ``Tensor::from_host_planar<float>(planes, shape)`` and ``plane<T>(i)`` are
the typed planar spellings (the element type names the dtype, a const element sets
``ANIRA_TENSOR_READ_ONLY``); ``from_dlpack`` throws ``anira::Error``; ``data_f32()``, ``data(dtype)``,
``num_elements()`` and ``extent(axis)`` are the reads, ``SyncToken::reset()`` and ``dup()`` the two
token calls. The draft factories have no C++ spelling: call ``anira_tensor_init_metal(&tensor, ...)``
on a ``Tensor``.
