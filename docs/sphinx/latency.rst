Latency
=======

Overview
--------

Latency is a critical factor in real-time audio processing applications. anira computes the latency of every session in closed form, together with the number of inference slots the session allocates and the sizes of its ring buffers. The calculation lives in :cpp:class:`anira::LatencyCalculator`; :cpp:class:`anira::SessionElement` applies its result in ``prepare()``.

The model behind the formulas is the scheduler's actual worst case:

* every inference takes exactly ``max_inference_time``,
* at most ``num_parallel_processors`` inferences run at once, in submission order,
* an inference is submitted by the host callback that completes its hop of input,
* results are collected once per callback, after the blocking wait (``blocking_ratio``),
* the host pushes and pops a sample only once a whole one has accumulated in its block, so that a block may be a fractional number of samples on a stream (see the note below).

Quantities
----------

For a session with reference stream size :math:`R` (see the usage guide, section 4.1), host buffer size :math:`B` and host sample rate :math:`f_s`, both stated in samples of the reference stream, the calculator derives:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Symbol
     - Value
     - Meaning
   * - :math:`\rho = p / q`
     - :math:`B / R`
     - the host block measured in hops, as a reduced fraction
   * - :math:`\kappa`
     - :math:`T f_s / (1000 R)`
     - the maximum inference time :math:`T` (ms) measured in hop periods
   * - :math:`\tau`
     - :math:`\kappa / \rho`
     - the inference time measured in host blocks
   * - :math:`\beta`
     - ``blocking_ratio``
     - the fraction of a host block the driving thread waits for results
   * - :math:`n`
     - ``num_parallel_processors``
     - the parallelism of the session
   * - :math:`H`
     - largest streamable hop
     - the stream on which the host block is finest; the unit of the ``allow_smaller_buffers`` grid
   * - :math:`d_m`
     - :math:`\max(0, \lceil (m+1)\tau - \beta \rceil)`
     - callbacks after submission at which the :math:`(m+1)`-th batch of :math:`n` inferences submitted together is collected

Buffer adaptation
-----------------

Repackaging a stream from host blocks of :math:`b` samples into model blocks of :math:`P` samples requires a delay. Rath and Geier [RG26]_ prove that the minimum delay for constant block sizes is

.. math::

   \Delta = P - \gcd(b, P),

which replaces the PortAudio-style loop over every multiple of :math:`b` below :math:`\mathrm{lcm}(b, P)` that anira used before. In hop units the adaptation is :math:`1 - 1/q` for every stream at once, because :math:`\gcd(\rho P, P) = P / q` whenever the block :math:`\rho P` is a whole number of samples. If the host block may vary in size (``allow_smaller_buffers``) or is a fractional number of samples on a stream, the worst case is :math:`P - 1` samples [RG26]_, section 5.

Inference-caused latency
------------------------

The receive ring of an output never runs dry if and only if its zero priming :math:`L` (in hops) covers, at every callback :math:`k`, the samples the host has popped minus the results that have been collected:

.. math::

   L \;\ge\; (k+1)\rho - C(k) \quad \text{for all } k,

where :math:`C(k)` is the number of inferences collected by callback :math:`k`. Inference :math:`j` is submitted at callback :math:`a_j = \lceil (j+1)/\rho \rceil - 1` (the last host block overlapping model block :math:`j`, the observation at the heart of [RG26]_), and with :math:`n` processors in submission order it finishes at :math:`F_j = \max(a_j, F_{j-n}) + \tau`. Unrolling the recursion gives

.. math::

   C(k) = \min_{m \ge 0} \left[ \lfloor \rho\,(k + 1 - d_m) \rfloor + m n \right],

and since :math:`(k+1)\rho - \lfloor \rho (k+1-d_m) \rfloor = \rho\, d_m + \mathrm{frac}(\rho (k + 1 - d_m))`, whose largest value over :math:`k` is :math:`\rho\, d_m + (q-1)/q`, the latency in hops is

.. math::

   \Lambda \;=\; \frac{q - 1}{q} \;+\; \max_{m \ge 0} \left[ \rho\, d_m - m n \right].

The first term is the buffer adaptation, the second the inference queue. The maximum exists if and only if :math:`\kappa < n`: every hop of audio brings :math:`\kappa` hop periods of inference work for :math:`n` processors, and beyond that the queue grows without bound (:cpp:func:`anira::LatencyCalculator::is_feasible` is false, ``prepare()`` logs a warning, and the values describe one host block processed by an idle pool). Only batches with :math:`m < \rho / (n - \kappa)` can contribute, so the search is finite; in the common case :math:`\rho \lceil \tau \rceil \le n` the maximum is the first term, :math:`\rho \lceil \tau - \beta \rceil`.

The latency of output tensor :math:`i` in its own samples is :math:`P_i \Lambda`. It is the same number of hops for every output, and it is rounded down because the host pops whole samples only (:cpp:func:`anira::LatencyCalculator::get_output_latencies` returns the unrounded value).

.. note::
    When the host buffer size is a fractional (floating-point) value on a stream, the host and that stream exchange samples at a non-integer ratio. anira assumes the worst case: a sample is pushed to the :cpp:class:`anira::InferenceHandler` only when the host buffer accumulates a full sample, and popped under the same rule. For example, if the host buffer size is 0.25 samples of a control-rate stream, the :cpp:class:`anira::InferenceHandler` receives one sample every four host buffer cycles, and the latency is calculated as if the sample is delivered during the fourth cycle. If your system always sends the sample at the first host buffer cycle, a lower latency is possible; consider configuring :cpp:class:`anira::InferenceHandler` with a custom latency value.

Inference slots
---------------

The same recursion bounds the number of inferences that are submitted but not yet collected right after the submissions of a callback, which is the number of thread-safe structures a session allocates:

.. math::

   S \;=\; \max_{m \ge 0} \left[ \lceil (d_m + 1)\rho \rceil - m n \right].

With a blocking ratio that covers the inference time this is 1: the result is collected in the callback that submitted it.

Adaptive buffer handling
------------------------

For hosts that may use smaller buffers than the configured maximum (``allow_smaller_buffers``), the host block may be any whole number :math:`j` of samples of the finest stream, :math:`1 \le j \le \lfloor \rho H \rfloor`, i.e. :math:`\rho' = j / H` hops with :math:`\tau' = \kappa / \rho'`. The calculator takes the worst case over that grid: the flexible-host adaptation :math:`(H-1)/H` plus the maximum over :math:`j` of the inference term and of the slot count. Both are increasing in :math:`j` on every interval where :math:`\lceil (m+1)\tau' - \beta \rceil` is a constant :math:`c`, so the maximum is attained at the last grid point of each interval, :math:`j_c = \lceil (m+1)\kappa H / (c - 1 + \beta) \rceil - 1`, and the remaining intervals are bounded by :math:`(m+1)\kappa\, c / (c - 1 + \beta)`. Only these breakpoints are evaluated; the former countdown over every block size, which stalled for blocks above :math:`2^{24}` samples, is gone.

Note that a smaller host block can require *more* latency and more slots than the maximum block: a block slightly shorter than one inference time means that results arrive two callbacks after their submission instead of one.

Internal model latency
----------------------

Additional latency inherent to the model itself, such as look-ahead requirements or internal buffering (``internal_model_latency``), is added to every streamable output after the calculation above.

Latency synchronization
-----------------------

When several output tensors are present, the integer latencies are raised to a common whole number of hops, :math:`\lceil \max_i L_i / P_i \rceil \cdot P_i`, so that the outputs stay coherent.

The latency vector returned by :cpp:func:`anira::InferenceHandler::get_latency_vector` is index-aligned with the output tensor list. Non-streamable outputs (``postprocess_output_size == 0``) carry no stream latency and always report ``0``.

Ring buffer sizes
-----------------

A send ring holds one host block plus the largest leftover the adaptation can leave behind, :math:`\lceil b \rceil + P - \gcd(\lceil b \rceil, P)` for a constant whole-sample block and :math:`\lceil b \rceil + P - 1` for a fractional or variable one, plus the history a receptive-field model peeks at. A receive ring holds one result per inference slot plus the latency priming, :math:`S P_i + L_i`.

These calculations count buffer sizes in samples of the *reference stream* selected by the :cpp:struct:`anira::HostConfig` — a streamable input for effects and analysers, the streamable output for a generator model with no streamable input (see the usage guide, section 4.1).

Output Behavior
---------------

The final latency value represents the total delay (in samples) between when input data enters the system and when the processed output data becomes available.

.. important::
    Before the first valid output is produced, the :cpp:func:`anira::InferenceHandler::process` and :cpp:func:`anira::InferenceHandler::pop_data` methods will return zeroed data. This ensures real-time audio processing without introducing unexpected delays or artifacts in the output signal.

.. [RG26] Matthias Rath and Matthias Geier, "Minimum required delay for realtime block size adaptation in digital audio signal processing", Proceedings of the 20th Linux Audio Conference (LAC-26), Maynooth, 2026. `hal-05697688 <https://hal.science/hal-05697688v1>`_
