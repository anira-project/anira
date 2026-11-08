Anira Documentation
===================

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :end-before: ## Documentation

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- Features -->

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Start

    about
    getting_started
    examples

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: User guide

    usage
    custom_preprocessing
    custom_engines
    logging
    latency
    benchmarking

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Web

    web-api/index

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Reference

    api/index

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Integration

    plugins
    troubleshooting

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Project

    architecture
    migration
    changelog
    contributing

Next Steps
~~~~~~~~~~

- New to anira: :doc:`getting_started`, then the :doc:`usage`.
- A processor or an engine of your own: :doc:`custom_preprocessing`, :doc:`custom_engines`.
- In the browser: :doc:`web-api/index`.
- In a plugin: :doc:`plugins`. Something wrong: :doc:`troubleshooting`.
- Coming from anira 2.x: :doc:`migration`.
