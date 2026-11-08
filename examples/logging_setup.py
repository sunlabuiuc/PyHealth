"""Control where PyHealth's log messages go.

PyHealth prints progress messages to standard output through its own handler
on the ``pyhealth`` logger. This script shows the three common setups:

1. The default: each message printed once, even when the root logger also
   has a handler (as in Jupyter and Colab, or after ``logging.basicConfig``).
2. Quieter output by raising the ``pyhealth`` logger's level.
3. Routing PyHealth's messages through your own root configuration instead.

Run it with ``python examples/logging_setup.py``; it needs no data.
"""

import logging
import sys

import pyhealth


def main() -> None:
    # A root handler, like the one Jupyter and Colab install.
    logging.basicConfig(
        level=logging.INFO, format="root: %(message)s", stream=sys.stdout
    )
    pyhealth_logger = logging.getLogger("pyhealth")
    datasets_logger = logging.getLogger("pyhealth.datasets")

    print("1. Default: printed once, by PyHealth's handler")
    datasets_logger.info("Loading tables...")

    print("\n2. Quieter: only warnings and errors")
    pyhealth_logger.setLevel(logging.WARNING)
    datasets_logger.info("This info message is hidden")
    datasets_logger.warning("This warning is shown")
    pyhealth_logger.setLevel(logging.INFO)

    print("\n3. Through your own root configuration (note the 'root:' prefix)")
    pyhealth_logger.removeHandler(pyhealth.handler)
    pyhealth_logger.propagate = True
    datasets_logger.info("Loading tables...")


if __name__ == "__main__":
    main()
