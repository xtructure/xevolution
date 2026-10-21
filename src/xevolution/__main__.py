"""Expose the installed command as ``python -m xevolution``."""

from xevolution.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
