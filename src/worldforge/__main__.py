"""Allow ``python -m worldforge``."""

from worldforge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
