"""
Allow smoke test to be run as a module: python -m xhalo.halolink
"""

from .smoketest import main

if __name__ == "__main__":
    main()
