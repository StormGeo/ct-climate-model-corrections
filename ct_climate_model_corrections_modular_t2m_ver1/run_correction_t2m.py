#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from hindcast_t2m.correction.cli import main


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
