#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from hindcast_t2m.cli import main
import hindcast_t2m.pipeline as p
print("USING PIPELINE:", p.__file__)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
