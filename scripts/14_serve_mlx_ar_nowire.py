#!/usr/bin/env python3
from __future__ import annotations

import mlx.core as mx
import mlx_lm.server as server


def no_set_wired_limit(limit):
    return limit


mx.set_wired_limit = no_set_wired_limit
server.mx.set_wired_limit = no_set_wired_limit


if __name__ == "__main__":
    server.main()
