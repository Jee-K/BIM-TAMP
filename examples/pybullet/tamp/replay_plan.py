#!/usr/bin/env python3
"""Replay stored trajectories with attachment updates and collision checks.

No replanning, placement snapping, or physics integration between assigned poses.
Unsafe plans stop at the first collision. --audit collects failures headlessly.
"""
from audited_replay import main

if __name__ == '__main__':
    raise SystemExit(main())
