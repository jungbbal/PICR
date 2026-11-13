#!/usr/bin/env python3
"""Train PICR with the cdelan trainer.

    source env.sh
    python3 scripts/train.py --config configs/picr.yaml

The checkpoint of the last epoch is ``runs/<run_name>/last.pt`` (the released checkpoint is epoch 40 ``last.pt``).
"""
from cdelan.training import train as T

if __name__ == "__main__":
    T.main()
