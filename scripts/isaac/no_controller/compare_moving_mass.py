"""Matched moving-mass pairs: left zero commands, right checkpoint or fixed command.

Usage (from the repo root, after the Isaac Lab setup in the root README):

    PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
        $ISAACLAB_PYTHON scripts/isaac/no_controller/compare_moving_mass.py --cmd .2 0 0 --viz kit --plot
No auto-reset: each pair receives identical disturbances for the full preview.
"""
import argparse
from pathlib import Path
from isaaclab.app import add_launcher_args
from hybrid_control.sim_interfaces.isaac.moving_mass.demo import add_demo_args, validate_args, run_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_demo_args(parser, Path(__file__).resolve().parents[3]/'out', comparison=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    validate_args(parser, args, comparison=True)
    run_demo(args, comparison=True)


if __name__ == '__main__':
    main()
