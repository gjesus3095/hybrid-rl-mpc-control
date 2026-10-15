"""Visualize randomized moving-mass platforms, scripted probes, or episode splines.

Usage (from the repo root, after the Isaac Lab setup in the root README):

    PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
        $ISAACLAB_PYTHON scripts/isaac/no_controller/moving_mass_dr_demo.py --viz kit --plot
"""
import argparse
from pathlib import Path
from isaaclab.app import add_launcher_args
from hybrid_control.sim_interfaces.isaac.moving_mass.demo import add_demo_args, validate_args, run_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_demo_args(parser, Path(__file__).resolve().parents[3]/'out')
    add_launcher_args(parser)
    args = parser.parse_args()
    validate_args(parser, args)
    run_demo(args)


if __name__ == '__main__':
    main()
