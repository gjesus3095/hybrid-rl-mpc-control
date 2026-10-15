"""Assert physical mass/CoM, spline tracking, reset isolation and episode behavior.

Usage (from the repo root, after the Isaac Lab setup in the root README):

    PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src     \
        $ISAACLAB_PYTHON scripts/isaac/no_controller/check_moving_mass_env.py --num_envs 4
"""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
add_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 2:
    parser.error('at least two environments are needed for reset isolation')

from dataclasses import replace
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from hybrid_control.rl.policy_no_controller.config import create_environment
from hybrid_control.rl.policy_no_controller.env import StabilizationEnvCfg
from hybrid_control.sim_interfaces.isaac.moving_mass.diagnostics import analytical_com
from hybrid_control.sim_interfaces.isaac.moving_mass.params import MovingMassRandomization, PlatformRandomization


def main():
    # Fixed, easy dynamics isolate implementation correctness from learning difficulty.
    dr = MovingMassRandomization(platform=PlatformRandomization(length=(.4, .4), width=(.4, .4),
        base_mass=(30, 30), base_com_z=(.1, .1)), terminal_mass=(3, 3),
        carriage_mass=(.5, .5), gimbal_mass=(.25, .25), mast_mass=(.25, .25))
    cfg = replace(StabilizationEnvCfg(), domain_randomization=dr)
    sim_cfg = sim_utils.SimulationCfg(dt=cfg.physics_dt, device=args.device)
    with launch_simulation(sim_cfg, args):
        env = create_environment(SimulationContext(sim_cfg), cfg, args.num_envs, 2.5, args.seed)
        base, robot = env.env, env.env.robot
        try:
            assert set(robot.joint_names) == {'lift', 'roll', 'pitch'}
            assert not any('panda' in n for n in robot.body_names)
            from pxr import UsdPhysics
            stage = base.sim.stage
            for name in robot.body_names:
                assert stage.GetPrimAtPath(f'/World/envs/env_0/Robot/{name}').HasAPI(UsdPhysics.RigidBodyAPI)
                assert stage.GetPrimAtPath(f'/World/envs/env_0/Robot/{name}/visual').HasAPI(UsdPhysics.CollisionAPI)
            for name, axis in (('roll', 'X'), ('pitch', 'Y')):
                joint = UsdPhysics.RevoluteJoint.Get(stage, f'/World/envs/env_0/Robot/{name}')
                assert joint.GetAxisAttr().Get() == axis
                assert joint.GetLowerLimitAttr().Get() == -90 and joint.GetUpperLimitAttr().Get() == 90
            expected_mass = 34.0
            assert torch.allclose(base.mass_total, torch.full_like(base.mass_total, expected_mass), atol=1e-5)
            # Upright mast: all horizontal CoM contributions vanish.
            m = cfg.mechanism
            pivot_z = m.platform_height + .08 + m.initial[2]
            upper_moment = ((.5 + .25)*pivot_z
                            + .25*(pivot_z + m.mast_length/2)
                            + 3*(pivot_z + m.mast_length + .04))
            expected = torch.tensor([0, 0, (30*.1 + upper_moment)/expected_mass + .01], device=env.device)
            error = (base.com_total-base.origins-expected).abs().max()
            assert error < 1e-4, f'initial analytical CoM mismatch {error}'
            print(f'[CHECK] bodies={robot.body_names}, mass={expected_mass}, CoM error={float(error):.3g}')

            zero = torch.zeros(env.num_envs, 3, device=env.device)
            env.auto_reset = False
            max_error = torch.zeros(3, device=env.device)
            max_speed = 0.0
            for _ in range(env.max_episode_length):
                obs, reward, done, extras = env.step(zero)
                assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['critic']).all()
                assert torch.isfinite(reward).all()
                q, _, _ = env.traj.evaluate(env.episode_length_buf.float()*env.dt)
                max_error = torch.maximum(max_error, (robot.data.joint_pos.torch[:, base.joint_ids]-q).abs().amax(0))
                max_speed = max(max_speed, float(robot.data.body_com_lin_vel_w.torch[:, base.base_id, :2].norm(dim=-1).max()))
                expected = analytical_com(robot.data.joint_pos.torch[:, base.joint_ids],
                    robot.data.body_pos_w.torch[:, base.base_id], robot.data.body_quat_w.torch[:, base.base_id],
                    base.params, cfg.mechanism)
                assert torch.allclose(base.com_total, expected, atol=1e-4), 'moving analytical CoM mismatch'
                assert not extras['failed'].any(), 'easy baseline tipped'
            assert done.all() and extras['time_outs'].all()
            assert (max_error[:2] < .05).all() and max_error[2] < .02, max_error
            assert max_speed > 1e-6, 'moving mechanism caused no measurable base response'
            print(f'[CHECK] full episode: tracking errors {max_error.tolist()}, peak base speed {max_speed:.6f} m/s')

            before = (robot.data.joint_pos.torch.clone(), base.base_pos.clone(), env.traj.control.clone())
            ids = torch.arange(0, env.num_envs, 2, device=env.device)
            keep = torch.arange(1, env.num_envs, 2, device=env.device)
            env._reset_idx(ids)
            assert torch.equal(before[0][keep], robot.data.joint_pos.torch[keep])
            assert torch.equal(before[1][keep], base.base_pos[keep])
            assert torch.equal(before[2][keep], env.traj.control[keep])
            assert (env.episode_length_buf[ids] == 0).all()
            print('[CHECK] partial reset leaves neighboring states, trajectories and clocks unchanged')

            env.auto_reset = True
            env._reset_idx(torch.arange(env.num_envs, device=env.device))
            # Force physical tip on the timeout step: failure must win over bootstrap.
            env.episode_length_buf[:] = env.max_episode_length-1
            pose = robot.data.root_pose_w.torch.clone()
            pose[:, 3:] = torch.tensor([.3428978, 0, 0, .9393727], device=env.device)  # roll 0.7
            robot.write_root_pose_to_sim_index(root_pose=pose)
            _, reward, done, extras = env.step(zero)
            assert done.all() and extras['failed'].all() and not extras['time_outs'].any()
            assert not env.episode_length_buf.any()
            print('[CHECK] tip wins over simultaneous timeout; automatic reset succeeds')

            # Overlapping slots must not physically push one another.
            pose = robot.data.root_pose_w.torch.clone()
            pose[:, :3] = pose[0, :3].clone()
            robot.write_root_pose_to_sim_index(root_pose=pose)
            base._update_state()
            initial = torch.tensor(cfg.mechanism.initial, device=env.device).expand(env.num_envs, -1)
            for _ in range(30):
                base.step(initial, zero, zero)
            xy = robot.data.body_pos_w.torch[:, base.base_id, :2]
            assert (xy-xy[0]).abs().max() < 1e-4, 'environment collision filtering failed'
            assert not base.tipped.any()
            print('[CHECK] overlapping environment slots remain physically isolated')

            # Drop the whole mechanism sideways. Every body's box must respect the ground.
            env._reset_idx(torch.arange(env.num_envs, device=env.device))
            pose = robot.data.root_pose_w.torch.clone()
            pose[:, 2] = 1.5
            pose[:, 3:] = torch.tensor([.8660254, 0, 0, .5], device=env.device)  # roll 120 degrees
            robot.write_root_pose_to_sim_index(root_pose=pose)
            base._update_state()
            m = cfg.mechanism
            shapes = {
                'base_link': ((0, 0, m.platform_height/2), (.4, .4, m.platform_height)),
                'carriage': ((0, 0, 0), (.06, .06, .08)),
                'roll_hub': ((0, 0, 0), (.06, .06, .06)),
                'mast': ((0, 0, m.mast_length/2), (.035, .035, m.mast_length)),
                'terminal': ((0, 0, 0), (.08, .08, .08)),
            }
            upper_contact = False
            for _ in range(360):
                base.step(initial, zero, zero)
                for name, (centre, size) in shapes.items():
                    body_id = robot.find_bodies(name)[0][0]
                    quat = robot.data.body_quat_w.torch[:, body_id]
                    x, y, z, w = quat.unbind(-1)
                    axis = torch.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1)
                    low = (robot.data.body_pos_w.torch[:, body_id, 2]
                           + (axis*axis.new_tensor(centre)).sum(-1)
                           - (axis.abs()*axis.new_tensor(size)/2).sum(-1))
                    assert (low > -.025).all(), f'{name} penetrated ground: {low}'
                    if name in ('mast', 'terminal'):
                        upper_contact |= bool((low < .01).any())
            assert upper_contact, 'drop did not exercise upper-body ground contact'
            print('[CHECK] sideways drop: upper bodies contact the ground without passing through it')
        finally:
            env.close()


if __name__ == '__main__':
    main()
