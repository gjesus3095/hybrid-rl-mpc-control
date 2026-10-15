"""Check physical mass randomization, fixed footprints, and per-slot spline resampling.

Example: $ISAACLAB_PYTHON scripts/isaac/no_controller/check_moving_mass_randomization.py --num_envs 8 --resets 5
"""
import argparse
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=8)
parser.add_argument('--resets', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--config', default=None)
add_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 2 or args.resets < 1:
    parser.error('--num_envs must be at least 2 and --resets must be positive')

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from hybrid_control.rl.policy_no_controller.config import create_environment, load_configuration
from hybrid_control.sim_interfaces.isaac.moving_mass.diagnostics import analytical_com


def main():
    cfg, _ = load_configuration(args.config)
    sim_cfg = sim_utils.SimulationCfg(dt=cfg.physics_dt, device=args.device)
    with launch_simulation(sim_cfg, args):
        env = create_environment(SimulationContext(sim_cfg), cfg, args.num_envs, 2.5, args.seed)
        base, robot = env.env, env.env.robot
        original_footprints = [p.footprint for p in base.params]
        ids = torch.arange(0, env.num_envs, 2, device=env.device)
        keep = torch.arange(1, env.num_envs, 2, device=env.device)
        observations = []
        changed_mass = changed_spline = False
        try:
            for iteration in range(args.resets):
                params_before = list(base.params)
                curves_before = env.traj.control.clone()
                joint_before = robot.data.joint_pos.torch.clone()
                root_before = robot.data.root_pose_w.torch.clone()
                masses_before = robot.data.body_mass.torch.clone()
                inertias_before = robot.data.body_inertia.torch.clone()
                env._reset_idx(ids)
                assert torch.equal(masses_before[keep], robot.data.body_mass.torch[keep])
                assert torch.equal(inertias_before[keep], robot.data.body_inertia.torch[keep])
                assert [p.footprint for p in base.params] == original_footprints
                assert torch.equal(curves_before[keep], env.traj.control[keep])
                assert torch.equal(joint_before[keep], robot.data.joint_pos.torch[keep])
                assert torch.equal(root_before[keep], robot.data.root_pose_w.torch[keep])
                assert all(base.params[i] == params_before[i] for i in keep.tolist())
                changed_spline |= not torch.equal(curves_before[ids], env.traj.control[ids])
                changed_mass |= any(base.params[i] != params_before[i] for i in ids.tolist())
                for i, p in enumerate(base.params):
                    platform = cfg.domain_randomization.platform
                    for value, bounds in ((p.footprint[0], platform.length), (p.footprint[1], platform.width),
                                          (p.base_mass, platform.base_mass), (p.base_com_z, platform.base_com_z),
                                          (p.terminal_mass, cfg.domain_randomization.terminal_mass)):
                        assert bounds[0] <= value <= bounds[1]
                    for name, body_id in base.mass_model.upper_ids.items():
                        value = getattr(p, name)
                        lo, hi = getattr(cfg.domain_randomization, name)
                        assert lo <= value <= hi
                        assert abs(float(robot.data.body_mass.torch[i, body_id])-value) < 1e-5
                        size = {'carriage_mass': (.06, .06, .08), 'gimbal_mass': (.06, .06, .06),
                                'mast_mass': (.035, .035, cfg.mechanism.mast_length)}[name]
                        x, y, z = size
                        expected_inertia = torch.diag(torch.tensor(
                            [value*(y*y+z*z)/12, value*(x*x+z*z)/12, value*(x*x+y*y)/12],
                            device=env.device)).flatten()
                        assert torch.allclose(robot.data.body_inertia.torch[i, body_id],
                                              expected_inertia, atol=1e-7, rtol=1e-4)
                    assert torch.allclose(env.param_vec[i, 5:8],
                        env.param_vec.new_tensor([p.carriage_mass, p.gimbal_mass, p.mast_mass]))
                    expected_mass = p.base_mass + p.carriage_mass + p.gimbal_mass + p.mast_mass + p.terminal_mass
                    assert abs(float(base.mass_total[i])-expected_mass) < 1e-4
                    assert abs(float(robot.data.body_mass.torch[i, base.base_id])-p.base_mass) < 1e-4
                    assert abs(float(robot.data.body_mass.torch[i, base.mass_model.terminal_id])-p.terminal_mass) < 1e-4
                    assert abs(float(env.param_vec[i, -1])-expected_mass) < 1e-4
                    if i in ids.tolist():
                        observations.append([*p.footprint, p.base_mass, p.base_com_z, p.terminal_mass, p.carriage_mass, p.gimbal_mass, p.mast_mass])
                expected_com = analytical_com(robot.data.joint_pos.torch[:, base.joint_ids],
                    robot.data.body_pos_w.torch[:, base.base_id], robot.data.body_quat_w.torch[:, base.base_id],
                    base.params, cfg.mechanism)
                assert torch.allclose(expected_com, base.com_total, atol=1e-4)
                for t in torch.linspace(0, env.traj.duration, 31, device=env.device):
                    q, qd, qdd = env.traj.evaluate(t.expand(env.num_envs))
                    assert (q >= env.traj.lower-1e-5).all() and (q <= env.traj.upper+1e-5).all()
                    assert (qd.abs() <= qd.new_tensor(cfg.trajectory.max_velocity)+1e-5).all()
                    assert (qdd.abs() <= qdd.new_tensor(cfg.trajectory.max_acceleration)+1e-5).all()
                for _ in range(4):
                    obs, reward, _, _ = env.step(torch.zeros(env.num_envs, 3, device=env.device))
                    assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
            dr = cfg.domain_randomization
            has_dynamic_range = any(lo < hi for lo, hi in (dr.platform.base_mass, dr.platform.base_com_z, dr.terminal_mass, dr.carriage_mass, dr.gimbal_mass, dr.mast_mass))
            if has_dynamic_range:
                assert changed_mass, 'nonconstant dynamics did not change across resets'
            if cfg.trajectory.difficulty > 0:
                assert changed_spline, 'nonconstant splines did not change across resets'
            data = torch.tensor(observations)
            for i, name in enumerate(('length_m', 'width_m', 'base_mass_kg', 'base_com_z_m', 'terminal_mass_kg', 'carriage_mass_kg', 'gimbal_mass_kg', 'mast_mass_kg')):
                print(f'[SAMPLES] {name}: min={data[:, i].min():.4f}, mean={data[:, i].mean():.4f}, max={data[:, i].max():.4f}')
            print(f'[PASS] {args.resets} partial resets: fixed footprints, simulator masses/CoM, parameter observations, '
                  'bounded fresh splines, and untouched neighboring slots')
        finally:
            env.close()


if __name__ == '__main__':
    main()
