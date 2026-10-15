"""Shared runtime for randomized visual demos and paired comparisons."""
import csv
import math
from pathlib import Path


def add_demo_args(parser, out_dir, comparison=False):
    parser.add_argument('--num_envs', type=int, default=4 if comparison else 8,
                        help='Number of matched pairs in comparison mode; otherwise number of environments.')
    parser.add_argument('--num_cols', type=int, default=4)
    parser.add_argument('--spacing', type=float, default=2.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', default=None, help='Moving-mass environment YAML.')
    parser.add_argument('--scenario', choices=('spline', 'extend', 'joint_sweep'), default='spline')
    parser.add_argument('--joint', choices=('roll', 'pitch', 'lift'), default='pitch')
    parser.add_argument('--duration', type=float, default=None, help='Spline preview duration; default: full episode.')
    parser.add_argument('--settle', type=float, default=1.0, help='Upright settling time for extend/joint_sweep [s].')
    parser.add_argument('--swing', type=float, default=4.0, help='One-way motion time for extend/joint_sweep [s].')
    parser.add_argument('--hold', type=float, default=3.0, help='Target hold time for extend/joint_sweep [s].')
    parser.add_argument('--cmd', nargs=3, type=float, default=(0, 0, 0), metavar=('VX', 'VY', 'WZ'),
                        help='Base command [m/s, m/s, rad/s]; comparison applies it only to the right column.')
    if comparison:
        parser.add_argument('--checkpoint', help='Optional trained policy for the right column instead of --cmd.')
    parser.add_argument('--video', action='store_true')
    stem = 'moving_mass_compare' if comparison else 'moving_mass_dr'
    parser.add_argument('--video_path', default=str(out_dir/f'{stem}.mp4'))
    parser.add_argument('--video_fps', type=int, default=30)
    parser.add_argument('--csv', default=str(out_dir/f'{stem}.csv'), help='State/command traces in CSV.')
    parser.add_argument('--plot', action='store_true', help='Save a summary PNG next to the CSV.')
    parser.add_argument('--no_markers', action='store_true')


def validate_args(parser, args, comparison=False):
    for name in ('num_envs', 'num_cols', 'spacing', 'swing', 'video_fps'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'--{name} must be positive and finite')
    for name in ('settle', 'hold'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f'--{name} must be non-negative and finite')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be positive and finite')
    if args.duration is not None and args.scenario != 'spline':
        parser.error('--duration only applies to spline previews; use --settle/--swing/--hold for probes')
    if not all(math.isfinite(x) for x in args.cmd):
        parser.error('--cmd must be finite')
    if comparison and args.checkpoint and any(args.cmd):
        parser.error('select either --checkpoint or a nonzero --cmd')
    if args.video:
        args.enable_cameras = True


def _observations(base, cfg, last_action):
    import torch
    from tensordict import TensorDict
    from hybrid_control.rl.policy_no_controller.observations import Snapshot, observe
    robot = base.robot
    params = torch.tensor([[*p.footprint, p.base_mass, p.base_com_z, p.terminal_mass, p.carriage_mass, p.gimbal_mass, p.mast_mass, float(base.mass_total[i])]
                           for i, p in enumerate(base.params)], device=base.sim.device)
    snap = Snapshot(yaw=base.yaw, roll=base.roll, pitch=base.pitch, centre_xy=base.base_pos[:, :2],
        lin_vel_w=robot.data.body_com_lin_vel_w.torch[:, base.base_id],
        ang_vel_w=robot.data.body_com_ang_vel_w.torch[:, base.base_id],
        com_total=base.com_total-base.origins, com_mechanism=base.com_mechanism-base.origins,
        joint_pos=robot.data.joint_pos.torch[:, base.joint_ids], joint_vel=robot.data.joint_vel.torch[:, base.joint_ids],
        half_size=torch.stack((base.half_length, base.half_width), -1), params=params, last_action=last_action)
    actor, critic = observe(snap, cfg.observation)
    return TensorDict({'actor': actor, 'critic': critic}, batch_size=[base.num_envs])


def run_demo(args, comparison=False):
    import torch
    from isaaclab.app import launch_simulation
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from hybrid_control.rl.policy_no_controller.config import load_configuration
    from hybrid_control.rl.policy_no_controller.trajectories import JointSplines
    from .diagnostics import analytical_com, protocol_targets
    from .env import MovingMassEnv
    from .viz import DemoView

    checkpoint = getattr(args, 'checkpoint', None)
    cfg, runner_cfg = load_configuration(args.config, checkpoint)
    if args.video_fps > 1/cfg.physics_dt:
        raise ValueError('video_fps cannot exceed the physics rate')
    if any(abs(x) > limit for x, limit in zip(args.cmd, cfg.max_base_speed)):
        raise ValueError('--cmd must lie within the configured base velocity limits')
    horizon = round(cfg.episode_length_s/(cfg.physics_dt*cfg.decimation))*cfg.physics_dt*cfg.decimation
    duration = (args.duration or horizon) if args.scenario == 'spline' else args.settle + args.swing + args.hold
    if args.scenario == 'joint_sweep':
        duration += args.swing + args.hold
    torch.manual_seed(args.seed)
    params = cfg.domain_randomization.sample_many(args.num_envs, args.seed)
    if comparison:
        params = [p for p in params for _ in range(2)]
    sim_cfg = sim_utils.SimulationCfg(dt=cfg.physics_dt, device=args.device)
    with launch_simulation(sim_cfg, args):
        sim = SimulationContext(sim_cfg)
        base = MovingMassEnv(params, sim, spacing=args.spacing, num_cols=2 if comparison else args.num_cols,
                             friction=cfg.friction, tau=cfg.base_time_constant, max_force=cfg.max_base_force,
                             mechanism=cfg.mechanism)
        view = DemoView(base, args)  # camera/marker prims must exist before reset
        base.start()
        device, n = sim.device, base.num_envs
        splines = JointSplines(n, device, horizon, cfg.mechanism, cfg.trajectory, args.seed)
        if comparison:
            splines.control[1::2] = splines.control[::2].clone()
            assert all(base.params[i] == base.params[i+1] for i in range(0, n, 2))
            assert torch.equal(splines.control[::2], splines.control[1::2])
            print('[CHECK] paired masses, footprints, initial states and complete spline coefficients match')
        initial = torch.tensor(cfg.mechanism.initial, device=device).expand(n, -1)
        target = initial.clone()
        target[:, 1], target[:, 2] = cfg.mechanism.upper[1], cfg.mechanism.upper[2]
        if args.scenario == 'joint_sweep':
            index = ('roll', 'pitch', 'lift').index(args.joint)
            target = initial.clone()
            target[:, index] = cfg.mechanism.lower[index] if index == 1 else cfg.mechanism.upper[index]
        if args.scenario != 'spline':
            # Quintic blend max speed = 1.875 * excursion / swing.
            # Explicit probes may exceed the random spline limits, but not drive speed limits.
            if ((target-initial).abs().amax(0)*1.875/args.swing > torch.tensor(cfg.mechanism.velocity_limit, device=device)).any():
                raise ValueError('probe swing is too fast for actuator velocity limits; increase --swing')
        last_action = torch.zeros(n, 3, device=device)
        max_vel = torch.tensor(cfg.max_base_speed, device=device)
        fixed = torch.tensor(args.cmd, device=device).expand(n, -1).clone()
        if comparison:
            fixed[::2] = 0
        labels = [f'pair {i//2} / {"zero" if i%2 == 0 else "policy" if checkpoint else "command"}'
                  if comparison else f'env {i}' for i in range(n)]
        for i, p in enumerate(base.params):
            print(f'[INFO] {labels[i]}: {p.footprint[0]:.3f}x{p.footprint[1]:.3f} m, '
                  f'base={p.base_mass:.2f} kg, terminal={p.terminal_mass:.2f} kg, carriage/hub/mast={p.carriage_mass:.2f}/{p.gimbal_mass:.2f}/{p.mast_mass:.2f} kg')
        policy = None
        if checkpoint:
            from rsl_rl.runners import OnPolicyRunner
            # The runner needs only initial observations and environment metadata to load inference weights.
            class InferenceEnv:
                num_envs, num_actions = n, 3
                def get_observations(self):
                    return _observations(base, cfg, last_action)
            proxy = InferenceEnv()
            proxy.device, proxy.cfg = device, {'env': cfg}
            runner = OnPolicyRunner(proxy, runner_cfg, log_dir=None, device=str(device))
            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=str(device))
        tip_time = torch.full((n,), float('nan'), device=device)
        rows = []
        fields = ['time_s', 'env', 'label', 'base_mass_kg', 'terminal_mass_kg', 'carriage_mass_kg', 'gimbal_mass_kg', 'mast_mass_kg', 'length_m', 'width_m',
                  'rho', 'roll_rad', 'pitch_rad', 'speed_m_s', 'tipped', 'com_error_m',
                  'roll_joint_rad', 'pitch_joint_rad', 'lift_m', 'target_roll_rad', 'target_pitch_rad', 'target_lift_m',
                  'cmd_vx_m_s', 'cmd_vy_m_s', 'cmd_wz_rad_s']
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        try:
            view.start()
            command = fixed.clone()
            with open(args.csv, 'w', newline='') as output, torch.inference_mode():
                writer = csv.writer(output)
                writer.writerow(fields)
                steps = max(1, math.ceil(duration/base.dt))
                for step in range(steps):
                    if step % cfg.decimation == 0:
                        if policy is not None:
                            action = policy(_observations(base, cfg, last_action)).clamp(-1, 1)
                            action[::2] = 0
                            command = action*max_vel
                        else:
                            command = fixed if args.scenario == 'spline' or step*base.dt >= args.settle else torch.zeros_like(fixed)
                        last_action = command/max_vel
                    t = (step+1)*base.dt
                    if args.scenario == 'spline':
                        q, qd, _ = splines.evaluate(torch.full((n,), t, device=device))
                    else:
                        q, qd = protocol_targets(t, initial, target, args.settle, args.swing, args.hold,
                                                 args.scenario == 'joint_sweep')
                    base.step(q, command, qd)
                    new = base.tipped & torch.isnan(tip_time)
                    tip_time[new] = t
                    for i in new.nonzero(as_tuple=False).flatten().tolist():
                        print(f'[TIP] t={t:.3f}s {labels[i]}')
                    view.update(t, labels)
                    if step % cfg.decimation == 0 or step == steps-1:
                        robot = base.robot
                        actual = robot.data.joint_pos.torch[:, base.joint_ids]
                        expected = analytical_com(actual, robot.data.body_pos_w.torch[:, base.base_id],
                                                  robot.data.body_quat_w.torch[:, base.base_id], base.params, cfg.mechanism)
                        errors = (expected-base.com_total).norm(dim=-1)
                        speed = robot.data.body_com_lin_vel_w.torch[:, base.base_id, :2].norm(dim=-1)
                        data = torch.cat(((base.com_offset/base.half_length)[:, None], base.roll[:, None],
                            base.pitch[:, None], speed[:, None], base.tipped.float()[:, None], errors[:, None],
                            actual, q, command), -1).cpu().tolist()
                        for i, values in enumerate(data):
                            p = base.params[i]
                            row = [t, i, labels[i], p.base_mass, p.terminal_mass, p.carriage_mass, p.gimbal_mass, p.mast_mass, *p.footprint, *values]
                            writer.writerow(row)
                            if args.plot:
                                rows.append(row)
                        if not torch.isfinite(actual).all() or not torch.isfinite(base.com_total).all():
                            raise RuntimeError('non-finite simulation state; inspect the CSV trace')
            for i in range(n):
                print(f'[RESULT] {labels[i]}: tipped={bool(base.tipped[i])}, first_tip_s={float(tip_time[i]):.3f}')
            print(f'[INFO] CSV: {args.csv}')
            if args.plot:
                _plot(rows, fields, args.csv)
            if args.video:
                print(f'[INFO] Video: {args.video_path}')
        finally:
            view.close()


def _plot(rows, fields, csv_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    # All states are retained after failure; curves show the ensuing physical fall.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for i in sorted({int(row[1]) for row in rows}):
        samples = [row for row in rows if int(row[1]) == i]
        t = np.array([r[0] for r in samples])
        label = samples[0][2]
        for ax, field in zip(axes.flat, ('rho', 'roll_rad', 'pitch_rad', 'speed_m_s')):
            ax.plot(t, [r[fields.index(field)] for r in samples], label=label,
                    linestyle='--' if '/ zero' in label else '-')
            ax.set_ylabel(field)
            ax.grid(alpha=.25)
    axes[0, 0].axhline(1, color='black', linewidth=1)
    for ax in (axes[0, 1], axes[1, 0]):
        ax.axhline(.5, color='black', linewidth=1)
        ax.axhline(-.5, color='black', linewidth=1)
    for ax in axes[1]:
        ax.set_xlabel('Time [s]')
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = Path(csv_path).with_suffix('.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'[INFO] Plot: {path}')
