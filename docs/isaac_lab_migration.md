# mobile_franka on Isaac Lab 3.0 / Isaac Sim 6.x

> Note: in this repo the demo lives at `scripts/isaac/ridgeback_demo.py`. Sections 2 and 4 below describe the original standalone `mobile_franka` repo and are kept for reference only.

How to recreate our Ridgeback-Franka demo on your installation and set up the shared repo.
From now on, **Isaac Lab 3.0 is the version we develop on.**

## What the demo does

It spawns NVIDIA's Ridgeback-Franka (an omnidirectional base with a Franka Panda arm). Then it:
- drives the base through a schedule (stand still, forward, backward, left, right, turn CCW, turn CW; 2 s each),
- swings arm joints 1 and 4,
- opens and closes the gripper every second,
- resets every 14 s.

At each phase it prints the base pose in **world coordinates**.

Target architecture later: arm controlled by a QP, base controlled by an RL policy.

## What changed compared to our Isaac Lab 2.3 version

| Topic | 2.3 (old) | 3.0 (this version) |
|---|---|---|
| Robot USD | The 5.1 asset had no fixed joint anchoring the dummy `world` link, so the base didn't move and the robot tumbled. We shipped `assets/ridgeback_franka_fixed.usda` to add it. | The 6.x asset already contains `/panda_mobile/world/FixedJoint`, so the **override is dropped** and the stock `RIDGEBACK_FRANKA_PANDA_CFG` is used. |
| App start | `AppLauncher(args)` | `add_launcher_args(parser)` + `with launch_simulation(sim_cfg, args):` |
| GUI / headless | GUI by default, `--headless` | Headless by default, `--viz kit` for the GUI |
| Data | `robot.data.x` (tensor) | `robot.data.x.torch` (ProxyArray → tensor) |
| Targets | `robot.set_joint_velocity_target(v, joint_ids=...)` | `robot.actuators.target_command.set_velocity_index(value=v, joint_ids=...)` (same for `set_position_index`) |
| State writes | `write_root_pose_to_sim(...)`, `write_joint_state_to_sim(...)`, `default_root_state` | `write_root_pose_to_sim_index(root_pose=...)`, `write_joint_position_to_sim_index(position=...)`, etc.; `default_root_pose` / `default_root_vel` |
| Quaternions | (w, x, y, z) | **(x, y, z, w)** |

One rule still holds in 3.0: **never address joints by index.** The simulator orders joints its own way, so always use `robot.find_joints([...names...], preserve_order=True)`.

## 1. Check your installation

Isaac Lab 3.0 needs Python 3.12 and Isaac Sim 6.x. Run this in your Isaac Lab folder:

```bash
cat VERSION                                                   # should print 3.0.x
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz kit   # Windows: isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py --viz kit
```

If an empty viewport opens, the installation works. The first launch can take more than 10 minutes because of shader compilation and extension downloads.

**Do not edit the Isaac Lab checkout itself.** Everything custom goes in our repo, so everyone shares an unmodified Isaac Lab.

## 2. Create the project repo

Put the project folder **next to** Isaac Lab, not inside it. Use the files I sent (`scripts/isaac/ridgeback_demo.py`, `.gitignore`, `MIGRATION.md`):

```bash
mkdir -p ~/repos/mobile_franka/scripts && cd ~/repos/mobile_franka
# copy ridgeback_demo.py into scripts/, and .gitignore + MIGRATION.md into the root
```

Layout:
```
mobile_franka/
├── .gitignore
├── MIGRATION.md
└── scripts/
    └── ridgeback_demo.py
```

## 3. Run it

Activate the Python environment you installed Isaac Lab into, then from inside `mobile_franka/` run:

```bash
# Linux
<path-to-IsaacLab>/isaaclab.sh -p scripts/isaac/ridgeback_demo.py --viz kit   # GUI
<path-to-IsaacLab>/isaaclab.sh -p scripts/isaac/ridgeback_demo.py             # headless
```
```bat
REM Windows
<path-to-IsaacLab>\isaaclab.bat -p scripts\isaac\ridgeback_demo.py --viz kit
```
Optional: `--base_speed 0.3` sets the base speed in m/s and the turn rate in rad/s (default 0.5).

Use `isaaclab.sh -p` / `isaaclab.bat -p` (or `uv run python ...` if you use the uv workflow), not plain `python`, because it sets up the Isaac Sim paths. In 3.0, `isaaclab.sh` prints a deprecation warning; it is slated for removal in 3.1 in favour of `uv run isaaclab`, but it still works.

### Expected output (check this!)

The base must really move in the **world** frame. The printed `base_link` pose should go roughly like this:

```
t= 0.00s  base_link at x=+0.00, y=+0.00, yaw=+0.00 -> standing still
t= 2.00s  base_link at x=+0.00, y=+0.00             -> driving forward (+x)
t= 4.00s  base_link at x≈+1.0                       -> driving backward (-x)
t= 6.00s  base_link at x≈ 0.0                       -> sliding left (+y)
t= 8.00s  y≈+1.0 ...  then yaw changes during the turning phases
```

A few cm of drift is normal, caused by lag when the base speeds up and slows down. If x and y stay at 0 while the joints move, the base anchor is missing again. In that case, tell me and we add a `USD override` like the one we had for 2.3: a `PhysicsFixedJoint` with `body1 = </panda_mobile/world>`.

## 4. Put it on GitHub (private)

```bash
cd ~/repos/mobile_franka
git init -b main
git add .gitignore MIGRATION.md scripts/
git commit -m "Ridgeback-Franka base+arm demo for Isaac Lab 3.0"

# with the GitHub CLI (gh auth login first), or create an empty private repo on github.com and add it as remote
gh repo create mobile_franka --private --source . --push
#   git remote add origin git@github.com:<user-or-org>/mobile_franka.git && git push -u origin main
```
Then add the colleagues under *Settings → Collaborators*. Don't commit videos, logs or checkpoints; the `.gitignore` already excludes them.

## Next steps (roadmap)

1. Restructure into an installable Isaac Lab extension with Isaac Lab's external-project template (manager-based, rsl_rl); check `uv run isaaclab --help` for the 3.0 command.
2. Manager-based env with RL action = base velocity (vx, vy, wz) on `dummy_base_.*` joints; the arm uses an IK placeholder; the reward is the end-effector reaching a target.
3. Replace the IK with a QP that is **batched over envs** (GPU/torch).
4. Swap in our own mobile Franka: URDF → USD, with the same dummy-joint base and an anchored root.
