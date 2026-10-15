"""Analytical geometry checks, independent of Isaac (metres, kilograms, xyzw)."""
import torch


def analytical_com(joint_pos, root_pos, root_quat, params, cfg):
    """Total CoM (N, 3) from measured joints (N, 3), root pose and physical parameters.

    Joint order: roll [rad], pitch [rad], lift [m]. Body origins and
    dimensions match the primitive mechanism; this does not use simulator CoM data.
    """
    roll, pitch, lift = joint_pos.unbind(-1)
    # R_x(roll) R_y(pitch) applied to the upright mast's local +Z axis.
    direction = torch.stack((pitch.sin(), -roll.sin()*pitch.cos(), roll.cos()*pitch.cos()), -1)
    carriage = torch.zeros_like(joint_pos)
    carriage[:, 2] = cfg.platform_height + 0.08 + lift
    positions = torch.stack((carriage, carriage, carriage + .5*cfg.mast_length*direction,
                             carriage + (cfg.mast_length+.04)*direction), 1)
    masses = joint_pos.new_tensor([[p.carriage_mass, p.gimbal_mass, p.mast_mass, p.terminal_mass] for p in params])
    base_mass = joint_pos.new_tensor([p.base_mass for p in params])
    weighted = (positions * masses[..., None]).sum(1)
    weighted[:, 2] += base_mass * joint_pos.new_tensor([p.base_com_z for p in params])
    local = weighted / (base_mass + masses.sum(1))[:, None]
    xyz, w = root_quat[:, :3], root_quat[:, 3:]
    uv = 2 * torch.linalg.cross(xyz, local)
    return root_pos + local + w * uv + torch.linalg.cross(xyz, uv)


def protocol_targets(time, initial, target, settle, swing, hold, return_to_start=False):
    """Quintic extend/hold or out-and-back probe, with analytic velocity.

    All joints use the same phase; set unchanged target entries for a single-joint probe.
    The caller supplies initial/target tensors (N, 3). No position is written directly.
    """
    reverse = return_to_start and time > settle + swing + hold
    local_time = time - (settle + swing + hold if reverse else settle)
    u = min(max(local_time/swing, 0.0), 1.0)
    blend = 10*u**3 - 15*u**4 + 6*u**5
    speed = (30*u**2 - 60*u**3 + 30*u**4)/swing
    if reverse:
        return target + blend*(initial-target), speed*(initial-target)
    return initial + blend*(target-initial), speed*(target-initial)
