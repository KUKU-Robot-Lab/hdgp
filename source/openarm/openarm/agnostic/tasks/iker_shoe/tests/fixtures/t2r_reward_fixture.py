import torch
import math


def compute_reward(ctx):
    # test fixture of the t2r smoke (not a training reward)
    approach = torch.exp(-5.0 * ctx.palm_gap)
    touch = torch.tanh(ctx.link_shoe_force.sum(dim=(1, 2)) / 5.0)
    lift = torch.clamp(ctx.dz_free / ctx.lift_height, 0.0, 1.0) * ctx.held.float()
    success = 10.0 * ctx.success.float()
    return approach + touch + lift + success, {"approach": approach, "touch": touch, "lift": lift, "success": success}
