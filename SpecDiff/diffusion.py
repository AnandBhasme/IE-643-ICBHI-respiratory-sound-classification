"""Continuous-time cosine-schedule diffusion with v-prediction, classifier-free
guidance on the class label, and a deterministic DDIM sampler."""
import math

import torch

from model import N_CLASSES

S_OFFSET = 0.008
AB_MIN = 1e-5


def alpha_bar(t):
    f = torch.cos((t + S_OFFSET) / (1 + S_OFFSET) * math.pi / 2) ** 2
    f0 = math.cos(S_OFFSET / (1 + S_OFFSET) * math.pi / 2) ** 2
    return (f / f0).clamp(AB_MIN, 1.0)


def alpha_sigma(t):
    ab = alpha_bar(t)
    return ab.sqrt(), (1 - ab).sqrt()


def training_loss(model, x0, cls, dev, dur, p_drop_class=0.15, p_drop_dev=0.1, min_snr_gamma=5.0,
                  cls_margin_w=0.0, cls_margin=0.02):
    b = x0.shape[0]
    true_cls = cls
    t = torch.rand(b, device=x0.device)
    alpha, sigma = alpha_sigma(t)
    a, s = alpha[:, None, None, None], sigma[:, None, None, None]
    eps = torch.randn_like(x0)
    xt = a * x0 + s * eps
    v = a * eps - s * x0
    cls = torch.where(torch.rand(b, device=x0.device) < p_drop_class, torch.full_like(cls, N_CLASSES), cls)
    dev = torch.where(torch.rand(b, device=x0.device) < p_drop_dev, torch.full_like(dev, 4), dev)
    pred = model(xt, t, cls, dev, dur)
    snr = alpha ** 2 / sigma ** 2
    weight = torch.clamp(snr, max=min_snr_gamma) / (snr + 1)
    per_item = ((pred.float() - v) ** 2).mean(dim=(1, 2, 3))
    loss = (weight * per_item).mean()
    if cls_margin_w > 0:
        # Same noisy input, a random WRONG class: its error should exceed the true-class error
        # by a relative margin. Items whose class was dropped to null are skipped.
        wrong = (true_cls + torch.randint(1, N_CLASSES, (b,), device=x0.device)) % N_CLASSES
        pred_wrong = model(xt, t, wrong, dev, dur)
        per_wrong = ((pred_wrong.float() - v) ** 2).mean(dim=(1, 2, 3))
        kept = (cls == true_cls).float()
        hinge = torch.relu(cls_margin + (per_item - per_wrong) / per_item.detach())
        loss = loss + cls_margin_w * (kept * hinge).sum() / kept.sum().clamp(min=1.0)
    return loss


@torch.no_grad()
def ddim_sample(model, shape, cls, dev, dur, steps=50, guidance=2.0, generator=None, autocast_dtype=torch.bfloat16,
                x_init=None, t_start=1.0):
    """guidance = 1 disables classifier-free guidance. Guidance contrasts the
    class-conditional and class-null predictions at the SAME stethoscope and
    duration, so it amplifies class evidence rather than device evidence.

    x_init (model space) with t_start < 1 switches to partial re-generation
    (SDEdit): x_init is noised to time t_start and denoised from there under
    the given conditioning, so coarse content of x_init is kept."""
    device = cls.device
    x = torch.randn(shape, device=device, generator=generator)
    if x_init is not None:
        a0, s0 = alpha_sigma(torch.tensor(t_start, device=device))
        x = a0 * x_init + s0 * x
    else:
        t_start = 1.0
    ts = torch.linspace(t_start, 0.0, steps + 1, device=device)
    null_cls = torch.full_like(cls, N_CLASSES)
    for i in range(steps):
        t = ts[i].expand(shape[0])
        t_next = ts[i + 1].expand(shape[0])
        with torch.autocast('cuda', dtype=autocast_dtype, enabled=autocast_dtype is not None):
            if guidance == 1.0:
                v = model(x, t, cls, dev, dur)
            else:
                v_both = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([cls, null_cls]),
                               torch.cat([dev, dev]), torch.cat([dur, dur]))
                v_cond, v_null = v_both.chunk(2)
                v = v_null + guidance * (v_cond - v_null)
        v = v.float()
        alpha, sigma = alpha_sigma(t)
        a, s = alpha[:, None, None, None], sigma[:, None, None, None]
        x0 = a * x - s * v
        eps = s * x + a * v
        alpha_n, sigma_n = alpha_sigma(t_next)
        x = alpha_n[:, None, None, None] * x0 + sigma_n[:, None, None, None] * eps
    return x0
