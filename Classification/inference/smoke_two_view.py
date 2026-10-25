"""GPU smoke check for the two-channel AST patch embedding used by SC2."""

import torch
from torch import nn

from models.ast import ASTModel


def inflate_projection(model):
    old = model.v.patch_embed.proj
    new = nn.Conv2d(2, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, dilation=old.dilation,
                    groups=old.groups, bias=old.bias is not None,
                    padding_mode=old.padding_mode).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight[:, 0].copy_(old.weight[:, 0])
        new.weight[:, 1].zero_()
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.v.patch_embed.proj = new


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('This smoke check must run inside its Slurm GPU allocation')
    model = ASTModel(label_dim=4, input_fdim=498, input_tdim=128,
                     imagenet_pretrain=True, audioset_pretrain=True, verbose=False).cuda().eval()
    mel = torch.randn(1, 1, 498, 128, device='cuda')
    with torch.no_grad():
        reference = model.mlp_head(model(mel))
    inflate_projection(model)
    two_view = torch.cat((mel, torch.zeros_like(mel)), dim=1)
    with torch.no_grad():
        actual = model.mlp_head(model(two_view))
    max_error = float((reference - actual).abs().max().item())
    if not torch.allclose(reference, actual, rtol=1e-5, atol=1e-5):
        raise AssertionError(f'SC2 zero-channel initialization changed pretrained features; max error={max_error}')

    model.train()
    logits = model(torch.randn(2, 2, 498, 128, device='cuda')).mean()
    logits.backward()
    grad = model.v.patch_embed.proj.weight.grad
    if grad is None or not torch.isfinite(grad).all():
        raise AssertionError('SC2 patch embedding did not produce finite gradients')
    print(f'SC2 zero-channel feature equivalence passed; max_abs_error={max_error:.8g}; backward finite', flush=True)


if __name__ == '__main__':
    main()
