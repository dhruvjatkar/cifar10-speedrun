#############################################
#                  Setup                    #
#############################################

import os
from math import ceil
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
import torchvision


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ------------------------------
# Config knobs for direct A/Bs.
# ------------------------------
ORTHO_IMPL = os.environ.get("ORTHO_IMPL", "polar_express_fast")
# Options:
#   legacy_fast         -> same odd polynomial family as your current script, but without
#                          cross-shape padding and with the rectangular fast-path.
#   cans_fast           -> degree-5 CANS coefficients from the 2025/2026 paper.
#   polar_express_fast  -> default; degree-5 Polar Express coefficients.

EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "10000"))
TTA_BATCH_SIZE = int(os.environ.get("TTA_BATCH_SIZE", "2500"))
TTA_UNCERTAIN_QUANTILE = 0.25
TTA_PAD = 1
ORTHO_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
FAST_RECT_RESTART = int(os.environ.get("FAST_RECT_RESTART", "3"))
FAST_RECT_FIRST_SHIFT = float(os.environ.get("FAST_RECT_FIRST_SHIFT", "0.0"))
POLAR_FAST_FALLBACK_TO_DIRECT = os.environ.get("POLAR_FAST_FALLBACK_TO_DIRECT", "0") == "1"

#############################################
#          Polynomial orthogonalizers       #
#############################################

# Your current odd degree-5 polynomial, repeated 3 times.
LEGACY_COEFFS = (
    (3.4576, -4.7391, 2.0843),
    (3.4576, -4.7391, 2.0843),
    (3.4576, -4.7391, 2.0843),
)

# Polar Express degree-5 coeffs from Algorithm 1, with the 1.01 safety scaling
# applied to every polynomial except the terminal asymptotic one.
_POLAR_RAW = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
_POLAR_SAFETY = 1.01
POLAR_EXPRESS_COEFFS = tuple(
    (a / _POLAR_SAFETY, b / (_POLAR_SAFETY ** 3), c / (_POLAR_SAFETY ** 5))
    for a, b, c in _POLAR_RAW
)

# CANS degree-5, 4-iteration coefficients (table entries for mm=12).
CANS5_COEFFS = (
    (8.420293602126344, -24.910491192120688, 18.472094206318726),
    (4.101228661246281, -3.0518555467946813, 0.5741241025302702),
    (3.6809819251109155, -2.75396502307162, 0.5401902781108926),
    (2.7280916801566666, -2.0315492757300913, 0.45866431681858805),
)


def _poly_direct_impl(X: torch.Tensor, coeffs: tuple[tuple[float, float, float], ...], norm_mul: float, eps: float):
    """
    Baseline odd-polynomial composition:
        X <- aX + (bA + cA^2)X,  A = XX^T

    X is assumed batched with shape [B, M, N] and M <= N.
    """
    X = X.to(ORTHO_DTYPE)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * norm_mul + eps)
    for a, b, c in coeffs:
        A = X @ X.transpose(-1, -2)
        X = a * X + (b * A + c * (A @ A)) @ X
    return X


def _poly_fast_wide_impl(
    X: torch.Tensor,
    coeffs: tuple[tuple[float, float, float], ...],
    norm_mul: float,
    eps: float,
    restart_interval: int,
    first_shift: float,
):
    """
    Stabilized fast rectangular polynomial iteration for wide matrices.

    For wide X (M <= N), any odd polynomial p(x) = x h(x^2) can be applied as
        p(X) = h(XX^T) X.

    The exact-arithmetic fast recurrence keeps a left factor P_t in the small M x M space.
    In finite precision, however, the wide analogue of Appendix F / Algorithm 4 must form
        R_t = P_{t-1} (XX^T) P_{t-1}^T
    rather than P_{t-1} (XX^T) P_{t-1}; otherwise R_t can drift away from being symmetric /
    PSD and the update ceases to track the intended odd polynomial composition.

    We also support restarting every few polynomials, which is the stabilization strategy
    recommended in Appendix F for longer compositions.
    """
    X = X.to(ORTHO_DTYPE)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * norm_mul + eps)

    if restart_interval <= 0:
        restart_interval = len(coeffs)

    batch_size = X.shape[0]
    for block_start in range(0, len(coeffs), restart_interval):
        Y = X @ X.transpose(-1, -2)
        _, m, _ = Y.shape
        eye = torch.eye(m, device=X.device, dtype=X.dtype).unsqueeze(0).expand(batch_size, m, m)
        if first_shift != 0.0 and block_start == 0:
            Y = Y + first_shift * eye
        P = eye.clone()

        block_end = min(block_start + restart_interval, len(coeffs))
        for idx in range(block_start, block_end):
            a, b, c = coeffs[idx]
            R = P @ Y @ P.transpose(-1, -2)
            H = a * eye + b * R + c * (R @ R)
            P = H @ P

        X = P @ X

    return X


@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
def _legacy_direct(X: torch.Tensor):
    # The original script's two-stage magnitude normalization collapses algebraically
    # to a single norm-normalization up to the added eps terms, so we use the simpler form.
    return _poly_direct_impl(X, LEGACY_COEFFS, norm_mul=1.0, eps=1e-6)


@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
def _legacy_fast_wide(X: torch.Tensor):
    return _poly_fast_wide_impl(
        X,
        LEGACY_COEFFS,
        norm_mul=1.0,
        eps=1e-6,
        restart_interval=FAST_RECT_RESTART,
        first_shift=0.0,
    )


@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
def _cans_direct(X: torch.Tensor):
    return _poly_direct_impl(X, CANS5_COEFFS, norm_mul=1.0, eps=1e-6)


@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
def _cans_fast_wide(X: torch.Tensor):
    return _poly_fast_wide_impl(
        X,
        CANS5_COEFFS,
        norm_mul=1.0,
        eps=1e-6,
        restart_interval=FAST_RECT_RESTART,
        first_shift=0.0,
    )


@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
def _polar_express_direct(X: torch.Tensor):
    return _poly_direct_impl(X, POLAR_EXPRESS_COEFFS, norm_mul=1.01, eps=1e-7)


@torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
def _polar_express_fast_wide(X: torch.Tensor):
    return _poly_fast_wide_impl(
        X,
        POLAR_EXPRESS_COEFFS,
        norm_mul=1.01,
        eps=1e-7,
        restart_interval=FAST_RECT_RESTART,
        first_shift=FAST_RECT_FIRST_SHIFT,
    )


def orthogonalize_batch(G: torch.Tensor, impl: str) -> torch.Tensor:
    """
    G: [B, M, N]

    We transpose tall matrices to make the working representation wide, because both the
    direct and rectangular-fast paths are cheaper when the small dimension is on the left.
    """
    transposed = G.size(-2) > G.size(-1)
    X = G.transpose(-1, -2) if transposed else G

    # Use the fast rectangular path only when the aspect ratio is large enough to justify it.
    use_fast = 2 * X.size(-1) >= 3 * X.size(-2)

    if impl == "legacy_fast":
        X = _legacy_fast_wide(X) if use_fast else _legacy_direct(X)
    elif impl == "cans_fast":
        X = _cans_fast_wide(X) if use_fast else _cans_direct(X)
    elif impl == "polar_express_fast":
        # Appendix F was not used in the Polar Express Muon experiments; keep a one-line
        # escape hatch to the direct path in case the rectangular recurrence still needs
        # retuning for a given hardware / dtype combination.
        X = _polar_express_fast_wide(X) if (use_fast and not POLAR_FAST_FALLBACK_TO_DIRECT) else _polar_express_direct(X)
    else:
        raise ValueError(f"Unknown ORTHO_IMPL={impl!r}")

    if transposed:
        X = X.transpose(-1, -2)
    return X.to(G.dtype)

#############################################
#                Data helpers               #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda", dtype=torch.float16).view(1, 3, 1, 1)


@torch.compile(fullgraph=True)
def normalize_images(images: torch.Tensor) -> torch.Tensor:
    return (images - CIFAR_MEAN) / CIFAR_STD


@torch.compile(fullgraph=True)
def batch_color_jitter(inputs: torch.Tensor, brightness_range: float, contrast_range: float):
    b = inputs.shape[0]
    brightness_shift = (
        torch.rand(b, 1, 1, 1, device=inputs.device, dtype=inputs.dtype) * 2 - 1
    ) * brightness_range
    contrast_scale = (
        torch.rand(b, 1, 1, 1, device=inputs.device, dtype=inputs.dtype) * 2 - 1
    ) * contrast_range + 1
    return (inputs + brightness_shift) * contrast_scale


@torch.compile(fullgraph=True)
def batch_flip_lr(inputs: torch.Tensor):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


@torch.compile(fullgraph=True)
def batch_crop(images: torch.Tensor, crop_size: int):
    b, c, h_padded, _ = images.shape
    span = h_padded - crop_size + 1
    y_offsets = torch.randint(0, span, (b, 1, 1, 1), device=images.device)
    x_offsets = torch.randint(0, span, (b, 1, 1, 1), device=images.device)

    base_y = torch.arange(crop_size, device=images.device).view(1, 1, crop_size, 1)
    base_x = torch.arange(crop_size, device=images.device).view(1, 1, 1, crop_size)

    y_idx = (y_offsets + base_y).expand(b, c, crop_size, crop_size)
    x_idx = (x_offsets + base_x).expand(b, c, crop_size, crop_size)
    batch_idx = torch.arange(b, device=images.device).view(b, 1, 1, 1).expand_as(y_idx)
    chan_idx = torch.arange(c, device=images.device).view(1, c, 1, 1).expand_as(y_idx)
    return images[batch_idx, chan_idx, y_idx, x_idx]


class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device("cuda"), weights_only=True)
        self.images = (data["images"].half() / 255).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        self.labels = data["labels"].contiguous()
        self.classes = data["classes"]

        self._norm_images = None
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train
        self._shuffle_indices = torch.empty(len(self.images), dtype=torch.long, device="cuda")
        self._sequential_indices = torch.arange(len(self.images), device="cuda", dtype=torch.long)

    def __len__(self):
        if self.drop_last:
            return len(self.images) // self.batch_size
        return ceil(len(self.images) / self.batch_size)

    @property
    def norm_images(self):
        if self._norm_images is None:
            self._norm_images = normalize_images(self.images).clone()  # clone outside compile to avoid cudagraph output reuse
        return self._norm_images

    def __iter__(self):
        if self.epoch == 0:
            base = self.proc_images["norm"] = self.norm_images
            if self.aug.get("flip", False):
                self.proc_images["flip_seed0"] = batch_flip_lr(base)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(base, (pad,) * 4, "reflect")

        pad = self.aug.get("translate", 0)
        if pad > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-1])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip_seed0"]
        else:
            images = self.proc_images["norm"]

        if self.aug.get("flip", False) and (self.epoch % 2 == 1):
            images = images.flip(-1)

        color_jitter_config = self.aug.get("color_jitter", {"enabled": False})
        if color_jitter_config.get("enabled", False):
            images = batch_color_jitter(
                images,
                color_jitter_config.get("brightness_range", 0.1),
                color_jitter_config.get("contrast_range", 0.1),
            )

        self.epoch += 1

        if self.shuffle:
            torch.randperm(len(self.images), out=self._shuffle_indices)
            indices = self._shuffle_indices
        else:
            indices = self._sequential_indices

        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield images[idxs], self.labels[idxs]

#############################################
#            Network Definition             #
#############################################

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.5566, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            mod.half()
        self.to(memory_format=torch.channels_last)

    def reset(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        w = self.head.weight.data
        w.mul_(1.0 / w.std())

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = (
            train_images.unfold(2, h, 1)
            .unfold(3, w, 1)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(-1, c, h, w)
            .float()
        )
        patches_flat = patches.flatten(1)
        cov = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        inv_sqrt = torch.rsqrt(eigvals.clamp_min(0) + eps)
        basis = (eigvecs * inv_sqrt.unsqueeze(0)).T.reshape(-1, c, h, w)
        self.whiten.weight.data[:] = torch.cat((basis, -basis), dim=0).to(self.whiten.weight.dtype)

    def forward(self, x, whiten_bias_grad=True):
        x = x.to(memory_format=torch.channels_last)
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.reshape(len(x), -1)
        return self.head(x) / x.size(-1)

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-" * len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-" * len(print_string))


logging_columns_list = ["run   ", "epoch", "train_acc", "val_acc", "tta_val_acc", "time_seconds"]


def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#         Top-level compiled helpers       #
############################################

@torch.compile(fullgraph=True)
def forward_step(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, whiten_bias_grad: bool):
    outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
    return F.cross_entropy(outputs, labels, label_smoothing=0.09, reduction="sum")


@torch.compile(fullgraph=True)
def infer_basic_batch(model: nn.Module, inputs: torch.Tensor):
    return model(inputs)


@torch.compile(fullgraph=True)
def infer_mirror_batch(model: nn.Module, inputs: torch.Tensor):
    return 0.5 * model(inputs) + 0.5 * model(inputs.flip(-1))


@torch.compile(fullgraph=True)
def tta_logits_batch(model: nn.Module, images_batch: torch.Tensor):
    padded_inputs = F.pad(images_batch, (TTA_PAD,) * 4, "reflect")
    crop_tl = padded_inputs[:, :, 0:32, 0:32]
    crop_br = padded_inputs[:, :, 2:34, 2:34]
    base_views = torch.cat((images_batch, crop_tl, crop_br), dim=0)
    combined_inputs = torch.cat((base_views, base_views.flip(-1)), dim=0)
    combined_logits = model(combined_inputs)
    return combined_logits.view(6, images_batch.shape[0], -1).mean(dim=0)

#############################################
#              Muon optimizer               #
#############################################

class MuonFast(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.08,
        momentum=0.88,
        nesterov=True,
        norm_freq=1,
        total_train_steps=None,
        weight_decay=0.0,
        ortho_impl="polar_express_fast",
        preserve_legacy_param_renorm=True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            norm_freq=norm_freq,
            total_train_steps=total_train_steps,
            weight_decay=weight_decay,
            ortho_impl=ortho_impl,
            preserve_legacy_param_renorm=preserve_legacy_param_renorm,
        )
        super().__init__(params, defaults)
        self.step_count = 0
        self.last_norm_step = 0
        self.total_train_steps = total_train_steps

        buckets = defaultdict(list)
        for group in self.param_groups:
            for p in group["params"]:
                if len(p.shape) == 4 and p.requires_grad:
                    d = p.shape[0]
                    k = p.numel() // d
                    buckets[(d, k)].append(p)
        self.shape_buckets = dict(sorted(buckets.items()))

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        group = self.param_groups[0]
        progress = self.step_count / self.total_train_steps
        group["norm_freq"] = 2 + int(15 * progress)

        do_param_renorm = self.step_count - self.last_norm_step >= group["norm_freq"]
        if do_param_renorm:
            self.last_norm_step = self.step_count

        weight_decay_factor = 1.0 - group["lr"] * group["weight_decay"]
        momentum = group["momentum"]
        momentum_buffer_dtype = group["momentum_buffer_dtype"]

        for (d, k), params in self.shape_buckets.items():
            active_params = [p for p in params if p.grad is not None]
            if not active_params:
                continue

            momentum_buffers = []
            for p in active_params:
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad,
                        dtype=momentum_buffer_dtype,
                        memory_format=torch.preserve_format,
                    )
                momentum_buffers.append(state["momentum_buffer"])

            grads = [p.grad for p in active_params]
            if momentum_buffer_dtype != grads[0].dtype:
                grads_for_buf = [g.to(momentum_buffer_dtype) for g in grads]
            else:
                grads_for_buf = grads

            torch._foreach_mul_(momentum_buffers, momentum)
            torch._foreach_add_(momentum_buffers, grads_for_buf)

            if group["nesterov"]:
                update_grads = torch._foreach_add(grads_for_buf, momentum_buffers, alpha=momentum)
            else:
                update_grads = list(momentum_buffers)

            stacked_updates = torch.stack([g.reshape(d, k) for g in update_grads], dim=0)
            ortho_updates = orthogonalize_batch(stacked_updates, group["ortho_impl"])

            if do_param_renorm and group["preserve_legacy_param_renorm"]:
                # Kept for apples-to-apples fairness with the legacy script.
                param_norms = torch._foreach_norm(active_params)
                scale_factors = [
                    (len(p.data) ** 0.5 / (n + 1e-7)).to(p.dtype)
                    for p, n in zip(active_params, param_norms)
                ]
                torch._foreach_mul_(active_params, scale_factors)

            ortho_update_list = [u.view_as(p).to(p.dtype) for u, p in zip(ortho_updates, active_params)]
            torch._foreach_add_(active_params, ortho_update_list, alpha=-group["lr"])

            if weight_decay_factor != 1.0:
                torch._foreach_mul_(active_params, weight_decay_factor)

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):
    test_images = loader.norm_images
    model.eval()

    with torch.inference_mode():
        if tta_level == 0:
            return torch.cat(
                [
                    infer_basic_batch(model, inputs.contiguous(memory_format=torch.channels_last)).clone()
                    for inputs in test_images.split(EVAL_BATCH_SIZE)
                ],
                dim=0,
            )

        if tta_level == 1:
            return torch.cat(
                [
                    infer_mirror_batch(model, inputs.contiguous(memory_format=torch.channels_last)).clone()
                    for inputs in test_images.split(EVAL_BATCH_SIZE)
                ],
                dim=0,
            )

        initial_logits = torch.cat(
            [
                infer_basic_batch(model, inputs.contiguous(memory_format=torch.channels_last)).clone()
                for inputs in test_images.split(EVAL_BATCH_SIZE)
            ],
            dim=0,
        )
        confidences = initial_logits.softmax(dim=1).amax(dim=1)
        n = test_images.shape[0]
        k_uncertain = max(1, int(n * TTA_UNCERTAIN_QUANTILE))
        uncertain_indices = torch.topk(confidences, k_uncertain, largest=False, sorted=False).indices

        final_logits = initial_logits.clone()
        for batch_indices in uncertain_indices.split(TTA_BATCH_SIZE):
            batch_images = test_images[batch_indices].contiguous(memory_format=torch.channels_last)
            final_logits[batch_indices] = tta_logits_batch(model, batch_images).clone()
        return final_logits


def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

def main(run, model):
    training_batch_size = 1536
    bias_lr = 0.0573
    head_lr = 0.5415
    wd = 1.0418e-06 * training_batch_size

    test_loader = CifarLoader("cifar10", train=False, batch_size=EVAL_BATCH_SIZE)
    train_loader = CifarLoader(
        "cifar10",
        train=True,
        batch_size=training_batch_size,
        aug={
            "flip": True,
            "translate": 2,
            "color_jitter": {
                "enabled": True,
                "brightness_range": 0.1399,
                "contrast_range": 0.1308,
            },
        },
    )

    if run == "warmup":
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
        train_loader.images = torch.randn_like(train_loader.images, device=train_loader.images.device)
        train_loader._norm_images = None
        train_loader.proc_images = {}

        test_loader.labels = torch.randint(0, 10, size=(len(test_loader.labels),), device=test_loader.labels.device)
        test_loader.images = torch.randn_like(test_loader.images, device=test_loader.images.device)
        test_loader._norm_images = None
        test_loader.proc_images = {}

    total_train_steps = ceil(7.65 * len(train_loader))
    whiten_bias_train_steps = ceil(0.2 * len(train_loader))

    model.reset()

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]

    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]

    optimizer1 = torch.optim.SGD(param_configs, momentum=0.825, nesterov=True, fused=True)
    optimizer2 = MuonFast(
        filter_params,
        lr=0.205,
        momentum=0.655,
        nesterov=True,
        norm_freq=4,
        total_train_steps=total_train_steps,
        weight_decay=wd,
        ortho_impl=ORTHO_IMPL,
        preserve_legacy_param_renorm=True,
    )
    optimizer2.param_groups[0]["momentum_buffer_dtype"] = torch.float16

    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # For accurately timing GPU code.
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer():
        starter.record()

    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    step = 0
    start_timer()

    with torch.inference_mode():
        model.init_whiten(train_loader.norm_images[:960])

    lr_factor1_base = 1.0 / max(1, whiten_bias_train_steps)
    lr_factor2_base = 1.0 / total_train_steps
    bias_group = optimizer1.param_groups[0]
    scheduled_groups = optimizer1.param_groups[1:] + optimizer2.param_groups
    bias_group_initial_lr = bias_group["initial_lr"]
    scheduled_group_initial_lrs = [group["initial_lr"] for group in scheduled_groups]

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        model.train()
        for inputs, labels in train_loader:
            whiten_bias_grad = step < whiten_bias_train_steps
            loss = forward_step(model, inputs, labels, whiten_bias_grad)
            loss.backward()

            lr_factor1 = 1 - step * lr_factor1_base
            lr_factor2 = 1 - step * lr_factor2_base
            bias_group["lr"] = bias_group_initial_lr * lr_factor1
            for group, initial_lr in zip(scheduled_groups, scheduled_group_initial_lrs):
                group["lr"] = initial_lr * lr_factor2

            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            if step >= total_train_steps:
                break
        if step >= total_train_steps:
            break

    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()

    epoch = "eval"
    train_acc = evaluate(model, train_loader, tta_level=0)
    val_acc = evaluate(model, test_loader, tta_level=0)
    print_training_details(locals(), is_final_entry=True)
    return val_acc, tta_val_acc, time_seconds


if __name__ == "__main__":
    print(f"# ORTHO_IMPL={ORTHO_IMPL} | ORTHO_DTYPE={ORTHO_DTYPE} | EVAL_BATCH_SIZE={EVAL_BATCH_SIZE} | TTA_BATCH_SIZE={TTA_BATCH_SIZE}")
    model = CifarNet().cuda().to(memory_format=torch.channels_last)

    print_columns(logging_columns_list, is_head=True)
    main("warmup", model)

    results = []
    for run in range(200):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda._sleep(int(6000000000))
        val_acc, tta_val_acc, time_seconds = main(run + 1, model)
        results.append((val_acc, tta_val_acc, time_seconds))
        accs_so_far = [a for _, a, _ in results]
        times_so_far = [t for _, _, t in results]
        print(
            f"Mean accuracy after {run + 1} runs: {sum(accs_so_far) / len(accs_so_far):.6f} | Mean time: {sum(times_so_far) / len(times_so_far):.6f}s",
            end="\r",
            flush=True,
        )

    print()
    _, accs, times = zip(*results)
    accs = torch.tensor(accs)
    times = torch.tensor(times)
    print("Accuracies: Mean: %.6f    Std: %.6f" % (accs.mean(), accs.std()))
    print("Times (s):  Mean: %.6f    Std: %.6f" % (times.mean(), times.std()))
