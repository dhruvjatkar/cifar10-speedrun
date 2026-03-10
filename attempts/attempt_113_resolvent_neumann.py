#############################################
#                  Setup                    #
#############################################

# Attempt 113: Resolvent Muon with GEMM-only Neumann series (no Cholesky).
# Replaces Cholesky solves with Neumann series: (A+σI)^{-1}X ≈ (1/σ) Σ_{k=0}^{K} (-A/σ)^k X
# All operations are matmuls → torch.compile compatible, FP16-safe on tensor cores.
# Uses 6 resolvent terms, K=3 Neumann terms per resolvent.

import argparse
import os
import sys
if os.path.exists(sys.argv[0]):
    with open(sys.argv[0]) as f:
        code = f.read()
else:
    code = ""
import uuid
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

DEFAULT_ORTHO_BACKEND = "resolvent"
DEFAULT_RESOLVENT_TERMS = 6
DEFAULT_RESOLVENT_RIDGE = 1e-04
DEFAULT_NEUMANN_TERMS = 3

RESOLVENT_GAUSS_LEGENDRE_TABLES = {
    6: {
        "shifts": [
            0.001221165271724694,
            0.041592400462557522,
            0.377857678322422896,
            2.646499085157422382,
            24.042853715553736293,
            818.889975955232671367,
        ],
        "weights": [
            0.058412294956156262,
            0.166449171508366606,
            0.388329596568013413,
            1.027713922056798568,
            4.001913081650773307,
            47.833242812136759881,
        ],
    },
    8: {
        "shifts": [
            0.000410357454964803,
            0.012808050745295277,
            0.096732013250941676,
            0.476094699148843614,
            2.100422461724081202,
            10.337839215708301666,
            78.075893036832752614,
            2436.899800165127544460,
        ],
        "weights": [
            0.033540728449118845,
            0.087714811681077878,
            0.171629114044206110,
            0.329723347074752537,
            0.692558324350655230,
            1.774274185723466069,
            6.848412254557763035,
            81.735394455050538909,
        ],
    },
    10: {
        "shifts": [
            0.000174747333035996,
            0.005234468092545177,
            0.036440796941750453,
            0.156252937903819333,
            0.548835646169986591,
            1.822039087618368081,
            6.399879665722154165,
            27.441770870117650816,
            191.041378478202886981,
            5722.547993301908718422,
        ],
        "weights": [
            0.021786936306051188,
            0.054704464981026511,
            0.098903626700528602,
            0.166863284690868890,
            0.285074352673579168,
            0.519416613448765108,
            1.067904942648698841,
            2.714090662139555743,
            10.450816398887880609,
            124.676788638389737685,
        ],
    },
    12: {
        "shifts": [
            0.000086591891370282,
            0.002535674641524922,
            0.016901465830864527,
            0.067593244339839445,
            0.213599633770046732,
            0.604364770706602439,
            1.654629866712506114,
            4.681655967053585776,
            14.794377896292219887,
            59.166465796940222788,
            394.372362930053611763,
            11548.425426161758878152,
        ],
        "weights": [
            0.015297145316957566,
            0.037554344279937035,
            0.065064468837096573,
            0.102668292998810057,
            0.158897813258484455,
            0.250542048852635868,
            0.414554356898915177,
            0.743904895593350379,
            1.518913524591648745,
            3.849634670046157225,
            14.810395491967510040,
            176.657941926044003367,
        ],
    },
}
_RESOLVENT_QUADRATURE_CACHE = {}

#############################################
#               Muon optimizer              #
#############################################

@torch.compile(fullgraph=True)
def _zeropower_via_newtonschulz5(
    gradients_4d: list[torch.half],
    filter_meta_data: list[tuple],
    max_D: int,
    max_K: int,
    current_step: int,
    total_steps: int,
) -> list[torch.half]:
    a, b, c = (3.4576, -4.7391, 2.0843)
    eps_stable = 1e-05
    eps_gms = 1e-05
    progress_ratio = current_step / max(1, total_steps)

    initial_target_mag = 0.5012
    final_target_mag = 0.0786
    target_magnitude = (
        initial_target_mag * (1 - progress_ratio) + final_target_mag * progress_ratio
    )

    # Use stack instead of pre-allocated tensor for better performance
    if not filter_meta_data:
        return gradients_4d

    grad_list = []
    for meta in filter_meta_data:
        original_shape, reshaped_D, reshaped_K, list_idx = meta
        grad_to_orthogonalize = gradients_4d[list_idx]
        g_reshaped = grad_to_orthogonalize.reshape(reshaped_D, reshaped_K)
        padding_dims = (0, max_K - reshaped_K, 0, max_D - reshaped_D)
        g_padded = F.pad(g_reshaped, padding_dims, "constant", 0)
        grad_list.append(g_padded)

    if not grad_list:
        return gradients_4d

    X = torch.stack(grad_list)
    
    # Fuse normalization operations for better performance
    current_batch_mags = X.norm(dim=(1, 2), keepdim=True)
    scale_factor = target_magnitude / (current_batch_mags + eps_gms)
    X = X * scale_factor
    
    X_norm = X.norm(dim=(1, 2), keepdim=True)
    X = X / (X_norm + eps_stable)
    
    transposed = False
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
        transposed = True
    
    # Unroll the loop for better performance
    A = X @ X.transpose(1, 2)
    B = b * A + c * (A @ A)
    X = a * X + B @ X
    
    A = X @ X.transpose(1, 2)
    B = b * A + c * (A @ A)
    X = a * X + B @ X
    
    A = X @ X.transpose(1, 2)
    B = b * A + c * (A @ A)
    X = a * X + B @ X
    
    if transposed:
        X = X.transpose(1, 2)
        
    final_orthogonalized_grads_list = [None] * len(gradients_4d)
    for i, meta in enumerate(filter_meta_data):
        original_shape, reshaped_D, reshaped_K, list_idx = meta
        orthogonalized_g_padded = X[i]
        orthogonalized_g_reshaped = orthogonalized_g_padded[:reshaped_D, :reshaped_K]
        final_orthogonalized_grads_list[list_idx] = orthogonalized_g_reshaped.view(
            original_shape
        )
    return final_orthogonalized_grads_list


def _get_resolvent_quadrature(num_terms: int, device, dtype):
    if num_terms not in RESOLVENT_GAUSS_LEGENDRE_TABLES:
        raise ValueError(
            f"Unsupported resolvent term count {num_terms}. "
            f"Expected one of {sorted(RESOLVENT_GAUSS_LEGENDRE_TABLES)}."
        )
    cache_key = (num_terms, str(device), dtype)
    cached = _RESOLVENT_QUADRATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    table = RESOLVENT_GAUSS_LEGENDRE_TABLES[num_terms]
    cached = (
        torch.tensor(table["shifts"], device=device, dtype=dtype),
        torch.tensor(table["weights"], device=device, dtype=dtype),
    )
    _RESOLVENT_QUADRATURE_CACHE[cache_key] = cached
    return cached


@torch.compile(fullgraph=True)
def _neumann_resolvent_apply(gram_scaled, X, shifts, weights, neumann_terms):
    """Apply sum-of-resolvents via Neumann series. All GEMM, no Cholesky.

    (A+σI)^{-1} X ≈ (1/σ) Σ_{k=0}^{K} (-A/σ)^k X
    """
    B, D, K = X.shape
    num_terms = shifts.shape[0]
    result = torch.zeros_like(X)

    for t in range(num_terms):
        sigma = shifts[t]
        w = weights[t]
        # Compute (-gram_scaled/σ) — the iteration matrix
        neg_A_over_sigma = -gram_scaled / sigma
        # Neumann series: Y_0 = X/σ, Y_{k+1} = (-A/σ) Y_k + X/σ
        Y = X / sigma
        term = Y.clone()
        for _ in range(neumann_terms):
            term = neg_A_over_sigma @ term
            Y = Y + term
        result = result + w * Y

    return result


def _zeropower_via_resolvents(
    gradients_4d: list[torch.Tensor],
    filter_meta_data: list[tuple],
    max_D: int,
    max_K: int,
    num_terms: int,
    ridge_epsilon: float,
    neumann_terms: int = DEFAULT_NEUMANN_TERMS,
) -> list[torch.Tensor]:
    if not filter_meta_data:
        return gradients_4d

    grad_list = []
    for meta in filter_meta_data:
        original_shape, reshaped_D, reshaped_K, list_idx = meta
        grad_to_orthogonalize = gradients_4d[list_idx]
        g_reshaped = grad_to_orthogonalize.reshape(reshaped_D, reshaped_K)
        padding_dims = (0, max_K - reshaped_K, 0, max_D - reshaped_D)
        grad_list.append(F.pad(g_reshaped, padding_dims, "constant", 0))

    X = torch.stack(grad_list).float()
    transposed = False
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
        transposed = True

    B, D, K = X.shape
    gram = X @ X.transpose(1, 2)
    eye = torch.eye(D, device=gram.device, dtype=gram.dtype).unsqueeze(0)

    gram_scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-8)
    gram_scaled = gram / gram_scale.view(-1, 1, 1)
    gram_scaled = gram_scaled + ridge_epsilon * eye

    shifts, weights = _get_resolvent_quadrature(num_terms, gram.device, gram.dtype)

    orthogonalized = _neumann_resolvent_apply(gram_scaled, X, shifts, weights, neumann_terms)
    orthogonalized = orthogonalized / gram_scale.sqrt().view(-1, 1, 1)

    if transposed:
        orthogonalized = orthogonalized.transpose(1, 2)

    final_orthogonalized_grads_list = [None] * len(gradients_4d)
    for i, meta in enumerate(filter_meta_data):
        original_shape, reshaped_D, reshaped_K, list_idx = meta
        orthogonalized_g_padded = orthogonalized[i]
        orthogonalized_g_reshaped = orthogonalized_g_padded[:reshaped_D, :reshaped_K]
        final_orthogonalized_grads_list[list_idx] = orthogonalized_g_reshaped.reshape(
            original_shape
        ).to(gradients_4d[list_idx].dtype)
    return final_orthogonalized_grads_list


@torch.no_grad()
def validate_resolvent_orthogonalizer(
    filter_shapes: list[tuple[int, int]],
    num_terms: int = DEFAULT_RESOLVENT_TERMS,
    ridge_epsilon: float = DEFAULT_RESOLVENT_RIDGE,
    device: torch.device | None = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unique_shapes = sorted(set(filter_shapes))
    max_orthogonality_error = 0.0
    max_reference_error = 0.0
    generator = torch.Generator(device=device.type)
    generator.manual_seed(0)

    for reshaped_D, reshaped_K in unique_shapes:
        batch_size = 3
        X = torch.randn(
            batch_size,
            reshaped_D,
            reshaped_K,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        spectral_scales = torch.tensor(
            [0.25, 1.0, 4.0], device=device, dtype=torch.float32
        ).view(batch_size, 1, 1)
        X = X * spectral_scales
        U = _zeropower_via_resolvents(
            [X[i] for i in range(batch_size)],
            [((reshaped_D, reshaped_K), reshaped_D, reshaped_K, i) for i in range(batch_size)],
            reshaped_D,
            reshaped_K,
            num_terms,
            ridge_epsilon,
        )
        U = torch.stack(U).float()
        U_ref, _, Vh_ref = torch.linalg.svd(X, full_matrices=False)
        U_ref = U_ref @ Vh_ref
        identity = torch.eye(U.size(1), device=device, dtype=torch.float32).expand(
            batch_size, U.size(1), U.size(1)
        )
        orthogonality_error = (
            (U @ U.transpose(1, 2) - identity).norm(dim=(1, 2)) / (U.size(1) ** 0.5)
        ).max()
        reference_error = (
            (U - U_ref).norm(dim=(1, 2)) / (U_ref.norm(dim=(1, 2)) + 1e-8)
        ).max()
        max_orthogonality_error = max(
            max_orthogonality_error, float(orthogonality_error.item())
        )
        max_reference_error = max(max_reference_error, float(reference_error.item()))

    print(
        "Resolvent validation "
        f"| shapes={len(unique_shapes)} "
        f"| terms={num_terms} "
        f"| ridge={ridge_epsilon:.1e} "
        f"| max_orthogonality_error={max_orthogonality_error:.2e} "
        f"| max_reference_error={max_reference_error:.2e}"
    )
    if max_orthogonality_error > 5e-3 or max_reference_error > 5e-3:
        raise RuntimeError("Resolvent validation exceeded the configured error threshold.")


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.08,
        momentum=0.88,
        nesterov=True,
        norm_freq=1,
        total_train_steps=None,
        weight_decay=0.0,
        orthogonalization_backend=DEFAULT_ORTHO_BACKEND,
        resolvent_terms=DEFAULT_RESOLVENT_TERMS,
        resolvent_ridge=DEFAULT_RESOLVENT_RIDGE,
        neumann_terms=DEFAULT_NEUMANN_TERMS,
    ):
        if orthogonalization_backend not in {"newtonschulz", "resolvent"}:
            raise ValueError(
                "orthogonalization_backend must be 'newtonschulz' or 'resolvent'."
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            norm_freq=norm_freq,
            total_train_steps=total_train_steps,
            weight_decay=weight_decay,
            orthogonalization_backend=orthogonalization_backend,
            resolvent_terms=resolvent_terms,
            resolvent_ridge=resolvent_ridge,
            neumann_terms=neumann_terms,
        )
        super().__init__(params, defaults)
        self.step_count = 0
        self.last_norm_step = 0
        self.total_train_steps = total_train_steps
        self.filter_params_meta = []
        buckets_by_D = {}
        self.max_D, self.max_K = (0, 0)
        for group in self.param_groups:
            for p in group["params"]:
                if len(p.shape) == 4 and p.requires_grad:
                    reshaped_D = p.shape[0]
                    reshaped_K = p.data.numel() // reshaped_D
                    meta = {
                        "param": p,
                        "original_shape": p.data.shape,
                        "reshaped_dims": (reshaped_D, reshaped_K),
                    }
                    self.filter_params_meta.append(meta)
                    buckets_by_D.setdefault(reshaped_D, []).append(meta)
                    self.max_D = max(self.max_D, reshaped_D)
                    self.max_K = max(self.max_K, reshaped_K)
        self.max_D = max(1, self.max_D)
        self.max_K = (
            (max(1, self.max_K) + 15) // 16 * 16
        )
        self.buckets = []
        for reshaped_D in sorted(buckets_by_D):
            entries = buckets_by_D[reshaped_D]
            bucket_max_K = max(entry["reshaped_dims"][1] for entry in entries)
            bucket_max_K = (bucket_max_K + 15) // 16 * 16
            self.buckets.append(
                {
                    "D": reshaped_D,
                    "max_D": max(1, reshaped_D),
                    "max_K": bucket_max_K,
                    "entries": entries,
                }
            )
        self.current_grad_norms = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        group = self.param_groups[0]
        progress = self.step_count / self.total_train_steps
        group["norm_freq"] = 2 + int(15 * progress)
        # Prepare momentum buffers and track meta data
        filter_params_with_grad = []
        filter_meta_for_current_step = []
        momentum_buffers = [] if group["momentum_buffer_dtype"] == torch.half else None

        for p_meta in self.filter_params_meta:
            p = p_meta["param"]
            if p.grad is not None:
                filter_params_with_grad.append(p)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad,
                        dtype=group["momentum_buffer_dtype"],
                        memory_format=torch.preserve_format)
                if momentum_buffers is not None:
                    momentum_buffers.append(state["momentum_buffer"])
                filter_meta_for_current_step.append((
                    p_meta["original_shape"],
                    p_meta["reshaped_dims"][0],
                    p_meta["reshaped_dims"][1],
                    len(filter_params_with_grad) - 1  # Index in filter_params_with_grad
                ))

        if not filter_params_with_grad:
            return

        # Apply momentum and add gradients
        if momentum_buffers is not None:
            torch._foreach_mul_(momentum_buffers, group["momentum"])
            grad_casts = [g.to(mb.dtype) for g, mb in zip([p.grad for p in filter_params_with_grad], momentum_buffers)]
            torch._foreach_add_(momentum_buffers, grad_casts)
        else:
            momentum_buffers = [p.grad for p in filter_params_with_grad]

        if group["nesterov"]:
            nesterov_grads = torch._foreach_add(
                [p.grad for p in filter_params_with_grad], momentum_buffers, alpha=group["momentum"])
        else:
            nesterov_grads = momentum_buffers

        if group["orthogonalization_backend"] == "newtonschulz":
            final_orthogonalized_grads = _zeropower_via_newtonschulz5(
                nesterov_grads,
                filter_meta_for_current_step,
                self.max_D,
                self.max_K,
                self.step_count,
                self.total_train_steps,
            )
        else:
            param_to_nesterov_idx = {
                id(param): idx for idx, param in enumerate(filter_params_with_grad)
            }
            final_orthogonalized_grads = [None] * len(filter_params_with_grad)
            for bucket in self.buckets:
                bucket_grads = []
                bucket_meta = []
                bucket_global_indices = []
                for entry in bucket["entries"]:
                    param = entry["param"]
                    global_idx = param_to_nesterov_idx.get(id(param))
                    if global_idx is None:
                        continue
                    local_idx = len(bucket_grads)
                    bucket_grads.append(nesterov_grads[global_idx])
                    bucket_meta.append(
                        (
                            entry["original_shape"],
                            entry["reshaped_dims"][0],
                            entry["reshaped_dims"][1],
                            local_idx,
                        )
                    )
                    bucket_global_indices.append(global_idx)
                if not bucket_grads:
                    continue
                bucket_result = _zeropower_via_resolvents(
                    bucket_grads,
                    bucket_meta,
                    bucket["max_D"],
                    bucket["max_K"],
                    group["resolvent_terms"],
                    group["resolvent_ridge"],
                    neumann_terms=group["neumann_terms"],
                )
                for local_idx, global_idx in enumerate(bucket_global_indices):
                    final_orthogonalized_grads[global_idx] = bucket_result[local_idx]

        do_norm_scaling = (self.step_count - self.last_norm_step >= group["norm_freq"])
        if do_norm_scaling:
            self.last_norm_step = self.step_count
            self.current_grad_norms = torch._foreach_norm(filter_params_with_grad)
            scale_factors = [
                (len(p.data) ** 0.5 / (n + 1e-07)).to(p.data.dtype)
                for p, n in zip(filter_params_with_grad, self.current_grad_norms)]

        # Apply updates in a single fused operation when possible
        if do_norm_scaling:
            # Scale gradients first
            torch._foreach_mul_(filter_params_with_grad, scale_factors)
            # Then apply the orthogonalized updates
            torch._foreach_add_(
                filter_params_with_grad,
                final_orthogonalized_grads,
                alpha=-group["lr"])
        else:
            # Apply optimizer step directly
            torch._foreach_add_(
                filter_params_with_grad,
                final_orthogonalized_grads,
                alpha=-group["lr"])

        # Apply weight decay in a fused operation
        weight_decay_factor = 1 - group["lr"] * group["weight_decay"]
        if weight_decay_factor != 1.0:
            torch._foreach_mul_(filter_params_with_grad, weight_decay_factor)

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

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.half)
CIFAR_STD = torch.tensor((0.247, 0.2435, 0.2616), dtype=torch.half)

@torch.compile()
def batch_color_jitter(inputs, brightness_range: float, contrast_range: float):
    B = inputs.shape[0]
    device = inputs.device
    dtype = inputs.dtype
    brightness_shift = (
        torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2 - 1
    ) * brightness_range
    contrast_scale = (
        torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2 - 1
    ) * contrast_range + 1
    inputs = inputs + brightness_shift
    inputs = inputs * contrast_scale
    return inputs

@torch.compile()
def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

@torch.compile()
def batch_crop(images, crop_size):
    B, C, H_padded, W_padded = images.shape
    r = (H_padded - crop_size) // 2
    y_offsets = (torch.rand(B, device=images.device) * (2 * r + 1)).long()
    x_offsets = (torch.rand(B, device=images.device) * (2 * r + 1)).long()
    base_y_coords = torch.arange(crop_size, device=images.device).view(
        1, 1, crop_size, 1
    )
    base_x_coords = torch.arange(crop_size, device=images.device).view(
        1, 1, 1, crop_size
    )
    y_start_coords_expanded = y_offsets.view(B, 1, 1, 1)
    x_start_coords_expanded = x_offsets.view(B, 1, 1, 1)
    y_indices = y_start_coords_expanded + base_y_coords
    y_indices = y_indices.expand(B, C, crop_size, crop_size)
    x_indices = x_start_coords_expanded + base_x_coords
    x_indices = x_indices.expand(B, C, crop_size, crop_size)
    batch_indices = (
        torch.arange(B, device=images.device).view(B, 1, 1, 1).expand_as(y_indices)
    )
    channel_indices = (
        torch.arange(C, device=images.device).view(1, C, 1, 1).expand_as(y_indices)
    )
    cropped_images = images[batch_indices, channel_indices, y_indices, x_indices]
    return cropped_images

class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)
        data = torch.load(data_path, map_location=torch.device("cuda"), weights_only=True)
        self.images, self.labels, self.classes = (
            data["images"],
            data["labels"],
            data["classes"],
        )
        self.images = (
            (self.images.half() / 255)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )
        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train
        # Pre-allocate indices tensor for better performance
        self._indices = torch.empty(len(self.images), dtype=torch.long, device="cuda")

    def __len__(self):
        return (
            len(self.images) // self.batch_size
            if self.drop_last
            else ceil(len(self.images) / self.batch_size)
        )

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,)*4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        color_jitter_config = self.aug.get("color_jitter", {"enabled": False})
        if color_jitter_config.get("enabled", False):
            brightness = color_jitter_config.get("brightness_range", 0.1)
            contrast = color_jitter_config.get("contrast_range", 0.1)
            images = batch_color_jitter(images, brightness, contrast)

        self.epoch += 1

        if self.shuffle:
            torch.randperm(len(self._indices), out=self._indices)
            indices = self._indices
        else:
            indices = torch.arange(len(self.images), device=self.images.device)
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Definition             #
#############################################

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.5566, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
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
        self.whiten = nn.Conv2d(
            3, whiten_width, whiten_kernel_size, padding=0, bias=True
        )
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width,     widths["block1"]),
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

    def init_whiten(self, train_images, eps=0.0005):
        c, (h, w) = (train_images.shape[1], self.whiten.weight.shape[2:])
        patches = (
            train_images.unfold(2, h, 1)
            .unfold(3, w, 1)
            .transpose(1, 3)
            .reshape(-1, c, h, w)
            .float()
        )
        patches_flat = patches.view(len(patches), -1)
        # Use more efficient covariance computation with SVD for better numerical stability
        est_patch_covariance = torch.mm(patches_flat.t(), patches_flat) / len(patches_flat)
        U, S, V = torch.svd(est_patch_covariance)
        # More stable inverse square root computation
        inv_sqrt_S = torch.rsqrt(S + eps)
        eigenvectors_scaled = (U * inv_sqrt_S.unsqueeze(0)).T.reshape(-1, c, h, w)
        self.whiten.weight.data[:] = torch.cat(
            (eigenvectors_scaled, -eigenvectors_scaled)
        )

    def forward(self, x, whiten_bias_grad=True):
        x = x.to(memory_format=torch.channels_last)
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1).contiguous()
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
        print("-"*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-"*len(print_string))

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
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):
    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    @torch.compile(fullgraph=True)
    def _get_tta_logits(model, images_batch, pad):
        batch_size = images_batch.shape[0]
        padded_inputs = F.pad(images_batch, (pad,) * 4, "reflect")
        crop_tl = padded_inputs[:, :, 0:32, 0:32]
        crop_br = padded_inputs[:, :, 2:34, 2:34]
        base_views = torch.cat([images_batch, crop_tl, crop_br], dim=0)
        flipped_views = base_views.flip(-1)
        combined_inputs = torch.cat([base_views, flipped_views], dim=0)
        combined_logits = model(combined_inputs)
        num_views = combined_inputs.shape[0] // batch_size
        reshaped_logits = combined_logits.view(num_views, batch_size, -1)
        averaged_logits = reshaped_logits.mean(dim=0)
        return averaged_logits

    @torch.compile()
    def tta(model, test_images) -> torch.Tensor:
        with torch.no_grad():
            model.eval()
            device = test_images.device
            B = 2000
            pad = 1
            n = test_images.shape[0]
            all_logits_list = []
            for inputs_batch in test_images.split(B):
                inputs_batch = inputs_batch.contiguous(
                    memory_format=torch.channels_last
                )
                all_logits_list.append(model(inputs_batch).clone())
            initial_logits = torch.cat(all_logits_list, dim=0)
            probs = F.softmax(initial_logits, dim=1)
            confidences, _ = probs.max(dim=1)
            UNCERTAIN_QUANTILE = 0.25
            k_uncertain = int(n * UNCERTAIN_QUANTILE)
            _, uncertain_indices = torch.topk(
                confidences, k_uncertain, largest=False, sorted=False
            )

            tta_logits_parts = []
            tta_batch_size = 2000
            for i in range(0, k_uncertain, tta_batch_size):
                cur_batch_size = min(tta_batch_size, k_uncertain - i)
                batch_indices = uncertain_indices[i : i + cur_batch_size]
                images_batch = test_images[batch_indices]
                logits_batch = _get_tta_logits(
                    model,
                    images_batch.contiguous(memory_format=torch.channels_last),
                    pad,
                )
                tta_logits_parts.append(logits_batch)

            if tta_logits_parts:
                all_tta_logits_for_uncertain = torch.cat(tta_logits_parts, dim=0)
                final_logits = initial_logits.clone()
                final_logits[uncertain_indices] = all_tta_logits_for_uncertain
                return final_logits
            return initial_logits

    test_images = loader.normalize(loader.images)
    if tta_level < 2:
        model.eval()
        infer_fn = [infer_basic, infer_mirror, None][tta_level]
        with torch.no_grad():
            return torch.cat(
                [infer_fn(inputs, model) for inputs in test_images.split(2000)]
            )
    else:  # tta_level == 2
        return tta(model, test_images)

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

def main(
    run,
    model,
    *,
    ortho_backend=DEFAULT_ORTHO_BACKEND,
    resolvent_terms=DEFAULT_RESOLVENT_TERMS,
    resolvent_ridge=DEFAULT_RESOLVENT_RIDGE,
    neumann_terms=DEFAULT_NEUMANN_TERMS,
):
    training_batch_size = 1536
    bias_lr = 0.0573
    head_lr = 0.5415
    wd = 1.0418e-06 * training_batch_size
    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
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
        train_loader.labels = torch.randint(
            0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device
        )
        train_loader.images = torch.randn_like(
            train_loader.images, device=train_loader.images.device
        )
        test_loader.labels = torch.randint(
            0, 10, size=(len(test_loader.labels),), device=test_loader.labels.device
        )
        test_loader.images = torch.randn_like(
            test_loader.images, device=test_loader.images.device
        )
    total_train_steps = ceil(7.65 * len(train_loader))
    whiten_bias_train_steps = ceil(0.2 * len(train_loader))
    model.reset()
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(
        param_configs, momentum=0.825, nesterov=True, fused=True
    )
    optimizer2 = Muon(
        filter_params,
        lr=0.205,
        momentum=0.655,
        nesterov=True,
        norm_freq=4,
        total_train_steps=total_train_steps,
        weight_decay=wd,
        orthogonalization_backend=ortho_backend,
        resolvent_terms=resolvent_terms,
        resolvent_ridge=resolvent_ridge,
        neumann_terms=neumann_terms,
    )
    optimizer2.param_groups[0]["momentum_buffer_dtype"] = torch.half
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    # For accurately timing GPU code
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
    with torch.no_grad():
        train_images = train_loader.normalize(train_loader.images[:960])
        model.init_whiten(train_images)

    # Precompute LR factors to reduce computation in training loop
    lr_factor1_base = 1.0 / max(1, whiten_bias_train_steps)
    lr_factor2_base = 1.0 / total_train_steps

    # Precompute some values to reduce computation in training loop
    lr_factor1_initial = optimizer1.param_groups[0]["initial_lr"]
    lr_factors2_initial = [group["initial_lr"] for group in optimizer1.param_groups[1:] + optimizer2.param_groups]

    # Compile the forward pass function with reduced overhead
    @torch.compile(mode="max-autotune", fullgraph=True)
    def forward_step(inputs, labels, whiten_bias_grad):
        outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
        loss = F.cross_entropy(outputs, labels, label_smoothing=0.09, reduction="sum")
        return loss

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        ####################
        #     Training     #
        ####################
        model.train()
        for inputs, labels in train_loader:
            # Determine if we should train whiten bias
            whiten_bias_grad = step < whiten_bias_train_steps
            
            # Execute training step
            loss = forward_step(inputs, labels, whiten_bias_grad)
            loss.backward()

            # Update learning rates more efficiently
            lr_factor1 = 1 - step * lr_factor1_base
            lr_factor2 = 1 - step * lr_factor2_base

            # Apply learning rates in a fused way
            optimizer1.param_groups[0]["lr"] = lr_factor1_initial * lr_factor1
            for i, group in enumerate(optimizer1.param_groups[1:] + optimizer2.param_groups):
                group["lr"] = lr_factors2_initial[i] * lr_factor2

            # Optimizer steps
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            if step >= total_train_steps:
                break
        if step >= total_train_steps:
            break

    ####################
    #  TTA Evaluation  #
    ####################

    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = "eval"
    train_acc = evaluate(model, train_loader, tta_level=0)
    val_acc = evaluate(model, test_loader, tta_level=0)
    print_training_details(locals(), is_final_entry=True)
    return (val_acc, tta_val_acc, time_seconds)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ortho-backend",
        choices=["newtonschulz", "resolvent"],
        default=DEFAULT_ORTHO_BACKEND,
    )
    parser.add_argument(
        "--resolvent-terms",
        type=int,
        choices=sorted(RESOLVENT_GAUSS_LEGENDRE_TABLES),
        default=DEFAULT_RESOLVENT_TERMS,
    )
    parser.add_argument(
        "--resolvent-ridge",
        type=float,
        default=DEFAULT_RESOLVENT_RIDGE,
    )
    parser.add_argument(
        "--neumann-terms",
        type=int,
        default=DEFAULT_NEUMANN_TERMS,
    )
    parser.add_argument(
        "--validate-orthogonalizer",
        action="store_true",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
    )
    parser.add_argument(
        "--inter-run-sleep-cycles",
        type=int,
        default=6000000000,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("cifar10_speedrun.py requires CUDA for training and benchmarking.")
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")
    if args.validate_orthogonalizer and args.ortho_backend == "resolvent":
        filter_shapes = [
            (p.shape[0], p.data.numel() // p.shape[0])
            for p in model.parameters()
            if len(p.shape) == 4 and p.requires_grad
        ]
        validate_resolvent_orthogonalizer(
            filter_shapes,
            num_terms=args.resolvent_terms,
            ridge_epsilon=args.resolvent_ridge,
            device=model.head.weight.device,
        )
    print(
        f"Muon orthogonalizer: {args.ortho_backend} "
        f"| resolvent_terms={args.resolvent_terms} "
        f"| resolvent_ridge={args.resolvent_ridge:.1e}"
        f" | neumann_terms={args.neumann_terms}"
    )
    print_columns(logging_columns_list, is_head=True)
    if not args.skip_warmup:
        main(
            "warmup",
            model,
            ortho_backend=args.ortho_backend,
            resolvent_terms=args.resolvent_terms,
            resolvent_ridge=args.resolvent_ridge,
            neumann_terms=args.neumann_terms,
        )
    results = []
    for run in range(args.benchmark_runs):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if args.inter_run_sleep_cycles > 0:
            torch.cuda._sleep(int(args.inter_run_sleep_cycles))
        val_acc, tta_val_acc, time_seconds = main(
            run + 1,
            model,
            ortho_backend=args.ortho_backend,
            resolvent_terms=args.resolvent_terms,
            resolvent_ridge=args.resolvent_ridge,
            neumann_terms=args.neumann_terms,
        )
        results.append((val_acc, tta_val_acc, time_seconds))
        accs_so_far = [a for _, a, _ in results]
        times_so_far = [t for _, _, t in results]
        print(
            f"Mean accuracy after {run + 1} runs: {sum(accs_so_far) / len(accs_so_far):.6f} | Mean time: {sum(times_so_far) / len(times_so_far):.6f}s", end='\r', flush=True
        )
    if results:
        print()
        _, accs, times = zip(*results)
        accs = torch.tensor(accs)
        times = torch.tensor(times)
        print("Accuracies: Mean: %.6f    Std: %.6f" % (accs.mean(), accs.std()))
        print("Times (s):  Mean: %.6f    Std: %.6f" % (times.mean(), times.std()))
