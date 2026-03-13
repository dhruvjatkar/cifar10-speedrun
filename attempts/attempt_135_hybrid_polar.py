#############################################
#                  Setup                    #
#############################################

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

#############################################
#               Muon optimizer              #
#############################################

HYBRID_WARM_STEPS = 1  # A100-fast. Set to 2 if you want the safer/slower version.

@torch.compile(mode="max-autotune", fullgraph=False)
def _hybrid_polar_bucket(
    X: torch.Tensor,              # [B, D, K], padded inside a shape bucket
    progress_ratio: torch.Tensor, # scalar tensor on CUDA, avoids recompiling every step
) -> torch.Tensor:
    # Warm polynomial: same coefficients as the current Muon NS path.
    a, b, c = (3.4576, -4.7391, 2.0843)

    # 2-shift rational tail fitted to x^{-1/2} on [0.5, 1.5].
    # Final update is H(A) @ X with A = X X^T in the smaller row-space.
    m0 = 0.270818692
    w1 = 0.318805109
    w2 = 0.843671028
    s1 = 0.060984794
    s2 = 0.968423004

    eps_norm = 1e-5

    # Preserve the old magnitude schedule, but keep reductions in FP32.
    target_magnitude = 0.5012 + (0.0786 - 0.5012) * progress_ratio
    batch_mags = X.float().norm(dim=(1, 2), keepdim=True)
    X = X * (target_magnitude / (batch_mags + eps_norm)).to(X.dtype)
    X = X / (X.float().norm(dim=(1, 2), keepdim=True) + eps_norm).to(X.dtype)

    transposed = False
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2).contiguous()
        transposed = True

    # 1 cheap warm step in tensor-core dtype.
    for _ in range(HYBRID_WARM_STEPS):
        A = X @ X.transpose(1, 2)
        A2 = A @ A
        X = a * X + (b * A + c * A2) @ X

    # Rational tail in FP32 row-space.
    orig_dtype = X.dtype
    Xf = X.float()
    A = Xf @ Xf.transpose(1, 2)
    A = 0.5 * (A + A.transpose(1, 2))

    B, D, _ = A.shape
    I = torch.eye(D, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(B, 1, 1)

    # Small extra ridge helps padded zero rows and the earliest steps.
    ridge = 2.5e-4 * (1.0 - progress_ratio) + 1.0e-5

    Z1 = torch.cholesky_solve(
        I,
        torch.linalg.cholesky(A + (s1 + ridge) * I),
    )
    Z2 = torch.cholesky_solve(
        I,
        torch.linalg.cholesky(A + (s2 + ridge) * I),
    )

    H = m0 * I + w1 * Z1 + w2 * Z2
    X = (H @ Xf).to(orig_dtype)

    if transposed:
        X = X.transpose(1, 2).contiguous()
    return X


def _hybrid_zeropower_bucketed(
    gradients_4d: list[torch.Tensor],
    filter_meta_data: list[tuple],
    max_D: int,
    max_K: int,
    progress_ratio: torch.Tensor,
) -> list[torch.Tensor]:
    if not filter_meta_data:
        return []

    grad_list = []
    for original_shape, reshaped_D, reshaped_K, list_idx in filter_meta_data:
        g = gradients_4d[list_idx].reshape(reshaped_D, reshaped_K)
        g = F.pad(g, (0, max_K - reshaped_K, 0, max_D - reshaped_D), "constant", 0)
        grad_list.append(g)

    X = torch.stack(grad_list, dim=0).contiguous()
    X = _hybrid_polar_bucket(X, progress_ratio)

    out = [None] * len(filter_meta_data)
    for i, (original_shape, reshaped_D, reshaped_K, list_idx) in enumerate(filter_meta_data):
        out[list_idx] = X[i, :reshaped_D, :reshaped_K].view(original_shape)
    return out


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
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            norm_freq=norm_freq,
            total_train_steps=total_train_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.step_count = 0
        self.last_norm_step = 0
        self.total_train_steps = total_train_steps
        self.current_grad_norms = None

        self.filter_params_meta = []
        self.bucket_specs = {}

        for group in self.param_groups:
            for p in group["params"]:
                if len(p.shape) == 4 and p.requires_grad:
                    reshaped_D = p.shape[0]
                    reshaped_K = p.data.numel() // p.shape[0]

                    # Bucket by orientation + smaller dimension.
                    # For your CIFAR conv stack this becomes two buckets: 64 and 256.
                    bucket_key = (reshaped_D > reshaped_K, min(reshaped_D, reshaped_K))
                    bucket = self.bucket_specs.setdefault(
                        bucket_key, {"max_D": 0, "max_K": 0}
                    )
                    bucket["max_D"] = max(bucket["max_D"], reshaped_D)
                    bucket["max_K"] = max(bucket["max_K"], ((reshaped_K + 15) // 16) * 16)

                    self.filter_params_meta.append(
                        {
                            "param": p,
                            "original_shape": p.data.shape,
                            "reshaped_dims": (reshaped_D, reshaped_K),
                            "bucket_key": bucket_key,
                        }
                    )

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        group = self.param_groups[0]

        progress = self.step_count / max(1, self.total_train_steps)
        group["norm_freq"] = 2 + int(15 * progress)

        filter_params_with_grad = []
        step_buckets = {}
        momentum_buffers = [] if group["momentum_buffer_dtype"] == torch.half else None

        for p_meta in self.filter_params_meta:
            p = p_meta["param"]
            if p.grad is None:
                continue

            global_idx = len(filter_params_with_grad)
            filter_params_with_grad.append(p)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad,
                    dtype=group["momentum_buffer_dtype"],
                    memory_format=torch.preserve_format,
                )

            if momentum_buffers is not None:
                momentum_buffers.append(state["momentum_buffer"])

            bucket_key = p_meta["bucket_key"]
            bucket = step_buckets.setdefault(bucket_key, {"global_indices": [], "meta": []})
            local_idx = len(bucket["global_indices"])
            bucket["global_indices"].append(global_idx)
            bucket["meta"].append(
                (
                    p_meta["original_shape"],
                    p_meta["reshaped_dims"][0],
                    p_meta["reshaped_dims"][1],
                    local_idx,
                )
            )

        if not filter_params_with_grad:
            return

        if momentum_buffers is not None:
            torch._foreach_mul_(momentum_buffers, group["momentum"])
            grad_casts = [p.grad.to(mb.dtype) for p, mb in zip(filter_params_with_grad, momentum_buffers)]
            torch._foreach_add_(momentum_buffers, grad_casts)
        else:
            momentum_buffers = [p.grad for p in filter_params_with_grad]

        if group["nesterov"]:
            nesterov_grads = torch._foreach_add(
                [p.grad for p in filter_params_with_grad],
                momentum_buffers,
                alpha=group["momentum"],
            )
        else:
            nesterov_grads = momentum_buffers

        do_norm_scaling = (self.step_count - self.last_norm_step >= group["norm_freq"])
        if do_norm_scaling:
            self.last_norm_step = self.step_count
            self.current_grad_norms = torch._foreach_norm(filter_params_with_grad)
            scale_factors = [
                (len(p.data) ** 0.5 / (n + 1e-07)).to(p.data.dtype)
                for p, n in zip(filter_params_with_grad, self.current_grad_norms)
            ]

        # Tensor scalar so the compiled kernel does not specialize on Python ints every step.
        progress_ratio = torch.tensor(
            progress,
            device=filter_params_with_grad[0].device,
            dtype=torch.float32,
        )

        final_orthogonalized_grads = [None] * len(filter_params_with_grad)
        for bucket_key, bucket in step_buckets.items():
            bucket_grads = [nesterov_grads[idx] for idx in bucket["global_indices"]]
            bucket_out = _hybrid_zeropower_bucketed(
                bucket_grads,
                bucket["meta"],
                self.bucket_specs[bucket_key]["max_D"],
                self.bucket_specs[bucket_key]["max_K"],
                progress_ratio,
            )
            for global_idx, g in zip(bucket["global_indices"], bucket_out):
                final_orthogonalized_grads[global_idx] = g

        if do_norm_scaling:
            torch._foreach_mul_(filter_params_with_grad, scale_factors)
            torch._foreach_add_(
                filter_params_with_grad,
                final_orthogonalized_grads,
                alpha=-group["lr"],
            )
        else:
            torch._foreach_add_(
                filter_params_with_grad,
                final_orthogonalized_grads,
                alpha=-group["lr"],
            )

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

def main(run, model):
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

if __name__ == "__main__":
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")
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
            f"Mean accuracy after {run + 1} runs: {sum(accs_so_far) / len(accs_so_far):.6f} | Mean time: {sum(times_so_far) / len(times_so_far):.6f}s", end='\r', flush=True
        )
    print()
    _, accs, times = zip(*results)
    accs = torch.tensor(accs)
    times = torch.tensor(times)
    print("Accuracies: Mean: %.6f    Std: %.6f" % (accs.mean(), accs.std()))
    print("Times (s):  Mean: %.6f    Std: %.6f" % (times.mean(), times.std()))
