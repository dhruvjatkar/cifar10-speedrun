#############################################
#   CIFAR-10 fair benchmark rewrite (PT2.4) #
#############################################
#
# Goals:
# - Keep the benchmark apples-to-apples with the original:
#   same model, same augmentation policy, same optimizer behavior,
#   same LR schedules, same evaluation protocol, same timing scope.
# - Make the implementation faster and more robust on
#   PyTorch 2.4.1 + CUDA graphs + max-autotune.
#
# Notable implementation changes vs. the original:
# - Muon orthogonalization buckets conv filters by exact (D, K) shape
#   instead of padding every filter to one global max shape.
# - Compiled helpers live at module scope so the warmup run amortizes
#   compile cost across all measured runs.
# - Evaluation/TTA clones outputs that must stay live across the next
#   compiled invocation, which avoids CUDA-graph output-overwrite errors
#   on PyTorch 2.4.x.
# - The uncertain-TTA revisit pass uses a fixed 1250 batch size on
#   CIFAR-10 test (10,000 images, 25% revisit => 2,500 uncertain images),
#   so it runs as two equal compiled batches instead of a 2000/500 tail.

from __future__ import annotations

import os
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision


torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    raise RuntimeError("This script expects a CUDA GPU.")

#############################################
#                 Constants                 #
#############################################

TRAIN_BATCH_SIZE = 1536
EVAL_BATCH_SIZE = 2000
TTA_PAD = 1
# CIFAR-10 test set has exactly 10,000 images. With 25% uncertain routing,
# the revisit set is always 2,500 images, so 1,250 keeps the TTA revisit pass
# fixed-shape on PyTorch 2.4.1.
TTA_UNCERTAIN_BATCH_SIZE = 1250

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.half, device="cuda").view(1, 3, 1, 1)
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616), dtype=torch.half, device="cuda").view(1, 3, 1, 1)

#############################################
#            Compiled helper ops            #
#############################################

@torch.compile(fullgraph=True)
def normalize_images(inputs: torch.Tensor) -> torch.Tensor:
    return (inputs - CIFAR_MEAN) / CIFAR_STD


@torch.compile(fullgraph=True)
def batch_color_jitter(inputs: torch.Tensor, brightness_range: float, contrast_range: float) -> torch.Tensor:
    batch = inputs.shape[0]
    device = inputs.device
    dtype = inputs.dtype
    brightness_shift = (
        torch.rand(batch, 1, 1, 1, device=device, dtype=dtype) * 2 - 1
    ) * brightness_range
    contrast_scale = (
        torch.rand(batch, 1, 1, 1, device=device, dtype=dtype) * 2 - 1
    ) * contrast_range + 1
    return (inputs + brightness_shift) * contrast_scale


@torch.compile(fullgraph=True)
def batch_flip_lr(inputs: torch.Tensor) -> torch.Tensor:
    flip_mask = (torch.rand(inputs.shape[0], device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


@torch.compile(fullgraph=True)
def batch_crop(images: torch.Tensor, crop_size: int) -> torch.Tensor:
    batch, channels, padded_h, padded_w = images.shape
    radius = (padded_h - crop_size) // 2

    y_offsets = (torch.rand(batch, device=images.device) * (2 * radius + 1)).long()
    x_offsets = (torch.rand(batch, device=images.device) * (2 * radius + 1)).long()

    base_y = torch.arange(crop_size, device=images.device).view(1, 1, crop_size, 1)
    base_x = torch.arange(crop_size, device=images.device).view(1, 1, 1, crop_size)

    y_indices = y_offsets.view(batch, 1, 1, 1) + base_y
    x_indices = x_offsets.view(batch, 1, 1, 1) + base_x
    y_indices = y_indices.expand(batch, channels, crop_size, crop_size)
    x_indices = x_indices.expand(batch, channels, crop_size, crop_size)

    batch_indices = torch.arange(batch, device=images.device).view(batch, 1, 1, 1).expand_as(y_indices)
    channel_indices = torch.arange(channels, device=images.device).view(1, channels, 1, 1).expand_as(y_indices)

    return images[batch_indices, channel_indices, y_indices, x_indices]


@torch.compile(fullgraph=True)
def muon_zeropower_newtonschulz5_bucket(
    stacked_grads: torch.Tensor,
    current_step: int,
    total_steps: int,
) -> torch.Tensor:
    a, b, c = (3.4576, -4.7391, 2.0843)
    eps_stable = 1e-5
    eps_gms = 1e-5

    progress_ratio = current_step / max(1, total_steps)
    initial_target_mag = 0.5012
    final_target_mag = 0.0786
    target_magnitude = initial_target_mag * (1 - progress_ratio) + final_target_mag * progress_ratio

    current_batch_mags = stacked_grads.norm(dim=(1, 2), keepdim=True)
    stacked_grads = stacked_grads * (target_magnitude / (current_batch_mags + eps_gms))

    stacked_grads = stacked_grads / (stacked_grads.norm(dim=(1, 2), keepdim=True) + eps_stable)

    transposed = False
    if stacked_grads.size(1) > stacked_grads.size(2):
        stacked_grads = stacked_grads.transpose(1, 2)
        transposed = True

    A = stacked_grads @ stacked_grads.transpose(1, 2)
    B = b * A + c * (A @ A)
    stacked_grads = a * stacked_grads + B @ stacked_grads

    A = stacked_grads @ stacked_grads.transpose(1, 2)
    B = b * A + c * (A @ A)
    stacked_grads = a * stacked_grads + B @ stacked_grads

    A = stacked_grads @ stacked_grads.transpose(1, 2)
    B = b * A + c * (A @ A)
    stacked_grads = a * stacked_grads + B @ stacked_grads

    if transposed:
        stacked_grads = stacked_grads.transpose(1, 2)

    return stacked_grads


@torch.compile(mode="max-autotune", fullgraph=True)
def compiled_train_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    whiten_bias_grad: bool,
) -> torch.Tensor:
    logits = model(inputs, whiten_bias_grad=whiten_bias_grad)
    return F.cross_entropy(logits, labels, label_smoothing=0.09, reduction="sum")


@torch.compile(mode="max-autotune", fullgraph=True)
def compiled_infer_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return model(inputs, whiten_bias_grad=False)


@torch.compile(mode="max-autotune", fullgraph=True)
def compiled_infer_mirror_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return 0.5 * model(inputs, whiten_bias_grad=False) + 0.5 * model(inputs.flip(-1), whiten_bias_grad=False)


@torch.compile(mode="max-autotune", fullgraph=True)
def compiled_tta_uncertain_logits(model: nn.Module, images_batch: torch.Tensor, pad: int) -> torch.Tensor:
    batch = images_batch.shape[0]
    padded_inputs = F.pad(images_batch, (pad,) * 4, "reflect")
    crop_tl = padded_inputs[:, :, 0:32, 0:32]
    crop_br = padded_inputs[:, :, 2:34, 2:34]
    base_views = torch.cat([images_batch, crop_tl, crop_br], dim=0)
    mirrored_views = base_views.flip(-1)
    combined_inputs = torch.cat([base_views, mirrored_views], dim=0)
    combined_logits = model(combined_inputs, whiten_bias_grad=False)
    num_views = combined_inputs.shape[0] // batch
    return combined_logits.view(num_views, batch, -1).mean(dim=0)

#############################################
#               Muon optimizer              #
#############################################

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
        momentum_buffer_dtype=torch.half,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            norm_freq=norm_freq,
            total_train_steps=total_train_steps,
            weight_decay=weight_decay,
            momentum_buffer_dtype=momentum_buffer_dtype,
        )
        super().__init__(params, defaults)

        self.step_count = 0
        self.last_norm_step = 0
        self.total_train_steps = total_train_steps

        bucket_map: dict[tuple[int, int], dict[str, list[torch.Tensor] | int]] = {}
        for group in self.param_groups:
            momentum_buffer_dtype = group["momentum_buffer_dtype"]
            for param in group["params"]:
                if len(param.shape) != 4 or not param.requires_grad:
                    continue

                rows = param.shape[0]
                cols = param.numel() // rows
                state = self.state[param]
                state["momentum_buffer"] = torch.zeros_like(
                    param,
                    dtype=momentum_buffer_dtype,
                    memory_format=torch.preserve_format,
                )

                bucket = bucket_map.setdefault(
                    (rows, cols),
                    {
                        "rows": rows,
                        "cols": cols,
                        "params": [],
                        "buffers": [],
                        "sqrt_rows": [],
                    },
                )
                bucket["params"].append(param)
                bucket["buffers"].append(state["momentum_buffer"])
                bucket["sqrt_rows"].append(torch.tensor(rows ** 0.5, device=param.device, dtype=param.dtype))

        self.buckets = list(bucket_map.values())

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        group = self.param_groups[0]
        progress = self.step_count / self.total_train_steps
        group["norm_freq"] = 2 + int(15 * progress)
        do_norm_scaling = self.step_count - self.last_norm_step >= group["norm_freq"]
        if do_norm_scaling:
            self.last_norm_step = self.step_count

        active_params: list[torch.Tensor] = []
        active_sqrt_rows: list[torch.Tensor] = []
        live_buckets = []

        for bucket in self.buckets:
            live_params = []
            live_buffers = []
            live_grads = []
            live_sqrt_rows = []

            for param, buffer, sqrt_row in zip(bucket["params"], bucket["buffers"], bucket["sqrt_rows"]):
                grad = param.grad
                if grad is None:
                    continue
                live_params.append(param)
                live_buffers.append(buffer)
                live_grads.append(grad)
                live_sqrt_rows.append(sqrt_row)

            if not live_params:
                continue

            active_params.extend(live_params)
            active_sqrt_rows.extend(live_sqrt_rows)
            live_buckets.append((bucket["rows"], bucket["cols"], live_params, live_buffers, live_grads))

        if not active_params:
            return

        # This intentionally scales the parameters, not the update.
        if do_norm_scaling:
            param_norms = torch._foreach_norm(active_params)
            scale_factors = [
                (sqrt_row / (norm + 1e-7)).to(param.dtype)
                for param, norm, sqrt_row in zip(active_params, param_norms, active_sqrt_rows)
            ]
            torch._foreach_mul_(active_params, scale_factors)

        for rows, cols, live_params, live_buffers, live_grads in live_buckets:
            torch._foreach_mul_(live_buffers, group["momentum"])
            torch._foreach_add_(live_buffers, live_grads)

            if group["nesterov"]:
                nesterov_grads = torch._foreach_add(live_grads, live_buffers, alpha=group["momentum"])
            else:
                nesterov_grads = live_buffers

            stacked = torch.stack([grad.reshape(rows, cols) for grad in nesterov_grads], dim=0)
            stacked_updates = muon_zeropower_newtonschulz5_bucket(
                stacked,
                self.step_count,
                self.total_train_steps,
            )
            updates = [stacked_updates[i].reshape_as(param) for i, param in enumerate(live_params)]
            torch._foreach_add_(live_params, updates, alpha=-group["lr"])

        weight_decay_factor = 1 - group["lr"] * group["weight_decay"]
        if weight_decay_factor != 1.0:
            torch._foreach_mul_(active_params, weight_decay_factor)

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        if param.grad.grad_fn is not None:
                            param.grad.detach_()
                        else:
                            param.grad.requires_grad_(False)
                        param.grad.zero_()

#############################################
#                DataLoader                 #
#############################################

class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dataset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dataset.data)
            labels = torch.tensor(dataset.targets)
            torch.save({"images": images, "labels": labels, "classes": dataset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device("cuda"), weights_only=True)
        self.images = data["images"]
        self.labels = data["labels"]
        self.classes = data["classes"]

        self.images = (
            (self.images.half() / 255)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )

        self.proc_images: dict[str, torch.Tensor] = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train
        self._indices = torch.empty(len(self.images), dtype=torch.long, device="cuda")
        self._ordered_indices = torch.arange(len(self.images), dtype=torch.long, device="cuda")

    def __len__(self):
        if self.drop_last:
            return len(self.images) // self.batch_size
        return ceil(len(self.images) / self.batch_size)

    def normalized_images(self) -> torch.Tensor:
        images = self.proc_images.get("norm")
        if images is None:
            images = normalize_images(self.images)
            self.proc_images["norm"] = images
        return images

    def __iter__(self):
        pad = self.aug.get("translate", 0)

        if self.epoch == 0:
            images = self.normalized_images()
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")

        if pad > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]

        if self.aug.get("flip", False) and self.epoch % 2 == 1:
            images = images.flip(-1)

        color_jitter_cfg = self.aug.get("color_jitter", {"enabled": False})
        if color_jitter_cfg.get("enabled", False):
            images = batch_color_jitter(
                images,
                color_jitter_cfg.get("brightness_range", 0.1),
                color_jitter_cfg.get("contrast_range", 0.1),
            )

        self.epoch += 1

        if self.shuffle:
            torch.randperm(len(self.images), out=self._indices)
            indices = self._indices
        else:
            indices = self._ordered_indices

        for batch_idx in range(len(self)):
            idxs = indices[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
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
        weight = self.weight.data
        torch.nn.init.dirac_(weight[: weight.size(1)])


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

        for module in self.modules():
            module.half()
        self.to(memory_format=torch.channels_last)

    def reset(self):
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        head_weight = self.head.weight.data
        head_weight.mul_(1.0 / head_weight.std())

    def init_whiten(self, train_images, eps=0.0005):
        channels, (kernel_h, kernel_w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = (
            train_images.unfold(2, kernel_h, 1)
            .unfold(3, kernel_w, 1)
            .transpose(1, 3)
            .reshape(-1, channels, kernel_h, kernel_w)
            .float()
        )
        patches_flat = patches.view(len(patches), -1)
        patch_covariance = torch.mm(patches_flat.t(), patches_flat) / len(patches_flat)
        U, S, _ = torch.svd(patch_covariance)
        inv_sqrt_S = torch.rsqrt(S + eps)
        eigenvectors_scaled = (U * inv_sqrt_S.unsqueeze(0)).T.reshape(-1, channels, kernel_h, kernel_w)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        x = x.to(memory_format=torch.channels_last)
        bias = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, bias if whiten_bias_grad else bias.detach())
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1).contiguous()
        return self.head(x) / x.shape[-1]

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
#               Evaluation                 #
############################################

@torch.inference_mode()
def infer(model, loader, tta_level=0):
    test_images = loader.normalized_images()
    model.eval()

    if tta_level == 0:
        parts = []
        for inputs in test_images.split(EVAL_BATCH_SIZE):
            inputs = inputs.contiguous(memory_format=torch.channels_last)
            # Clone because compiled CUDA-graph outputs must not remain live
            # across the next compiled invocation on PyTorch 2.4.x.
            parts.append(compiled_infer_logits(model, inputs).clone())
        return torch.cat(parts, dim=0)

    if tta_level == 1:
        parts = []
        for inputs in test_images.split(EVAL_BATCH_SIZE):
            inputs = inputs.contiguous(memory_format=torch.channels_last)
            parts.append(compiled_infer_mirror_logits(model, inputs).clone())
        return torch.cat(parts, dim=0)

    initial_parts = []
    for inputs in test_images.split(EVAL_BATCH_SIZE):
        inputs = inputs.contiguous(memory_format=torch.channels_last)
        initial_parts.append(compiled_infer_logits(model, inputs).clone())
    initial_logits = torch.cat(initial_parts, dim=0)

    probs = F.softmax(initial_logits, dim=1)
    confidences, _ = probs.max(dim=1)
    k_uncertain = int(test_images.shape[0] * 0.25)
    _, uncertain_indices = torch.topk(confidences, k_uncertain, largest=False, sorted=False)

    tta_parts = []
    for start in range(0, k_uncertain, TTA_UNCERTAIN_BATCH_SIZE):
        batch_indices = uncertain_indices[start : start + TTA_UNCERTAIN_BATCH_SIZE]
        images_batch = test_images[batch_indices].contiguous(memory_format=torch.channels_last)
        tta_parts.append(compiled_tta_uncertain_logits(model, images_batch, TTA_PAD).clone())

    if tta_parts:
        initial_logits[uncertain_indices] = torch.cat(tta_parts, dim=0)

    return initial_logits


@torch.inference_mode()
def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

def main(run, model):
    bias_lr = 0.0573
    head_lr = 0.5415
    wd = 1.0418e-06 * TRAIN_BATCH_SIZE

    test_loader = CifarLoader("cifar10", train=False, batch_size=EVAL_BATCH_SIZE)
    train_loader = CifarLoader(
        "cifar10",
        train=True,
        batch_size=TRAIN_BATCH_SIZE,
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
        train_loader.proc_images.clear()
        test_loader.labels = torch.randint(0, 10, size=(len(test_loader.labels),), device=test_loader.labels.device)
        test_loader.images = torch.randn_like(test_loader.images, device=test_loader.images.device)
        test_loader.proc_images.clear()

    total_train_steps = ceil(7.65 * len(train_loader))
    whiten_bias_train_steps = ceil(0.2 * len(train_loader))

    model.reset()

    filter_params = [param for param in model.parameters() if len(param.shape) == 4 and param.requires_grad]
    norm_biases = [param for name, param in model.named_parameters() if "norm" in name and param.requires_grad]

    optimizer1 = torch.optim.SGD(
        [
            dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
            dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
            dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
        ],
        momentum=0.825,
        nesterov=True,
        fused=True,
    )
    optimizer2 = Muon(
        filter_params,
        lr=0.205,
        momentum=0.655,
        nesterov=True,
        norm_freq=4,
        total_train_steps=total_train_steps,
        weight_decay=wd,
        momentum_buffer_dtype=torch.half,
    )
    optimizers = [optimizer1, optimizer2]

    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

    bias_group = optimizer1.param_groups[0]
    bias_group_initial_lr = bias_group["initial_lr"]
    scheduled_groups = optimizer1.param_groups[1:] + optimizer2.param_groups
    scheduled_initial_lrs = [group["initial_lr"] for group in scheduled_groups]

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
        model.init_whiten(train_loader.normalized_images()[:960])

    bias_lr_decay = 1.0 / max(1, whiten_bias_train_steps)
    main_lr_decay = 1.0 / total_train_steps

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        model.train()
        for inputs, labels in train_loader:
            whiten_bias_grad = step < whiten_bias_train_steps

            loss = compiled_train_loss(model, inputs, labels, whiten_bias_grad)
            loss.backward()

            bias_group["lr"] = bias_group_initial_lr * (1 - step * bias_lr_decay)
            main_lr_factor = 1 - step * main_lr_decay
            for group, initial_lr in zip(scheduled_groups, scheduled_initial_lrs):
                group["lr"] = initial_lr * main_lr_factor

            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

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
        accs_so_far = [acc for _, acc, _ in results]
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
