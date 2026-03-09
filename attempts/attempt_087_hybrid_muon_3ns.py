#############################################
#                  Setup                    #
#############################################

import os
import sys
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

with open(sys.argv[0]) as f:
    code = f.read()

torch.backends.cudnn.benchmark = True

#############################################
#        Hybrid Muon for CIFAR speedrun     #
#############################################

# Hybrid Muon with 3 NS iterations + row preconditioner.
# Key changes from baseline Muon:
# - Row preconditioner before polynomial iteration (cheap spectrum flattening)
# - Parameters bucketed by D for static batch shapes
# - Periodic renorm + adaptive norm_freq preserved from baseline
# - Magnitude scheduling removed (row preconditioner replaces it)

POLAR_A = 3.4576
POLAR_B = -4.7391
POLAR_C = 2.0843


@torch.compile(fullgraph=True, mode="max-autotune")
def _wide_polar_fast(
    X: torch.Tensor,
    current_step: int,
    total_steps: int,
) -> torch.Tensor:
    # X: [B, D, K]
    eps = 1e-6

    # 1) target magnitude scheduling (same as baseline)
    progress = current_step / max(1, total_steps)
    initial_mag = 0.5012
    final_mag = 0.0786
    target_mag = initial_mag * (1 - progress) + final_mag * progress

    # 2) row preconditioner (cheap spectrum flattening)
    row_norm_sq = X.square().sum(dim=2, keepdim=True).clamp_min(eps)
    X = X * row_norm_sq.rsqrt()

    # 3) scale to target magnitude + Frobenius normalize for stable iteration
    fro = X.norm(dim=(1, 2), keepdim=True)
    X = X * (target_mag / (fro + eps))
    X = X / (X.norm(dim=(1, 2), keepdim=True) + eps)

    # 4) three fast polynomial polar steps
    A = X @ X.transpose(1, 2)
    B = POLAR_B * A + POLAR_C * (A @ A)
    X = POLAR_A * X + B @ X

    A = X @ X.transpose(1, 2)
    B = POLAR_B * A + POLAR_C * (A @ A)
    X = POLAR_A * X + B @ X

    A = X @ X.transpose(1, 2)
    B = POLAR_B * A + POLAR_C * (A @ A)
    X = POLAR_A * X + B @ X
    return X


class HybridMuonCIFAR(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.205,
        momentum=0.655,
        nesterov=True,
        weight_decay=0.0,
        total_train_steps=None,
        momentum_buffer_dtype=torch.half,
        norm_freq=4,
    ):
        params = list(params)
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            total_train_steps=total_train_steps,
            momentum_buffer_dtype=momentum_buffer_dtype,
            norm_freq=norm_freq,
        )
        super().__init__(params, defaults)
        self.step_count = 0
        self.last_renorm_step = 0
        self.total_train_steps = total_train_steps
        self.norm_freq = norm_freq

        # Bucket 4D conv weights by output dimension D for static batch shapes
        buckets = {}
        for p in params:
            if not p.requires_grad or p.ndim != 4:
                continue
            D = p.shape[0]
            K = p.numel() // D
            buckets.setdefault(D, []).append((p, K, tuple(p.shape)))

        self.buckets = []
        for D in sorted(buckets):
            entries = buckets[D]
            max_K = max(K for _, K, _ in entries)
            max_K = (max_K + 31) // 32 * 32
            device = entries[0][0].device
            batch = torch.zeros(
                len(entries), D, max_K,
                device=device, dtype=torch.half,
            )
            self.buckets.append({
                "D": D,
                "max_K": max_K,
                "entries": entries,
                "batch": batch,
            })

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        group = self.param_groups[0]
        mu = group["momentum"]
        lr = group["lr"]
        wd = group["weight_decay"]
        nesterov = group["nesterov"]
        mb_dtype = group["momentum_buffer_dtype"]

        # Adaptive norm_freq schedule (same as baseline)
        progress = self.step_count / self.total_train_steps
        self.norm_freq = 2 + int(15 * progress)

        decay = 1.0 - lr * wd

        # Periodic renorm (restored from baseline)
        do_renorm = (self.step_count - self.last_renorm_step) >= self.norm_freq
        if do_renorm:
            self.last_renorm_step = self.step_count
            for bucket in self.buckets:
                D = bucket["D"]
                for p, K, original_shape in bucket["entries"]:
                    if p.grad is None:
                        continue
                    scale = (D ** 0.5) / (p.norm() + 1e-7)
                    p.mul_(scale.to(dtype=p.dtype))

        for bucket in self.buckets:
            D = bucket["D"]
            batch = bucket["batch"]
            batch.zero_()

            active = []
            for i, (p, K, original_shape) in enumerate(bucket["entries"]):
                if p.grad is None:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, dtype=mb_dtype,
                        memory_format=torch.preserve_format,
                    )
                buf = state["momentum_buffer"]
                grad = p.grad

                buf.mul_(mu).add_(grad.to(dtype=buf.dtype))
                if nesterov:
                    upd = grad.to(dtype=buf.dtype) + mu * buf
                else:
                    upd = buf

                batch[i, :, :K].copy_(upd.to(dtype=batch.dtype).reshape(D, K))
                active.append((i, p, K, original_shape))

            if not active:
                continue

            Y = _wide_polar_fast(batch, self.step_count, self.total_train_steps)

            for i, p, K, original_shape in active:
                if decay != 1.0:
                    p.mul_(decay)
                p.add_(Y[i, :, :K].view(original_shape), alpha=-lr)

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
    base_y_coords = torch.arange(crop_size, device=images.device).view(1, 1, crop_size, 1)
    base_x_coords = torch.arange(crop_size, device=images.device).view(1, 1, 1, crop_size)
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
            data["images"], data["labels"], data["classes"],
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
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,)*4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]

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
        self.whiten = nn.Conv2d(
            3, whiten_width, whiten_kernel_size, padding=0, bias=True
        )
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
        est_patch_covariance = torch.mm(patches_flat.t(), patches_flat) / len(patches_flat)
        U, S, V = torch.svd(est_patch_covariance)
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
    else:
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
    optimizer2 = HybridMuonCIFAR(
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
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
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

    lr_factor1_base = 1.0 / max(1, whiten_bias_train_steps)
    lr_factor2_base = 1.0 / total_train_steps

    lr_factor1_initial = optimizer1.param_groups[0]["initial_lr"]
    lr_factors2_initial = [group["initial_lr"] for group in optimizer1.param_groups[1:] + optimizer2.param_groups]

    @torch.compile(mode="max-autotune", fullgraph=True)
    def forward_step(inputs, labels, whiten_bias_grad):
        outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
        loss = F.cross_entropy(outputs, labels, label_smoothing=0.09, reduction="sum")
        return loss

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        model.train()
        for inputs, labels in train_loader:
            whiten_bias_grad = step < whiten_bias_train_steps
            loss = forward_step(inputs, labels, whiten_bias_grad)
            loss.backward()

            lr_factor1 = 1 - step * lr_factor1_base
            lr_factor2 = 1 - step * lr_factor2_base

            optimizer1.param_groups[0]["lr"] = lr_factor1_initial * lr_factor1
            for i, group in enumerate(optimizer1.param_groups[1:] + optimizer2.param_groups):
                group["lr"] = lr_factors2_initial[i] * lr_factor2

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
    return (val_acc, tta_val_acc, time_seconds)

if __name__ == "__main__":
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")
    print_columns(logging_columns_list, is_head=True)
    main("warmup", model)
    results = []
    for run in range(50):
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
