import os
import torch
import safetensors.torch
import folder_paths
import comfy.sd
import comfy.model_management

class OASingleFileLoader:
    def __init__(self):
        self.data_handle = None
        self.current_mmap_data = None
        self.cpu_cache = {}
        self.max_buffers = [None, None]   # 双 GPU 缓冲区
        self.copy_events = [None, None]   # 每个缓冲区对应的复制完成事件
        self.timing_enabled = False
        self.step_timing = {"cpu_read":0.0, "transfer":0.0, "compute":0.0, "total":0.0, "count":0}
        self.preload_all = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "skeleton_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_oa_single_file"
    CATEGORY = "OA_Custom_Nodes"

    def load_oa_single_file(self, ckpt_name, skeleton_name):
        if os.path.exists(ckpt_name):
            ckpt_path = ckpt_name
        else:
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到 OA 压缩模型: {ckpt_name}")

        if os.path.exists(skeleton_name):
            skeleton_path = skeleton_name
        else:
            skeleton_path = folder_paths.get_full_path("checkpoints", skeleton_name)
        if not skeleton_path or not os.path.exists(skeleton_path):
            raise FileNotFoundError(f"找不到骨架模型: {skeleton_name}")

        self.current_mmap_data = safetensors.torch.safe_open(ckpt_path, framework="pt", device="cpu")

        # 预加载所有压缩数据到 CPU pinned memory
        if self.preload_all:
            self._preload_all_data()

        model, clip, vae = self._load_skeleton_and_hook(skeleton_path)
        return (model, clip, vae)

    def _preload_all_data(self):
        """遍历所有压缩层，读取数据并 pin_memory 存入 cpu_cache"""
        print("[OASingleFileLoader] 预加载所有压缩数据到 CPU pinned memory...")
        keys = list(self.current_mmap_data.keys())
        count = 0
        for key in keys:
            if key.endswith(".block_mean"):
                base = key[:-len(".block_mean")]
                try:
                    mean_block = self.current_mmap_data.get_tensor(key).pin_memory()
                    res_fp8 = self.current_mmap_data.get_tensor(base + ".block_residual").pin_memory()
                    scale_block = self.current_mmap_data.get_tensor(base + ".block_scale").pin_memory()
                    outlier_idx = self.current_mmap_data.get_tensor(base + ".outlier_indices").pin_memory().long()
                    outlier_val = self.current_mmap_data.get_tensor(base + ".outlier_values").pin_memory()
                    orig_shape = tuple(self.current_mmap_data.get_tensor(base + ".original_shape").tolist())
                    pad_len = self.current_mmap_data.get_tensor(base + ".pad_len").item()
                    self.cpu_cache[base] = (mean_block, res_fp8, scale_block, outlier_idx, outlier_val, orig_shape, pad_len)
                    count += 1
                except Exception as e:
                    print(f"  [警告] 预加载 {base} 失败: {e}")
        print(f"[OASingleFileLoader] 预加载完成，共 {count} 层")

    def _load_skeleton_and_hook(self, skeleton_path):
        out = comfy.sd.load_checkpoint_guess_config(
            skeleton_path, output_vae=True, output_clip=True
        )
        if isinstance(out, (list, tuple)):
            model, clip, vae = out[0], out[1], out[2]
        else:
            model, clip, vae = out

        diffusion_model = model.model.diffusion_model

        # 强制 float16，避免 dtype 冲突
        diffusion_model.to(torch.float16)
        print("[OASingleFileLoader] 已将 diffusion_model 转为 float16")

        # 包装 forward，统一输入 dtype 为 float16
        original_forward = diffusion_model.forward
        def wrapped_forward(x, timesteps=None, context=None, y=None, control=None, transformer_options=None, **kwargs):
            if isinstance(x, torch.Tensor) and x.dtype != torch.float16:
                x = x.to(torch.float16)
            if isinstance(timesteps, torch.Tensor) and timesteps.dtype != torch.float16:
                timesteps = timesteps.to(torch.float16)
            if isinstance(context, torch.Tensor) and context.dtype != torch.float16:
                context = context.to(torch.float16)
            if isinstance(y, torch.Tensor) and y.dtype != torch.float16:
                y = y.to(torch.float16)
            return original_forward(x, timesteps, context, y, control, transformer_options, **kwargs)
        diffusion_model.forward = wrapped_forward

        # 预分配双 GPU 缓冲区
        self._prepare_gpu_buffers()

        self._attach_oa_hooks(diffusion_model)
        return model, clip, vae

    def _prepare_gpu_buffers(self):
        """根据预加载的缓存计算最大元素数，并预分配两个 GPU 缓冲区"""
        max_numel = 0
        for (_, _, _, _, _, orig_shape, _) in self.cpu_cache.values():
            numel = 1
            for s in orig_shape:
                numel *= s
            if numel > max_numel:
                max_numel = numel
        if max_numel > 0:
            self.max_buffers[0] = torch.empty(max_numel, dtype=torch.float16, device="cuda")
            self.max_buffers[1] = torch.empty(max_numel, dtype=torch.float16, device="cuda")
            self.copy_events[0] = torch.cuda.Event()
            self.copy_events[1] = torch.cuda.Event()
            print(f"[OASingleFileLoader] 预分配双 GPU 缓冲区: 每个 {max_numel/1024**2:.2f} MB")
        else:
            print("[OASingleFileLoader] 警告：未找到压缩层，无法分配缓冲区")

    def _attach_oa_hooks(self, diffusion_model):
        # 建立层顺序列表
        layer_order = []
        for name, submodule in diffusion_model.named_modules():
            base_key = f"model.diffusion_model.{name}.weight" if name else "model.diffusion_model.weight"
            if base_key in self.cpu_cache:
                layer_order.append((base_key, submodule))

        print(f"[OASingleFileLoader] 注册 {len(layer_order)} 个压缩层 Hook，启用双缓冲预取")

        for idx, (base_key, submodule) in enumerate(layer_order):
            def make_hooks(base_key, layer_idx):
                def pre_hook(mod, input):
                    dev = mod.weight.device
                    buf_idx = layer_idx % 2
                    buf = self.max_buffers[buf_idx]
                    event = self.copy_events[buf_idx]

                    try:
                        (mean_block, res_fp8, scale_block, outlier_idx, outlier_val, orig_shape, pad_len) = self.cpu_cache[base_key]

                        # 等待该缓冲区上一个复制事件完成（确保可覆盖）
                        if event is not None:
                            event.synchronize()

                        # 异步传输当前层数据到 GPU 临时位置
                        mean_gpu = mean_block.to(dev, non_blocking=True)
                        res_gpu = res_fp8.to(dev, dtype=torch.float32, non_blocking=True)
                        scale_gpu = scale_block.to(dev, non_blocking=True)
                        idx_gpu = outlier_idx.to(dev, non_blocking=True)
                        val_gpu = outlier_val.to(dev, non_blocking=True)

                        # 解压计算
                        block_size = res_gpu.shape[1]
                        mean_flat = mean_gpu.float().expand(-1, block_size).flatten()
                        res_flat = (res_gpu * scale_gpu.float()).flatten()
                        if idx_gpu.numel() > 0:
                            idx = idx_gpu[:, 0] if idx_gpu.dim() > 1 else idx_gpu
                            res_flat[idx] = val_gpu.float()
                        flat = mean_flat + res_flat
                        if pad_len > 0:
                            flat = flat[:-pad_len]

                        # 复制到当前缓冲区
                        flat_contiguous = flat.contiguous()
                        n = flat_contiguous.numel()
                        buf[:n].copy_(flat_contiguous, non_blocking=True)
                        event.record()

                        # 等待复制完成，确保 forward 前数据就绪
                        event.synchronize()

                        weight = buf[:n].view(orig_shape)
                        mod.weight.data = weight

                    except Exception as e:
                        print(f"[OASingleFileLoader] 层 {base_key} 解压失败: {e}")
                    return None
                return pre_hook

            submodule.register_forward_pre_hook(make_hooks(base_key, idx))

NODE_CLASS_MAPPINGS = {"OASingleFileLoader": OASingleFileLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"OASingleFileLoader": "OA-FP8 2.5GB Limit Loader"}