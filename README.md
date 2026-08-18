# OA-FP8：面向 SDXL 的低显存压缩与动态加载方案
一种结合分块均值残差、异常值隔离与 FP8 量化的 SDXL 模型压缩方法，并配套 ComfyUI 动态解压加载器，实现 8GB 显卡低显存运行。

📌 项目简介
OA-FP8（Outlier-Aware FP8）是一种针对 Stable Diffusion XL (SDXL) 权重的有损压缩方案。
它在标准 FP8 量化基础上，通过以下技术降低量化误差：

分块均值残差：提取每个权重块的均值，仅对残差进行 FP8 量化，有效降低动态范围需求。

块内异常值隔离：以逐块标准差为阈值（σ=3.2），将极端残差单独用 FP16 保存，避免影响正常值量化。

敏感层保护：Norm、Bias、Embedding、LayerNorm、时间嵌入等层保持 FP16 不量化。

动态加载器：在 ComfyUI 中按需解压权重，保持低显存占用，接近 FP16 生成质量。

✨ 特性
压缩比约 1.69x（6.46 GB → 3.81 GB）

生成质量显著优于直接 FP8：

PSNR：21.09 dB（直接 FP8 为 16.90 dB）

SSIM：0.7383（直接 FP8 为 0.6023）

MSE：0.007971（直接 FP8 为 0.022586）

推理显存极低：动态解压过程中 allocated 约 84 MB，reserved 峰值约 1.2 GB

速度可接受：20 步采样约 36 秒（单缓冲稳定版），或优化版稳定步约 1.59 s/step

生成图像与原始 FP16 模型肉眼几乎无差异

📁 文件结构
text
.
├── compress_oa.py               # 压缩脚本：原始 SDXL -> OA-FP8 压缩包
├── decompress_oa.py             # 解压脚本：OA-FP8 压缩包 -> 标准 FP16 模型
├── OASingleFileLoader.py        # ComfyUI 自定义节点：动态加载 OA 压缩包
├── test_oa_loader.py            # 命令行测试脚本
├── README.md
└── results/                     # 测试图像与指标（可选）
实际文件名可能略有不同，请以发布版本为准。

🚀 快速开始
环境要求
ComfyUI（推荐版本 0.30.2 或更新）

PyTorch 2.7+（CUDA 12.8+）

8GB 或更大显存的 NVIDIA GPU

Python 3.13+

1. 生成 OA 压缩包
bash
python compress_oa.py
脚本会读取原始 SDXL 模型（hassakuXLIllustrious_v34.safetensors 或自定义路径），生成类似 oa_package_v4_1.safetensors 的压缩文件。

2. 安装 ComfyUI 节点
将 OASingleFileLoader.py 放入 ComfyUI 的 custom_nodes 目录，重启 ComfyUI。

3. 在 ComfyUI 中使用
在节点列表中找到 “OA-FP8 2.5GB Limit Loader”（或类似名称），添加后：

ckpt_name：选择 OA 压缩包（如 oa_package_v4_1.safetensors）

skeleton_name：选择标准 SDXL 骨架模型（如 hassakuXLIllustrious_v34.safetensors）

输出 MODEL、CLIP、VAE，连接到 KSampler 和 VAE Decode 即可。

📊 测试结果
压缩性能
指标	原始 FP16	直接 FP8	OA-FP8
文件大小	6.46 GB	3.41 GB	3.81 GB
PSNR (dB)	-	16.90	21.09
SSIM	-	0.6023	0.7383
MSE	-	0.022586	0.007971
以上为 5 种子平均，相对原始 FP16 模型生成图计算。

推理性能（8GB RTX 3070 Ti）
模型加载：约 5~40 秒（取决于预加载策略）

20 步采样：约 36~38 秒

稳定步耗时：约 1.59~1.85 秒/步

显存占用：allocated ~84 MB，reserved ~1.2 GB（清理后约 158 MB）

⚠️ 已知问题
动态加载器基于 Python Hook，每步仍有一定 CPU 侧开销；若要进一步提速，需要 CUDA 融合解压内核或连续块存储格式。

VAE 解码在 8GB 显存下可能触发 tiled 模式，增加约 20 秒耗时。

双缓冲异步预取版本目前未稳定，推荐使用单缓冲稳定版。

仅对 SDXL 做过深度验证，其他架构（如 DiT）效果可能不同。

🔭 未来方向
□ 连续块存储格式 v2，减少小张量传输
□ CUDA 融合解压内核，彻底消除 Python 临时张量分配
□ 真正的异步预取流水线
□ 扩展到更大模型（Flux、Z-Image 等）
📄 致谢
本项目为个人技术探索，不依赖任何外部库或框架。感谢 ComfyUI 社区提供的开放环境。

📬 反馈
如有问题或建议，欢迎提交 Issue。
使用前请备份模型，压缩与动态加载存在一定风险。
