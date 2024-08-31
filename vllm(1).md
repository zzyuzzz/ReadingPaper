# vLLM初识（一）

### 前言

在[LLM推理优化——KV Cache篇（百倍提速）](LLMInference.md)中，我们已经介绍了**KV Cache**技术的原理，从中我们可以知道，**KV Cache本质是空间换时间的技术**，对于大型模型和长序列，它可能会占用大量内存。实际上LLM从诞生之初就在与内存作斗争，只是计算时间问题更加尖锐，掩盖了这一部分。随着研究的推进，内存问题也变得越来越突出。

**vLLM**提出了**PagedAttention**方法，尝试通过将 KV 缓存划分为可通过查找表访问的块来优化内存使用。因此，KV 缓存不需要存储在连续内存中，并且根据需要分配块。内存效率可以提高内存受限工作负载上的 GPU 利用率，因此可以支持更多推理批处理。我接下来就使用几篇博客来初步了解一下**vLLM**。

### vLLM初探

vLLM 是一个快速且易于使用的库，用于 LLM 推理和服务。

vLLM速度很快，具有以下特点：

+ 最先进的服务吞吐量
+ 使用 PagedAttention 高效管理注意力键和值内存
+ 连续批处理传入请求
+ 使用 CUDA/HIP 图快速执行模型
+ 量化：GPTQ、AWQ、SqueezeLLM、FP8 KV 缓存
+ 优化的 CUDA 内核

vLLM 灵活且易于使用：

+ 与流行的 HuggingFace 型号无缝集成
+ 使用各种解码算法提供高吞吐量服务，包括并行采样、波束搜索等
+ 面向分布式推理的张量并行性和流水线并行性支持
+ 面向分布式推理的张量并行性和流水线并行性支持
+ 流式输出
+ 兼容 OpenAI 的 API 服务器
+ 支持 NVIDIA GPU 和 AMD GPU

#### 安装

为了提高性能，vLLM编译了许多cuda内核。该编译引入了与其他 CUDA 版本和 PyTorch 版本的二进制**不兼容**。安装时务必注意**cuda版本**和**pytorch版本**。

```bash
# Install vLLM with CUDA 12.1.
pip install vllm
```

```bash
# Install vLLM with CUDA 11.8.
export VLLM_VERSION=0.4.0
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

##### 从源代码构建

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .  # This may take 5-10 minutes.
```

##### 使用docker镜像

```bash
# Use `--ipc=host` to make sure the shared memory is large enough.
docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
```
