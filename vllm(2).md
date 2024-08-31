# LLM推理优化——PagedAttention初识篇（vLLM初识（二））

### 前言

在[LLM推理优化——KV Cache篇（百倍提速）](LLMInference.md)中，我们已经介绍了**KV Cache**技术的原理，从中我们可以知道，**KV Cache本质是空间换时间的技术**，对于大型模型和长序列，它可能会占用大量内存。实际上LLM从诞生之初就在与内存作斗争，只是计算时间问题更加尖锐，掩盖了这一部分。随着研究的推进，内存问题也变得越来越突出。

**vLLM**的作者们在论文**Efficient Memory Management for Large Language Model Serving with PagedAttention**提出了**PagedAttention**方法并在**vLLM**中实现。该算法受操作系统中的虚拟内存和分页技术启发，用于解决大型语言模型（LLM）服务中KV缓存内存管理效率低下的问题。

传统的内存管理方法在处理这种高动态性和大规模的KV缓存时，存在显著的缺陷。具体来说，这些方法要么导致内存的浪费（未被有效利用的内存区域），要么限制了批处理的能力，降低了系统的吞吐量。

### PagedAttention的核心思想

PagedAttention通过引入分页机制，将KV缓存的数据分块管理，以减少内存碎片并提高内存利用率。其核心思想包括以下几个方面：

1. 分块存储：

    将KV缓存的数据分割成固定大小的块（pages），每个块可以存储在不同的内存位置。这类似于操作系统中的分页机制，不要求数据在内存中是连续存储的。
2. 动态分页管理：

    当模型生成新序列时，PagedAttention可以动态分配或回收内存块，以确保只使用必要的内存空间。这避免了传统方法中预先分配大块连续内存所带来的浪费。
3. 跨请求共享：

    PagedAttention允许不同的请求共享同一个KV缓存的部分数据。这样，当多个请求使用相似的上下文时，可以复用之前存储的数据，进一步提高内存利用效率。

### PagedAttention的优势

1. 内存利用率提升：

    通过分页机制，PagedAttention避免了内存的碎片化问题，提高了内存的利用率。
2. 支持大批量处理：

    由于有效管理了KV缓存的内存占用，PagedAttention支持更大规模的批处理，进而提高了系统的吞吐量。
3. 灵活性与扩展性：

    PagedAttention可以灵活适应不同大小的模型和序列长度，并且在面对复杂解码任务时依然能够保持高效的性能表现。