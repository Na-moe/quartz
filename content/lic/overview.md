---
title: Learned Image Compression Overview
---
## 引言

鉴于秋招期间重新回顾了以前的关于 基于深度学习的图像压缩 (Learned Image Compression, LIC) 的工作，发觉自己曾经对整个体系的理解还是略有不足，借此机会通过 blog 的方式梳理一下一系列经典的 LIC 工作，以飨后继研究者。

这系列 blogs 会按照时间顺序对一系列 LIC 的工作进行整理，包括理论介绍和部分代码展示，也可能包括一些自己的思考。

在此之前，我们先对基于深度学习的图像压缩做一个简单的介绍。

图像（以及其连续排列，视频）也许是互联网上最值得压缩的数据。未经压缩的彩色图像中的每个像素通常需要 24 个比特，而一张 1080p FHD 的图像则包含了 `1920 * 1080 = 2073600` 个像素，需要 `6220800B = 6075MB ~= 6GB` 的空间。而互联网上每天可能产生数以百万计的图像，不加压缩地存储或者传输这些图像的代价是不可接受的，因此我们我们需要对图像进行压缩。

在这方面，传统的图像压缩已经取得了极大的成就，`.jpg` 成为我们最熟知的后缀名之一，而最新的标准 `VVC Intra` 仍然在持续开发以改善图像压缩的压缩率和失真。

基于深度学习的图像压缩尝试将神经网络应用于图像压缩，以取得比手工设计的图像压缩算法更好的效果。下图展示了一系列 LIC 的压缩率-失真曲线。(source: [Mateen Ulhaq's blog](https://yodaembedding.github.io/post/learned-image-compression))

![[Pasted image 20250102074320.png]]


## 熵建模、变换编码与量化

一个我们最常听说过的数据压缩的例子就是摩尔斯电码，它为英文字母分配了不同长度的点划排列，从而通过电报传输数据。其设计的一个标志特点便是为最常见的字母 e 分配了最简短的电码：一个点。这个设计揭示了数据压缩的一个准则：给高概率的符号分配短码。为了确定码字的分配，我们需要确定编码对象的概率分布，这个过程称之为熵建模。

摩尔斯电码就是通过静态统计英文字母的频率来进行熵建模的。然而，概率分布不总是一成不变的，当我们知道一些额外的信息后，条件概率分布和没有任何信息下的边际概率分布会有差别，使得码字分配也随之变化。这种根据先前信息动态确定编码分布的过程称为上下文建模。

然而，编码对象的原始表示并不一定利于我们确定其概率分布，例如图像在RGB空间中的概率分布估计通常是困难的。庆幸的是，我们可以将其变到容易估计概率分布的空间上，这个过程称之为变换编码。举例而言，JPEG 所采用的是离散余弦变换，而 JPEG 2000 采用的是小波变换。

变换编码通常将图像变换成为一个具有连续值的高维向量。为了使实际编码有意义，连续值需要被映射到有限个离散值上，这个过程被称为量化。量化引入的误差使得图像解压后通常具有失真。图像压缩所优化的目标就是尽可能高的压缩率和尽可能低的失真，如下图所示。

![[Pasted image 20250102202054.png]]


| 年份   | 第一作者    | 单位        | 标题                                                                                                       | 主要贡献                             |
| ---- | ------- | --------- | -------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 2017 | Balle   | NYU       | End-to-end Optimized Image Compression                                                                   | E2E, Transform: GDN, Quant: AUN, |
| 2018 | Balle   | Google    | Variational Image Compression with a Scale Hyperprior                                                    |                                  |
| 2018 | Minnen  | Google    | Joint Autoregressive and Hierarchical Priors for Learned Image Compression                               |                                  |
| 2020 | Cheng   | Waseda    | Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules            |                                  |
| 2020 | Minnen  | Google    | Channel-wise Autoregressive Entropy Models for Learned Image Compression                                 |                                  |
| 2020 | Mentzer | Google    | High-Fidelity Generative Image Compression                                                               |                                  |
| 2021 | He      | SenseTime | Checkerboard Context Model for Efficient Learned Image Compression                                       |                                  |
| 2022 | He      | SenseTime | ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding |                                  |


