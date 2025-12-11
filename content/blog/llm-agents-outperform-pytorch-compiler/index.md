---
title: "How LLM-Based Agents Outperform the PyTorch Compiler by 2.0×"
# title: "LLM-Based Agents Can Outperform the PyTorch Compiler by 2×"
description: "TODO"
date: 2025-12-10
draft: false
showAuthor: false
---

## GPU Optimization Problem

New generations of AI datacenter GPUs are now being rolled out on an annual basis, forcing software support to play a constant game of catch-up. To make the problem worse, new AI/ML model techniques are being proposed constantly. This leads to a set of workloads that ML compiler engineers are unlikely to optimize for, unless a particular technique gains a lot of traction.

GPU performance optimization using CUDA or Triton is a notoriously challenging process. This is why software lags behind the latest generation of NVIDIA GPUs, and compiler engineers are forced to optimize only for the most critical workloads.

Without excellent compiler support, demonstrating good performance for a new idea could mean tons of manual GPU programming. Thus, it's becoming more difficult for AI researchers to challenge conventional wisdom. Can we find a way to eliminate manual engineering from the equation, and what would such a system look like?

## Logical Framework

<!-- As new generations of GPUs roll out, and new AI model techniques are proposed, we are met with a constant dilemma: how do we keep up with  -->

<!-- Thus, GPU performance optimization has become a crucial aspect of modern AI inference.  -->

![PIKE Cost Graph](logical-framework-simplified.png)

![PIKE Cost Graph](pike-b-diagram.png)

![PIKE Cost Graph](pike-cost-level-3-pike.png)

![PIKE Cost Graph](pike-speedup-level-3-pike.png)

<!-- ![](pike-speedup-level-5.png) -->

<div class="flex justify-center">
  <img src="pike-speedup-level-5.png" alt="My image" width="520" />
</div>

<!-- ![](pike-hist-1.png)

![](pike-hist-2.png) -->

<div class="flex justify-center">
  <img src="pike-hist-1.png" alt="My image" width="500" />
</div>

<div class="flex justify-center">
  <img src="pike-hist-2.png" alt="My image" width="500" />
</div>
