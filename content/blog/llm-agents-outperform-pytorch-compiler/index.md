---
title: "How LLM-Based Agents Outperform the PyTorch Compiler by 2×"
# title: "LLM-Based Agents Can Outperform the PyTorch Compiler by 2×"
description: "Our LLM-based, multi-agent PyTorch optimization system achieves up to 2.88× speedup over PyTorch eager. We present a logical framework for comparing multi-agent evolutionary optimization systems, and explore the configuration space for PyTorch and GPU performance optimization, with the help of OpenEvolve."
date: 2025-12-10
draft: false
showAuthor: false
---

This post gives an overview of our recent paper preprint, *[Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2511.16964)*, which I authored along with [Luka Grbcic](https://lukagrbcic.github.io/), [Samuel Williams](https://profiles.lbl.gov/20370-samuel-williams), and [Costin Iancu](https://www.linkedin.com/in/costin-iancu-5a8b011/).

We introduce a logical framework for comparing multi-agent PyTorch optimization systems, along with our implementations within it, collectively known as *PyTorch Inference Kernel Evolution* (PIKE). We explore the configuration space with the help of [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve), and we manage to outperform PyTorch's eager execution mode by up to 2.88×!

## GPU Optimization Problem

New generations of AI datacenter GPUs are now being rolled out on an annual basis, forcing software support to play a constant game of catch-up. To make the problem worse, new AI/ML model techniques are being proposed constantly. This leads to a set of workloads that library/compiler engineers are unlikely to optimize for, unless an idea gains significant traction from the community.

<!-- GPU performance optimization using CUDA or Triton is a notoriously challenging process. This is why software lags behind the latest generation of NVIDIA GPUs, and ML library/compiler engineers are forced to optimize only for the most critical workloads. -->

Without excellent library/compiler support, demonstrating good performance for a new idea could mean tons of manual GPU programming. Thus, it's becoming more difficult for AI/ML researchers to challenge conventional wisdom.

To name one example, in December 2022, [H3](https://arxiv.org/abs/2212.14052) showed the viability of replacing the standard Transformer architecture in language modeling with a hybrid architecture that integrates state space model (SSM) layers. However, achieving competitive performance in their paper required expert-level GPU kernel development.
Adoption of the idea into modern LLM inference engines took **3 years**, mainly due to GPU memory management challenges [[vLLM announcement](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/), [SGLang announcement](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)].

<!-- The idea is **only now** being adopted into modern LLM inference engines, and doing so took **3 years** mainly due to GPU memory management adjustments [[vLLM announcement](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/), [SGLang announcement](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)]. -->

<!-- Furthermore, adoption on mainstream LLM inference engines took 3 years, -->

Can we find a way to eliminate manual GPU performance engineering from the equation using LLMs, and what would such a system look like?

## Our Solution

Our target GPU was an NVIDIA H100. We used a modified version of [KernelBench](https://arxiv.org/abs/2502.10517), a benchmark suite covering a range of machine learning architectures in PyTorch. We target these key levels from the suite:

- **Level 3 (Curated blocks from older models):** RNNs, Attention, Convolutions
- **Level 5 (Frontier workloads from SOTA 2024 models):** DeepSeek-V3, Llama 3, Mamba-2, Hunyuan Video

Numerous prior works have shown effective LLM-based optimization systems that target KernelBench tasks, but the dynamics of multi-agent systems for this performance engineering problem remain unexplored. Thus, we have developed a logical framework to analyze these systems and fill the gap!

<!-- As new generations of GPUs roll out, and new AI model techniques are proposed, we are met with a constant dilemma: how do we keep up with  -->

<!-- Thus, GPU performance optimization has become a crucial aspect of modern AI inference.  -->

![PIKE Logical Framework Simplified](logical-framework-simplified.png)

A simplified visual of the problem and our setup is shown above. We have built a robust, performant evaluator which receives PyTorch/CUDA/Triton code, checks for correctness, runs performance tests, then returns back errors and metrics. The idea is that we can plug in an LLM-based system which iteratively improves the performance of the original PyTorch code by querying the evaluator in a loop, then eventually returns the most performant, valid solution.

Using this setup, we develop a logical framework in which evolutionary, LLM-based multi-agent systems can operate. In particular, we identify key hyperparameters of these evolutionary strategies which impact the optimization process, such as the explore/exploit ratio, islands, mutation/crossover, and LLM-based error fixing.

### PIKE-B (Branching Search)

We initially developed PIKE-B, a hand-written, exploit-heavy, evolutionary strategy which operates in optimization rounds.

![PIKE-B Diagram](pike-b-diagram.png)

PIKE-B spends a limited number of LLM queries on fixing errors using an error fixing agent (EFA). After a particular cutoff point, the best solutions from this round are ranked by runtime, then the top-k solutions are used as seeds for the next set of LLM queries. 

We describe the strategy as exploit-heavy because it concentrates effort on the best existing solutions. As it turns out, we find this approach to work surprisingly well in our results. Prior works have proposed similar approaches, but we are especially interested in **why** it works so well.

### PIKE-O (OpenEvolve)

To explore the 

## Performance Results

![PIKE Level 3-pike Cost Graph](pike-cost-level-3-pike.png)

![PIKE Level 3-pike Speedup](pike-speedup-level-3-pike.png)

<!-- ![](pike-speedup-level-5.png) -->

<div class="flex justify-center">
  <img src="pike-speedup-level-5.png" alt="PIKE Level 5 Speedup" class="rounded-md" width="520" />
</div>

<!-- ![](pike-hist-1.png)

![](pike-hist-2.png) -->

## Understanding the LLM-Generated Code

<div class="flex justify-center">
  <img src="pike-hist-1.png" alt="PIKE code histograms 1" class="rounded-md" width="500" />
</div>

<div class="flex justify-center">
  <img src="pike-hist-2.png" alt="PIKE code histograms 2" class="rounded-md" width="500" />
</div>
