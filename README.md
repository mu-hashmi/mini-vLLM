# Mini-vLLM

> **⚠️ Work in Progress**  
> This project is actively under development. Features are being implemented incrementally. Check back regularly for updates!

## Overview

**Mini-vLLM** is a minimal, production-grade LLM inference server—a simplified re-implementation of core ideas from [vLLM](https://github.com/vllm-project/vllm), but scoped to be buildable by one person.

This project demonstrates how modern LLM inference servers work under the hood. By building a simplified version, it explores the complex systems that power today's AI applications. (I'm also doing it as a personal goal/learning exercise to dive deeper into LLM serving/infra).

## Key Concepts

This project covers:

- **Continuous batching and request scheduling** — How to efficiently batch multiple requests together and schedule them for optimal GPU utilization
- **KV-cache block management** — Techniques for managing key-value caches efficiently to reduce memory usage
- **Token streaming** — Implementing real-time token generation and streaming responses to clients
- **Async GPU execution** — Leveraging CUDA graphs and streams for optimal GPU performance
- **Serving multiple models concurrently** — Architecting a system that can handle multiple models simultaneously
- **Memory fragmentation issues** — How memory fragmentation occurs and how block allocators solve these problems
- **Throughput vs. latency trade-offs** — How to measure and optimize for different performance metrics

## Project Goals

- Build a working inference server from scratch
- Explore the internals of production LLM serving systems
- Create something that's both educational and potentially useful
- Demonstrate modern GPU programming and async systems design

## Status

🚧 **Early Development** — Core architecture and foundational components are being implemented.

## Acknowledgments

Inspired by the excellent work done by the [vLLM](https://github.com/vllm-project/vllm) team. This project explores their architecture concepts, not to compete with it.

