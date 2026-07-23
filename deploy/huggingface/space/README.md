---
title: JAX Addition Transformer
emoji: ➕
colorFrom: orange
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
short_description: Real JAX transformer inference for fixed three-digit addition
models:
  - marcoharuni95/jax-addition-transformer-10m
---

# JAX Addition Transformer

This Space runs the released 10,000,000-parameter JAX transformer for addition
requests with two integer operands from `0` through `999`. It is an arithmetic
model demonstration, not a general chatbot or an arbitrary-precision
calculator.

The Space downloads the stable Safetensors weights from
[marcoharuni95/jax-addition-transformer-10m](https://huggingface.co/marcoharuni95/jax-addition-transformer-10m),
loads them once, and returns only answers produced by greedy transformer
generation.

Source: [marcoharuni/jax-addition-transformer](https://github.com/marcoharuni/jax-addition-transformer)
