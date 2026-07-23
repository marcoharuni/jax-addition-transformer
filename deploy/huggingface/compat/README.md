# Release checkpoint compatibility environment

This isolated `uv` project exists only to read the trusted v0.1.0 Python
pickle and convert its inference parameters to deterministic Safetensors.
It intentionally does not alter the root project's dependency lock.

The version evidence comes from `notebooks/01_build_train_exact_10m.ipynb`:
the release notebook installs Flax 0.12.2 and serializes an NNX `State`
containing host NumPy arrays. Flax 0.12.2 itself requires JAX 0.8.1 or
newer; JAX 0.8.1 is therefore the lowest compatible version rather than
the older 0.7.2 from the root project. NumPy 2.3.3 matches the tagged lock.

Normal deployment never reads the pickle and does not depend on Flax.
