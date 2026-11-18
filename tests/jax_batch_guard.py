"""Skip helper for tests that require aweSOM.batch's JAX capabilities."""
import pytest


def require_batch_jax(*, allow_module_level: bool = False):
    jax = pytest.importorskip("jax")
    if not hasattr(jax, "enable_x64") or not hasattr(jax, "shard_map"):
        pytest.skip(
            f"aweSOM.batch requires jax with enable_x64 and shard_map (found {jax.__version__})",
            allow_module_level=allow_module_level,
        )
    return jax
