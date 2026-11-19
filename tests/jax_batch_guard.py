"""Skip helper for tests that require aweSOM.batch's JAX capabilities."""
import pytest


def require_batch_jax(*, allow_module_level: bool = False):
    try:
        import jax
    except ModuleNotFoundError as exc:
        if exc.name == "jax":
            pytest.skip("jax is not installed", allow_module_level=allow_module_level)
        raise
    if not hasattr(jax, "enable_x64") or not hasattr(jax, "shard_map"):
        version = getattr(jax, "__version__", "unknown")
        pytest.skip(
            f"aweSOM.batch requires jax with enable_x64 and shard_map (found {version})",
            allow_module_level=allow_module_level,
        )
    return jax
