"""Optional FP8 hooks; requires a torchao build that supports this Python/torch."""
from __future__ import annotations


def apply_fp8_inference(model):
    try:
        from torchao.quantization import quantize_, Float8WeightOnlyConfig
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "FP8 inference needs a compatible torchao build "
            "(Float8WeightOnlyConfig). Current environment import failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    quantize_(model, Float8WeightOnlyConfig())
    return model


def apply_fp8_training(model):
    try:
        from torchao.float8 import convert_to_float8_training
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "FP8 training needs a compatible torchao build "
            "(torchao.float8.convert_to_float8_training). Current environment "
            f"import failed: {type(exc).__name__}: {exc}"
        ) from exc
    return convert_to_float8_training(model)
