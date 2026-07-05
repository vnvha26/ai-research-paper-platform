import re
from pathlib import Path

from evaluation.f1_tuning.f1_benchmark_loader import log


def format_section_weights(weights: dict[str, float]) -> str:
    return (
        "SECTION_WEIGHTS = {\n"
        f"    \"core\": {weights['core']:.2f},\n"
        f"    \"intro\": {weights['intro']:.2f},\n"
        f"    \"method\": {weights['method']:.2f},\n"
        f"    \"conclusion\": {weights['conclusion']:.2f},\n"
        "}"
    )


def _replace_once(text: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"ko the cap nhat config F1: {pattern}")
    return updated


def apply_best_config(
    config_file: Path,
    best_config: dict,
    top_k_lexical: int,
    top_k_semantic: int,
) -> None:
    text = config_file.read_text(encoding="utf-8")
    replacements = [
        (
            r"SECTION_WEIGHTS\s*=\s*\{.*?\}",
            format_section_weights(best_config["section_weights"]),
        ),
        (
            r"ALPHA_COSINE\s*=\s*[0-9.]+",
            f"ALPHA_COSINE = {best_config['alpha_cosine']:.2f}",
        ),
        (
            r"ALPHA_BM25\s*=\s*[0-9.]+",
            f"ALPHA_BM25 = {best_config['alpha_bm25']:.2f}",
        ),
        (
            r"ACTIVE_CONFIG_NAME\s*=\s*\".*?\"",
            f"ACTIVE_CONFIG_NAME = \"{best_config['name']}\"",
        ),
        (
            r"BM25_K1\s*=\s*[0-9.]+",
            f"BM25_K1 = {best_config['bm25_k1']}",
        ),
        (
            r"BM25_B\s*=\s*[0-9.]+",
            f"BM25_B = {best_config['bm25_b']}",
        ),
        (
            r"TOP_K_LEXICAL\s*=\s*\d+",
            f"TOP_K_LEXICAL = {top_k_lexical}",
        ),
        (
            r"TOP_K_SEMANTIC\s*=\s*\d+",
            f"TOP_K_SEMANTIC = {top_k_semantic}",
        ),
    ]
    for pattern, replacement in replacements:
        text = _replace_once(text, pattern, replacement)
    config_file.write_text(text, encoding="utf-8")
    log(f"Đã áp dụng best config: {config_file}")
