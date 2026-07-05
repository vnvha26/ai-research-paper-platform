from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
BENCHMARK_FILE = EVALUATION_DIR / "f1_benchmark_queries.json"
RESULTS_DIR = EVALUATION_DIR / "results"
F1_RUNTIME_CONFIG_FILE = PROJECT_ROOT / "services" / "f1_paper_search" / "f1_search_config.py"

CONFIGS = [
    {
        "name": "baseline_60cos_40bm25",
        "alpha_cosine": 0.60,
        "alpha_bm25": 0.40,
        "section_weights": {"core": 0.40, "intro": 0.20, "method": 0.20, "conclusion": 0.20},
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    },
    {
        "name": "semantic_70cos_30bm25",
        "alpha_cosine": 0.70,
        "alpha_bm25": 0.30,
        "section_weights": {"core": 0.40, "intro": 0.20, "method": 0.20, "conclusion": 0.20},
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    },
    {
        "name": "balanced_50cos_50bm25",
        "alpha_cosine": 0.50,
        "alpha_bm25": 0.50,
        "section_weights": {"core": 0.40, "intro": 0.20, "method": 0.20, "conclusion": 0.20},
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    },
    {
        "name": "bm25_40cos_60bm25",
        "alpha_cosine": 0.40,
        "alpha_bm25": 0.60,
        "section_weights": {"core": 0.40, "intro": 0.20, "method": 0.20, "conclusion": 0.20},
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    },
    {
        "name": "core_heavy",
        "alpha_cosine": 0.60,
        "alpha_bm25": 0.40,
        "section_weights": {"core": 0.55, "intro": 0.15, "method": 0.20, "conclusion": 0.10},
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    },
    {
        "name": "method_heavy",
        "alpha_cosine": 0.60,
        "alpha_bm25": 0.40,
        "section_weights": {"core": 0.35, "intro": 0.15, "method": 0.35, "conclusion": 0.15},
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    },
    {
        "name": "intro_conclusion_heavy",
        "alpha_cosine": 0.60,
        "alpha_bm25": 0.40,
        "section_weights": {"core": 0.35, "intro": 0.30, "method": 0.10, "conclusion": 0.25},
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    },
    {
        "name": "bm25_long_doc_stronger",
        "alpha_cosine": 0.60,
        "alpha_bm25": 0.40,
        "section_weights": {"core": 0.40, "intro": 0.20, "method": 0.20, "conclusion": 0.20},
        "bm25_k1": 2.0,
        "bm25_b": 0.85,
    },
    {
        "name": "bm25_short_doc_stronger",
        "alpha_cosine": 0.60,
        "alpha_bm25": 0.40,
        "section_weights": {"core": 0.40, "intro": 0.20, "method": 0.20, "conclusion": 0.20},
        "bm25_k1": 1.2,
        "bm25_b": 0.55,
    },
]
