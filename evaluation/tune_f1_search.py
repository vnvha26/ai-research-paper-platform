import argparse
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.f1_tuning import run_tuning
from evaluation.f1_tuning.f1_tuning_config import (
    BENCHMARK_FILE,
    F1_RUNTIME_CONFIG_FILE,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tinh chỉnh F1 và tự động áp dụng cấu hình tốt nhất.",
        add_help=False,
    )
    parser._positionals.title = "tham số"
    parser._optionals.title = "tùy chọn"
    parser.add_argument("-h", "--help", action="help", help="Hiển thị trợ giúp.")
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_FILE, help="Đường dẫn benchmark.")
    parser.add_argument("--k", type=int, default=10, help="Số kết quả đánh giá.")
    parser.add_argument("--top-k-lexical", type=int, default=500, help="Số ứng viên từ khóa.")
    parser.add_argument("--top-k-semantic", type=int, default=300, help="Số ứng viên ngữ nghĩa.")
    parser.add_argument("--review-top-n", type=int, default=50, help="Số kết quả cần duyệt.")
    parser.add_argument("--skip-review", action="store_true", help="Bỏ qua bảng duyệt nhãn.")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=F1_RUNTIME_CONFIG_FILE,
        help="File cấu hình F1.",
    )
    args = parser.parse_args()
    return run_tuning(
        benchmark_path=args.benchmark,
        config_file=args.config_file,
        k=args.k,
        top_k_lexical=args.top_k_lexical,
        top_k_semantic=args.top_k_semantic,
        review_top_n=args.review_top_n,
        skip_review=args.skip_review,
    )


if __name__ == "__main__":
    raise SystemExit(main())
