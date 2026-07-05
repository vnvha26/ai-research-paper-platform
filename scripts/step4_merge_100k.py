import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_100K = BASE_DIR / "data/01_raw/papers_dense_fixed_v2.jsonl"
MATCHED_97K = BASE_DIR / "data/02_processed/papers_final_new.jsonl"
OUTPUT_100K = BASE_DIR / "data/02_processed/papers_100k_complete.jsonl"

def main():
    OUTPUT_100K.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_100K.exists() or not MATCHED_97K.exists():
        print("Lỗi: Không tìm thấy file đầu vào. Kiểm tra lại đường dẫn")
        return

    matched_papers = {}
    print(f"[*] Đang đọc dữ liệu từ file 97k: {MATCHED_97K}")
    with open(MATCHED_97K, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                p = json.loads(line)
                pid = str(p.get("corpusId") or p.get("paperId"))
                matched_papers[pid] = p
            except json.JSONDecodeError:
                continue

    print(f"   -> Đã tải xong {len(matched_papers):,} bài báo có nội dung (Sections).")

    count_matched = 0
    count_missing = 0
    
    print(f"\n[*] Đang gộp dữ liệu và tạo file mới: {OUTPUT_100K}")
    with open(INPUT_100K, 'r', encoding='utf-8') as fin, open(OUTPUT_100K, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip(): continue
            
            try:
                p_orig = json.loads(line)
                pid = str(p_orig.get("corpusId") or p_orig.get("paperId"))
                
                if pid in matched_papers:
                    fout.write(json.dumps(matched_papers[pid], ensure_ascii=False) + "\n")
                    count_matched += 1
                
                else:
                    p_orig["sections"] = {
                        "intro": "",
                        "method": "",
                        "conclusion": ""
                    }
                    fout.write(json.dumps(p_orig, ensure_ascii=False) + "\n")
                    count_missing += 1
                    
            except json.JSONDecodeError:
                continue

    print("\n" + "═"*50)
    print("Hoàn thành")
    print("═"*50)
    print(f"Tổng số bài báo trong file mới : {count_matched + count_missing:,}")
    print(f"Số bài có Sections từ S2ORC: {count_matched:,} bài")
    print(f"Số bài bị thiếu (bù rỗng)  : {count_missing:,} bài")
    print(f"\nFile dữ liệu hoàn chỉnh đã lưu tại: {OUTPUT_100K}")

if __name__ == "__main__":
    main()
