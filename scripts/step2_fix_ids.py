import json
import requests
import os
import time
from tqdm import tqdm
from pathlib import Path

API_KEY = os.environ.get("S2_API_KEY", "YOUR_API_KEY")
HEADERS = {"x-api-key": API_KEY} if API_KEY != "YOUR_API_KEY" else {}

RATE_LIMIT_DELAY = 1.1 if API_KEY == "YOUR_API_KEY" else 0.11 

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    input_file  = BASE_DIR / "data/01_raw/papers_dense_v2.jsonl"
    output_file = BASE_DIR / "data/01_raw/papers_dense_fixed_v2.jsonl"
    
    if not input_file.exists():
        print(f"Không tìm thấy file {input_file}!")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("1. Đang đọc dữ liệu...")
    papers = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
            
    print(f"Đã tải {len(papers)} bài báo. Đang xin cấp CorpusID...")
    
    BATCH_SIZE = 500
    
    for i in tqdm(range(0, len(papers), BATCH_SIZE), desc="Dịch ID"):
        batch = papers[i:i+BATCH_SIZE]
        ids = [p["paperId"] for p in batch] 
        
        for attempt in range(3):
            try:
                res = requests.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": "paperId,corpusId"},
                    json={"ids": ids},
                    headers=HEADERS,
                    timeout=30
                )
                
                if res.status_code == 200:
                    data = res.json()
                    for paper, api_data in zip(batch, data):
                        if api_data and api_data.get("corpusId"):
                            paper["corpusId"] = str(api_data["corpusId"])
                            
                    time.sleep(RATE_LIMIT_DELAY)
                    break 
                    
                elif res.status_code == 429:
                    time.sleep(5)
                else:
                    time.sleep(2)
                    
            except Exception:
                time.sleep(2)
            
    print("\n2. Đang lưu ra file mới...")
    with open(output_file, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            
    print(f"Hoàn thành! Đã lưu tại: {output_file}")

if __name__ == "__main__":
    main()
