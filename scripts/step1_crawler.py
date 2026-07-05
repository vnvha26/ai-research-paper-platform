import requests
import json
import time
import os
import heapq
from pathlib import Path
from tqdm import tqdm

API_KEY      = os.environ.get("S2_API_KEY", "YOUR_API_KEY")
TARGET_COUNT = 100_000 
YEAR_MIN     = 2015
YEAR_MAX     = 2024
FIELDS       = {"Computer Science", "Artificial Intelligence", "Machine Learning"}

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data/01_raw/papers_dense_v2.jsonl"
CHECKPOINT  = BASE_DIR / "data/checkpoints/step1_checkpoint.json"

BATCH_SIZE   = 200 
MAX_REFS     = 50 

class S2Client:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"x-api-key": api_key} if api_key != "YOUR_API_KEY" else {}
        
        self.rate_limit = 0.8 if api_key == "YOUR_API_KEY" else 10.0
        self.last_call = 0

    def _wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < 1.0 / self.rate_limit:
            time.sleep((1.0 / self.rate_limit) - elapsed)
        self.last_call = time.time()

    def get_seeds(self):
        print("[PHASE 0] Đang lấy Seed bằng từ khóa phổ biến...")
        seeds = []
        queries = ["Large Language Models", "Deep Learning", "Neural Networks", "Computer Vision", "Transformer"]
        
        for q in queries:
            for attempt in range(3):
                self._wait()
                url = f"{self.base_url}/paper/search"
                params = {"query": q, "year": f"{YEAR_MIN}-{YEAR_MAX}", "limit": 100, "fields": "paperId,citationCount"}
                try:
                    r = requests.get(url, params=params, headers=self.headers, timeout=20)
                    if r.status_code == 200:
                        data = r.json().get("data", [])
                        for p in data:
                            if p.get("paperId"):
                                seeds.append(p["paperId"])
                        break
                    elif r.status_code == 429:
                        print(f"Quá tải khi tìm '{q}'. Đang nghỉ 10s...")
                        time.sleep(10)
                    else:
                        break
                except Exception:
                    time.sleep(5)
        
        seeds.extend([
            "649fd3c26d73f9fadd13aa30f7ae4942971cf3b9",
            "8b8b939f1c7d2bdff6ebcf3ef5daebbbcd1ebfba",
            "13437e3d1ce2b92c4d081f9a6e133e9b1bbdd6c4"
        ])
        return list(set(seeds))

    def fetch_batch(self, ids):
        clean_ids = [str(i) for i in ids if i and str(i).strip() and str(i) != "None"]
        if not clean_ids: return []

        url = f"{self.base_url}/paper/batch"
        params = {"fields": "paperId,title,year,abstract,authors,s2FieldsOfStudy,citationCount,references"}
        
        for attempt in range(3):
            self._wait()
            try:
                r = requests.post(url, json={"ids": clean_ids}, params=params, headers=self.headers, timeout=45)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    print("\n API quá tải (429). Đang nghỉ 10 giây...")
                    time.sleep(10)
                elif r.status_code == 504:
                    print("\n Máy chủ S2 bị nghẽn (504). Đang nghỉ 5 giây...")
                    time.sleep(15)
                else:
                    print(f"\nAPI Error {r.status_code}: {r.text} | Thử lại...")
                    time.sleep(5)
            except Exception as e:
                time.sleep(5)
        return []

def is_valid(paper):
    if not paper or not paper.get("paperId"): return False
    
    refs = paper.get("references")
    if refs is None or len(refs) < 1: return False 
    
    year = paper.get("year")
    if not year or not (YEAR_MIN <= year <= YEAR_MAX): return False
    
    if not paper.get("abstract") or len(paper["abstract"]) < 20: return False
    
    s2_fields = paper.get("s2FieldsOfStudy")
    if not s2_fields or not isinstance(s2_fields, list): 
        return False
        
    paper_categories = {f.get("category") for f in s2_fields if isinstance(f, dict)}
    if not paper_categories.intersection(FIELDS):
        return False
    
    return True

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    client = S2Client(API_KEY)
    
    collected_ids = set()
    visited_ids = set()
    pq = [] 
    citation_counts = {}

    if CHECKPOINT.exists():
        print("[*] Phát hiện file Checkpoint. Đang khôi phục tiến độ cũ...")
        try:
            with open(CHECKPOINT, "r", encoding="utf-8") as f:
                state = json.load(f)
                collected_ids = set(state["collected_ids"])
                visited_ids = set(state.get("visited_ids", state["collected_ids"]))
                citation_counts = state["citation_counts"]
                pq = [tuple(item) for item in state["pq"]]
                heapq.heapify(pq)
            print(f"[*] Đã khôi phục thành công: {len(collected_ids)} bài báo AI/CS (Tổng quét: {len(visited_ids)}).")
            file_mode = "a"
        except Exception as e:
            print(f"Lỗi đọc Checkpoint: {e}. Sẽ chạy lại từ đầu.")
            file_mode = "w"
    else:
        print("[*] Chạy mới từ đầu...")
        seeds = client.get_seeds()
        if not seeds:
            print("Lỗi: Không lấy được Seed")
            return

        for s_id in seeds:
            heapq.heappush(pq, (-1, s_id))
            citation_counts[s_id] = 1
        file_mode = "w"

    if len(collected_ids) >= TARGET_COUNT:
        print("Đã đủ số lượng")
        return

    pbar = tqdm(total=TARGET_COUNT, initial=len(collected_ids), desc="Đang cào dữ liệu CS/AI")
    
    try:
        with open(OUTPUT_FILE, file_mode, encoding="utf-8") as f:
            while len(collected_ids) < TARGET_COUNT and pq:
                current_batch = []
                
                while len(current_batch) < BATCH_SIZE and pq:
                    _, pid = heapq.heappop(pq)
                    if pid not in visited_ids:
                        current_batch.append(pid)
                        visited_ids.add(pid)
                
                if not current_batch: break
                
                results = client.fetch_batch(current_batch)
                if not results: continue
                
                for p in results:
                    if not p or not p.get("paperId"): continue
                    
                    if is_valid(p):
                        p_id = p["paperId"]
                        
                        refs = [r["paperId"] for r in p["references"] if r.get("paperId")]
                        p["outCitations"] = refs
                        del p["references"]
                        
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
                        f.flush()
                        
                        collected_ids.add(p_id)
                        pbar.update(1)
                        
                        if len(collected_ids) >= TARGET_COUNT: break
                        
                        for ref_id in refs[:MAX_REFS]:
                            if ref_id not in visited_ids:
                                citation_counts[ref_id] = citation_counts.get(ref_id, 0) + 1
                                heapq.heappush(pq, (-citation_counts[ref_id], ref_id))
                                
                with open(CHECKPOINT, "w", encoding="utf-8") as cp_file:
                    json.dump({
                        "collected_ids": list(collected_ids),
                        "visited_ids": list(visited_ids),
                        "citation_counts": citation_counts,
                        "pq": pq
                    }, cp_file)

    except KeyboardInterrupt:
        print("Tiến độ vẫn được lưu")
        
    pbar.close()
    print(f"\n hoàn thành")
    print(f"Tổng số bài báo (CS/AI) thực tế đã lưu: {len(collected_ids)}")
    print(f"Tổng số bài báo đã quét qua (Bao gồm cả rác): {len(visited_ids)}")

if __name__ == "__main__":
    main()
