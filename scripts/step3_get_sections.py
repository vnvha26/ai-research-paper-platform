import requests
import gzip
import os
import threading
import concurrent.futures
import time
import re
from pathlib import Path

try:
    import orjson as json
except ImportError:
    import json

API_KEY      = os.environ.get("S2_API_KEY", "YOUR_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE   = BASE_DIR / "data/01_raw/papers_dense_fixed_v2.jsonl"
OUTPUT_FILE  = BASE_DIR / "data/02_processed/papers_final_new.jsonl"
CHECKPOINT   = BASE_DIR / "data/checkpoints/checkpoint_step3_new.json"

HEADERS  = {"x-api-key": API_KEY} if API_KEY != "YOUR_API_KEY" else {}
BASE_URL = "https://api.semanticscholar.org/datasets/v1"

MAX_WORKERS = 6              
SHARD_BATCH_SIZE = 12        

SECTION_PATTERNS = {
    "intro": r'\n\s*(?:[IVXLCDM]+|\d+(?:\.\d+)*)?\.?\s*(introduction|background|overview)',
    "method": r'\n\s*(?:[IVXLCDM]+|\d+(?:\.\d+)*)?\.?\s*(method|methodology|approach|experiment|material)',
    "conclusion": r'\n\s*(?:[IVXLCDM]+|\d+(?:\.\d+)*)?\.?\s*(conclusion|summary|discussion|future work)'
}

write_lock = threading.Lock()

def parse_flat_text(full_text):
    buckets = {"intro": "", "method": "", "conclusion": ""}
    
    if not full_text or not isinstance(full_text, str):
        return buckets
        
    text_lower = full_text.lower()
    
    pos = {}
    for key, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text_lower)
        if match:
            pos[key] = match.start()
            
    valid_pos = [(k, v) for k, v in pos.items()]
    valid_pos.sort(key=lambda x: x[1]) 
    
    if not valid_pos:
        buckets["intro"] = full_text[:2500]
        return buckets
        
    for i, (key, start_idx) in enumerate(valid_pos):
        end_idx = valid_pos[i+1][1] if i + 1 < len(valid_pos) else start_idx + 4000 
        text_chunk = full_text[start_idx:end_idx]
        if text_chunk:
            buckets[key] = text_chunk.strip()[:2500]
        
    return buckets

def extract_sections(data):
    buckets = {"intro": "", "method": "", "conclusion": ""}
    
    if not data:
        return buckets

    if isinstance(data, str):
        return parse_flat_text(data)

    if isinstance(data, dict) and data.get("text"):
        text_content = data.get("text")
        if isinstance(text_content, str):
            return parse_flat_text(text_content)

    if isinstance(data, list):
        intro_texts, method_texts, concl_texts = [], [], []
        for block in data:
            if isinstance(block, dict): 
                raw_sec = block.get("section")
                sec = str(raw_sec).lower() if raw_sec else ""
                
                raw_txt = block.get("text")
                txt = str(raw_txt) if raw_txt else ""
                
                if not txt: continue
                
                if any(kw in sec for kw in ["introduction", "background", "overview"]):
                    intro_texts.append(txt)
                elif any(kw in sec for kw in ["method", "methodology", "approach", "experiment", "material"]):
                    method_texts.append(txt)
                elif any(kw in sec for kw in ["conclusion", "summary", "discussion", "future work"]):
                    concl_texts.append(txt)
                    
        if intro_texts: buckets["intro"] = " ".join(intro_texts)[:2500]
        if method_texts: buckets["method"] = " ".join(method_texts)[:2500]
        if concl_texts: buckets["conclusion"] = " ".join(concl_texts)[:2500]
        
        if not buckets["intro"] and data and isinstance(data[0], dict):
            first_txt = data[0].get("text")
            if isinstance(first_txt, str):
                buckets["intro"] = first_txt[:2500]

    return buckets

def get_shard_urls():
    print("[*] Đang xin danh sách link từ API Semantic Scholar...")
    for _ in range(3):
        try:
            res = requests.get(f"{BASE_URL}/release/latest/dataset/s2orc", headers=HEADERS, timeout=30)
            res.raise_for_status()
            return res.json().get("files", [])
        except Exception as e:
            print(f"Lỗi lấy link (thử lại sau 5s): {e}")
            time.sleep(5)
    return []

def process_shard(shard_url, shard_idx, target_ids, papers_dict, matched_ids):
    session = requests.Session()
    session.headers.update(HEADERS)
    local_matched = 0
    lines_read = 0  
    
    try:
        with session.get(shard_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with gzip.GzipFile(fileobj=r.raw) as gz:
                for line in gz:
                    if not line.strip(): continue
                    
                    lines_read += 1
                    
                    if lines_read % 50_000 == 0:
                        with write_lock:
                            print(f"  ⚡ [Shard {shard_idx}] Đang quét... Đã đọc {lines_read:,} bài | Khớp: {local_matched}")

                    try:
                        rec = json.loads(line)
                        if not isinstance(rec, dict): continue
                        
                        c_id = str(rec.get("corpusid") or "")
                        if not c_id:
                            ext = rec.get("externalids") or rec.get("external_ids")
                            if isinstance(ext, dict): 
                                c_id = str(ext.get("CorpusId") or ext.get("corpusid") or "")
                    except:
                        continue 

                    if c_id and c_id in target_ids:
                        pid = c_id 
                        
                        with write_lock:
                            if pid not in matched_ids:
                                matched_ids.add(pid)
                                local_matched += 1
                                
                                base_paper = papers_dict[pid].copy()
                                raw_text_data = rec.get("content") or rec.get("body_text")
                                base_paper["sections"] = extract_sections(raw_text_data)
                                
                                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                                    dumped = json.dumps(base_paper)
                                    if isinstance(dumped, bytes):
                                        dumped = dumped.decode('utf-8')
                                    f.write(dumped + "\n")
                                
                                if local_matched % 2 == 0 or local_matched == 1:
                                    print(f"  🔥 [Shard {shard_idx}] TÌM THẤY {pid}! (Tổng hệ thống: {len(matched_ids)})")

    except Exception as e:
        print(f"\n[Lỗi Shard {shard_idx}]: Cảnh báo - {e}")
        return shard_idx, 0, False 
    
    return shard_idx, local_matched, True 

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Đang load danh sách mục tiêu từ {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print(f"Không tìm thấy {INPUT_FILE}.")
        return

    papers = {}
    target_ids = set()
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("corpusId"):
                pid = str(rec["corpusId"]) 
                papers[pid] = rec
                target_ids.add(pid)

    print(f"[*] Đã nạp thành công {len(target_ids)} bài báo mục tiêu.")

    matched_ids = set()
    completed_shards = set() 

    if CHECKPOINT.exists():
        try:
            with open(CHECKPOINT, "r", encoding="utf-8") as f:
                content = f.read()
                if content:
                    cp = json.loads(content)
                    matched_ids = set(cp.get("matched", []))
                    completed_shards = set(cp.get("completed_shards", []))
                    print(f"[*] Phục hồi: Đã xong {len(completed_shards)} shard, hệ thống đã khớp được {len(matched_ids)} bài.")
        except: pass

    all_shards = get_shard_urls()
    if not all_shards: return
    total_shards = len(all_shards)

    for batch_start in range(0, total_shards, SHARD_BATCH_SIZE):
        batch_end = min(batch_start + SHARD_BATCH_SIZE, total_shards)
        
        fresh_urls = get_shard_urls()
        if not fresh_urls: break

        shards_to_process = [
            (idx, fresh_urls[idx]) 
            for idx in range(batch_start, batch_end) 
            if idx not in completed_shards
        ]

        if not shards_to_process: continue

        print(f"\n[BATCH] quét từ Shard {batch_start} đến {batch_end-1} ({len(shards_to_process)} shards)")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_shard = {
                executor.submit(process_shard, url, idx, target_ids, papers, matched_ids): idx 
                for idx, url in shards_to_process
            }
            
            for future in concurrent.futures.as_completed(future_to_shard):
                shard_idx = future_to_shard[future]
                try:
                    _, num_matched, is_success = future.result()
                    
                    if is_success:
                        print(f"✅ [Shard {shard_idx}] XONG! (Khớp được {num_matched} bài báo)")
                        with write_lock:
                            completed_shards.add(shard_idx)
                            with open(CHECKPOINT, "w", encoding="utf-8") as f:
                                dumped = json.dumps({
                                    "matched": list(matched_ids), 
                                    "completed_shards": list(completed_shards)
                                })
                                if isinstance(dumped, bytes):
                                    dumped = dumped.decode('utf-8')
                                f.write(dumped)
                    else:
                        print(f"[Shard {shard_idx}] bị lỗi, sẽ quét lại")
                except Exception as exc:
                    print(f"Shard {shard_idx} văng lỗi: {exc}")

    print(f"\n Hoàn thành, tổng báo thu được: {len(matched_ids)}")

if __name__ == "__main__":
    main()
