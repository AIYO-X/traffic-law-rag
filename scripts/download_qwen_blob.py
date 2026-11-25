# -*- coding: utf-8 -*-
"""并发分段下载 Ollama 模型 blob，支持断点续传与 SHA256 校验。"""
import hashlib, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BLOB = "2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730"
URL = f"https://registry.ollama.ai/v2/library/qwen2.5/blobs/sha256:{BLOB}"
DEST = Path(r"C:/Users/loser/.ollama/models/blobs") / f"sha256-{BLOB}"
TMP = DEST.parent / "_qwen_parts"
TMP.mkdir(exist_ok=True)
TOTAL = 4683073952
N = 12
CHUNK = (TOTAL + N - 1) // N

def part_path(i):
    return TMP / f"part{i:02d}"

def download_part(i):
    start = i * CHUNK
    end = min(start + CHUNK, TOTAL) - 1
    p = part_path(i)
    existing = p.stat().st_size if p.exists() else 0
    want = end - start + 1
    if existing >= want:
        return i, want, "complete"
    # 若已有部分与期望不符则从头重下
    if existing > 0 and existing < want:
        p.unlink()
        existing = 0
    cmd = ["curl", "-sL", "--retry", "15", "--retry-delay", "2", "--retry-all-errors",
           "--connect-timeout", "20", "-r", f"{start}-{end}", "-o", str(p), URL]
    for attempt in range(6):
        r = subprocess.run(cmd)
        size = p.stat().st_size if p.exists() else 0
        if size >= want:
            return i, want, "complete"
        print(f"  part{i:02d} 第{attempt+1}次: {size}/{want}", flush=True)
        time.sleep(2)
    return i, size, "FAILED"

print(f"开始并发下载 {N} 段，总大小 {TOTAL/1024/1024/1024:.2f} GB", flush=True)
with ThreadPoolExecutor(max_workers=N) as ex:
    futs = [ex.submit(download_part, i) for i in range(N)]
    results = [f.result() for f in as_completed(futs)]

failed = [r for r in results if r[2] != "complete"]
if failed:
    print("以下分段失败:", failed, flush=True)
    sys.exit(1)

print("分段下载完成，正在合并...", flush=True)
with DEST.open("wb") as out:
    for i in range(N):
        p = part_path(i)
        with p.open("rb") as f:
            while True:
                b = f.read(4 * 1024 * 1024)
                if not b:
                    break
                out.write(b)
print("合并完成，校验 SHA256...", flush=True)
h = hashlib.sha256()
with DEST.open("rb") as f:
    for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
        h.update(b)
digest = h.hexdigest()
print(f"实际: sha256:{digest}", flush=True)
print(f"期望: sha256:{BLOB}", flush=True)
if digest != BLOB:
    print("校验失败！", flush=True)
    sys.exit(1)
print("校验通过，清理临时分段", flush=True)
for i in range(N):
    part_path(i).unlink(missing_ok=True)
TMP.rmdir()
print("MODEL_BLOB_READY", flush=True)
