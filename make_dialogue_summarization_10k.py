# -*- coding: utf-8 -*-
# 한글 깨짐 방지: 원본 인코딩 자동 감지 후 10,000줄만 UTF-8로 저장
import chardet

src = r".\data\processed\dialogue_summarization.jsonl"
dst = r".\data\processed\dialogue_summarization_10k.jsonl"

# 1. 인코딩 감지
with open(src, 'rb') as f:
    raw = f.read(1000000)  # 1MB만 샘플
    enc = chardet.detect(raw)['encoding']
    print(f"Detected encoding: {enc}")

# 2. 10,000줄만 읽어서 UTF-8로 저장
count = 0
with open(src, 'r', encoding=enc) as fin, open(dst, 'w', encoding='utf-8') as fout:
    for line in fin:
        fout.write(line)
        count += 1
        if count >= 10000:
            break
print(f"Wrote {count} lines to {dst}")
