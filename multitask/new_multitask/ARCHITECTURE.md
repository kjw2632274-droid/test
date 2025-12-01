# Multi-Head KoBART Architecture

본 문서는 `new_multitask/model.py`에 구현된 멀티헤드 KoBART 아키텍처(공유 인코더 + 태스크별 디코더 4개)를 설명합니다.

## 개요
- 기반 모델: Hugging Face KoBART(`BartForConditionalGeneration` 계열 체크포인트)
- 공유 자원: 임베딩(shared embedding), 인코더(encoder)
- 태스크별 자원: 디코더(decoder) × 4, LM Head × 4 (각 태스크 전용)
- 태스크 키: `qa`, `role_generation`, `style_transfer`, `dialogue_summarization`
- 가중치 초기화: 모두 KoBART 사전학습 가중치에서 시작 (디코더는 KoBART 디코더를 deep copy)
- 임베딩-헤드 타이잉: 4개 LM Head의 가중치는 shared embedding과 tie

간단히 말해, “하나의 KoBART 인코더를 공유하고, 4개의 디코더+헤드로 태스크를 분리”하는 구조입니다.

## 구성요소
```
MultiHeadKoBART
├─ shared_embedding  (KoBART shared embedding)
├─ encoder           (KoBART encoder)
├─ decoders          (ModuleDict)
│   ├─ qa                    : BartDecoder (KoBART decoder 복제)
│   ├─ role_generation       : BartDecoder (KoBART decoder 복제)
│   ├─ style_transfer        : BartDecoder (KoBART decoder 복제)
│   └─ dialogue_summarization: BartDecoder (KoBART decoder 복제)
└─ lm_heads          (ModuleDict)
    ├─ qa                    : Linear(d_model → vocab), weight tied to shared_embedding
    ├─ role_generation       : Linear(d_model → vocab), weight tied to shared_embedding
    ├─ style_transfer        : Linear(d_model → vocab), weight tied to shared_embedding
    └─ dialogue_summarization: Linear(d_model → vocab), weight tied to shared_embedding
```

## 태스크 라우팅
- 호출 시 `task_name` 인자로 디코더/헤드를 선택합니다.
- 입력 인코딩은 항상 공유 인코더로 수행 후, 선택된 디코더로 디코딩합니다.
- 로스 계산과 로짓 생성은 선택된 LM Head를 통해 이뤄집니다.

## 초기화/불러오기
- `MultiHeadKoBART.from_pretrained(base_model_name_or_path)`
  - 예: `gogamza/kobart-base-v2`
  - KoBART 가중치로 shared embedding/encoder/decoder를 초기화하고, 디코더는 태스크 수만큼 복제합니다.
- 토크나이저: KoBART 토크나이저를 함께 사용해야 `vocab_size`, `special tokens`가 일치합니다.

## Forward 경로
1. 입력(`input_ids`, `attention_mask`) → 공유 인코더
2. `task_name`에 맞는 디코더 선택 → 디코딩(`decoder_input_ids`가 없으면 `labels`로부터 shift)
3. 태스크별 LM Head 통과 → `logits`
4. `labels`가 있으면 `CrossEntropyLoss(ignore_index=pad)` 계산

반환: `{ 'loss', 'logits', 'encoder_last_hidden_state', 'decoder_hidden_states' }`

## Generate (테스트용)
- `generate(...)`는 간단한 greedy 디코딩을 제공합니다.
- 프로덕션에서는 HF의 beam search/top-k/top-p 등으로 확장하는 것을 권장합니다.

## 사용 예시
```python
from transformers import PreTrainedTokenizerFast
from new_multitask.model import MultiHeadKoBART

model = MultiHeadKoBART.from_pretrained('gogamza/kobart-base-v2')
model.print_model_info()

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
enc = tokenizer("[style_transfer] 안녕하세요. 저는 고양이 6마리 키워요.", return_tensors='pt')

out = model(
    input_ids=enc['input_ids'],
    attention_mask=enc['attention_mask'],
    labels=enc['input_ids'],
    task_name='style_transfer',
)
print('loss:', float(out['loss']))

gen_ids = model.generate(
    input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], task_name='style_transfer', max_length=32
)
print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
```

PowerShell 실행 예시
```powershell
# 환경변수로 KoBART 체크포인트 지정 가능
$env:KOBART_MODEL = "gogamza/kobart-base-v2"
py .\new_multitask\model.py
```

## 프롬프트 권장 포맷(예)
- `style_transfer`: "[style_transfer] {formal}"
- `dialogue_summarization`: "대화 요약: {dialogue}"
- `role_generation`: "역할: {role} | 대화: {context}"
- `qa`: "질문: {question}"

데이터/인퍼런스 모두 동일한 포맷을 유지해야 성능이 안정적입니다.

## 저장/불러오기 주의
- 현재 클래스의 `from_pretrained`는 KoBART 베이스에서 **초기화**하기 위한 헬퍼입니다.
- 멀티헤드 가중치를 학습 후 지속적으로 사용하려면 다음 중 하나를 권장합니다.
  1) `state_dict`로 저장/로드:
     ```python
     torch.save(model.state_dict(), 'multihead_kobart.pt')
     model.load_state_dict(torch.load('multihead_kobart.pt', map_location='cpu'))
     ```
  2) 필요 시 `save_pretrained`/`from_pretrained` 커스텀 구현 추가 (config에 태스크 구조를 직렬화)

## 메모리/성능 고려
- 디코더를 4개 복제하므로 베이스 KoBART 대비 디코더 파라미터가 4배가 됩니다.
- 가능한 경우 디코더 일부(하위 블록) 공유/어댑터/LoRA 등을 도입하면 파라미터와 VRAM을 절약할 수 있습니다.

## 확장
- 태스크 추가: `TASKS` 목록과 `task_names`에 키 추가 → 디코더/헤드 자동 생성
- 디코딩 고급화: HF generate API로 beam, top-k/p, length penalty 등 적용

## 기존 멀티태스크 모델과의 차이
- 기존: 경량 BART 구성(d_model=128, 2/2 레이어) + 디코더 그룹 2개
- 현재: **KoBART 사전학습 인코더** 공유 + **태스크별 디코더 4개** (모두 KoBART 가중치 기반)
- 목적: 베이스 언어모델의 지식을 그대로 물려받아 태스크 간 간섭을 줄이고 초기 성능을 높임

## 체크리스트
- [ ] KoBART 토크나이저 사용 (vocab/special tokens 일치)
- [ ] 태스크 프롬프트/데이터 포맷 일관성 유지
- [ ] 학습 저장 시 `state_dict`로 보존 또는 커스텀 save_pretrained 구현
- [ ] 디코딩 전략(beam/top-k/p) 필요 시 확장

---
문의/개선 요청이 있으면 `new_multitask/model.py`와 함께 조정 가능합니다.
