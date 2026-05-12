# Term Project 제안: Personal Real-World Tool Calling — 본인 사용 데이터로 학습한 작은 모델이 큰 모델을 능가할 수 있는가?

> **분야**: Tool Calling / Function Calling + Small Language Models + Trajectory-Aware Evaluation
> **제안자**: 손성준
> **작성일**: 2026-05-12
> **관련 후보**: 1차 제안 `sonsj-proposal.md` (Grokking + Code Variable Binding)과는 결이 완전 다르며, Step 4(옵션)에서 mechanistic 분석 연결 가능

---

## 주제

> **"본인이 1년간 실사용한 Claude Code 데이터(7천 페어)로 fine-tuning한 350M~1.5B 작은 모델이, 동일 personal task에서 GPT-4·Claude 같은 큰 일반 모델을 trajectory 효율성·정확도 모두에서 능가할 수 있는가?"**

본 프로젝트는 tool calling 분야의 **#1 challenge — "real-world function-calling data scarcity"** — 에 unique 해결책을 제시한다. 본인이 실제 Claude Code를 사용하며 자동 수집·업로드한 7,000+ trajectory 페어를 활용하여, **small targeted model이 personal domain에서 large general model을 trajectory 효율성에서 능가하는지** 검증한다.

본 분야는 ICLR 2026의 가장 많이 등장한 키워드 **"Agent"** 의 핵심 sub-area이며, BFCL V4(Berkeley Function Calling Leaderboard), TRAJECT-Bench 등 활발한 표준화가 진행 중이다. 또한 Amazon의 2025.12 페이퍼 "Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning" (arXiv:2512.15943) 가 small targeted model의 가능성을 입증한 직후이며, 본 프로젝트는 이 결과를 **personal domain으로 확장 검증**한다.

### 19개 기존 후보 + 1차 제안과의 차별점

* 19개 기존 후보: 모두 응용 앱 (이미지·텍스트·오디오 처리 서비스).
* 1차 제안 (MI 후보): 작은 toy transformer scratch 학습 + 회로 분석.
* **본 제안**: Production-level tool calling 모델 + **본인 실사용 데이터** + trajectory-aware 평가.

### 19개 후보 #2 (SLLM Tool Calling Agent)와의 결정적 차별

| 차원 | 19개 #2 (SLLM Tool Calling) | 본 제안 |
|------|---------------------------|--------|
| 데이터 | 공개 데이터셋 (xLAM, ToolBench) — **synthetic** | **본인 실사용 7,000+ 페어 (real-world)** |
| 평가 | Tool selection accuracy 단일 | **Trajectory length, efficiency, recovery 등 multi-dimensional** |
| 모델 비교 | 일반 fine-tuning vs base | **Small targeted vs Large general (compute-matched)** |
| 분석 깊이 | 단순 성능 비교 | **Trajectory 효율성 분석 + (옵션) mechanistic** |

---

## 배경: Tool Calling과 Trajectory란?

### Tool Calling이란?

**한 줄 정의**: LLM이 외부 도구(API, 함수, MCP server 등)를 호출하여 task를 수행하는 기능.

```python
# 예시: 사용자가 "이 코드 버그 찾아줘"
User: "main.py의 버그 찾아서 고쳐줘"

LLM: 
  Step 1: call Read(file="main.py")
  Step 2: call Bash("python main.py")  # 에러 확인
  Step 3: call Edit(file, old, new)     # 수정
  Step 4: call Bash("python main.py")  # 검증
  Step 5: "수정 완료. NoneType 오류를 해결했습니다."
```

대표 도구: Bash, Read, Edit, Write, Grep, WebSearch, WebFetch 등.

### Trajectory란?

**한 줄 정의**: Task를 푸는 **전체 행동 sequence** — 어떤 tool을 어떤 순서로 호출했고, 각 결과를 보고 다음 행동을 어떻게 결정했는지의 흐름.

같은 task에 대해서도 trajectory는 다를 수 있다:

```
[효율적 trajectory] — 4 step
Read → Bash → Edit → Bash(검증)

[비효율적 trajectory] — 12 step
Read → Read → Bash → Bash → Grep → Edit(잘못) → 
Bash → Read → Edit(재시도) → Bash → Edit(또) → Bash
```

→ **같은 결과지만 효율 차이 3배**. 실제 사용 시 비용·시간·신뢰성 모두 차이.

### 왜 Trajectory가 중요한가 (2025-2026 트렌드)

전통적 평가는 **"최종 답이 맞나"** 만 봄. 최근 연구(TRAJECT-Bench, TRACE framework, "Beyond the Final Answer" 등)는 trajectory 자체의 평가가 필요하다고 주장.

* TRAJECT-Bench: trajectory-aware tool use 평가 표준
* TRACE framework: efficiency, hallucination, adaptivity 등 multi-dimensional 평가
* "Beyond the Final Answer": "정답 일치율만으로는 agent 평가 부족" — ICLR 2026

→ **본 프로젝트는 이 trajectory-aware 평가를 본인 데이터에 직접 적용**.

---

## 핵심 연구 질문

> **"Personal Domain에서 Small Targeted Model이 Large General Model을 능가하는가? — Trajectory 효율성 관점에서"**

### 가설

* **H1 (Performance)**: 본인 데이터로 targeted fine-tuning한 1.5B 모델이 personal task accuracy에서 base 1.5B 대비 ≥30% 향상.
* **H2 (Efficiency)**: Fine-tuned 모델의 trajectory가 base 모델 대비 평균 step 수 ≥30% 감소.
* **H3 (Competitive)**: 본인 personal task에서 fine-tuned 1.5B가 GPT-4·Claude의 90% 이상 성능 달성.
* **H4 (Trade-off)**: Personal fine-tuning이 general capability (HumanEval 등)를 어느 정도 손상시키는지 정량 측정.

세 가지 가능한 결론 시나리오:

### Case A — "Small + Targeted Wins"
H1~H3 모두 입증. Personal domain에서 작은 모델이 큰 모델 능가.
→ **"본인 데이터의 unique 가치 + Small Model의 효율성 입증"**

### Case B — "Large General Wins"
H1, H2는 성립하지만 H3 미달성.
→ **"Personal fine-tuning은 효과 있지만 큰 모델 일반 능력 못 따라잡음"**

### Case C — "Trade-off Severe"
H4가 심각함. Personal 성능 향상 vs general 성능 손상 trade-off 큼.
→ **"Personal specialization의 한계 정량화"**

**세 결과 모두 학술적·실용적 가치 있음**. 실패 없는 실험 디자인.

---

## 대략적인 계획

### Step 1: Personal Tool Calling 벤치마크 구축 (Week 1)

본인 사용 데이터(이미 HuggingFace `SuperPowerMz/claude-code-personal-v1`에 자동 수집됨, 2026-05-12 기준 sft 2,388 / guppy 7,352 / dpo 179)에서 평가용 벤치마크 구축.

* **Task 분류**: 본인 사용 task를 카테고리별 분류
  * 코드 디버깅, 파일 분석, 시스템 명령, 검색, 문서 작성 등
* **Train/Test split**: Trajectory time 기반 (최근 trajectory를 test로) → leakage 방지
* **Trajectory 추출**: 각 trajectory의 step sequence, tool calls, results, 최종 답변 파싱
* **TRACE-style metric 정의**:
  * Tool selection accuracy (적절한 tool 선택 비율)
  * Trajectory length (평균 step 수)
  * Trajectory efficiency ratio (정답 / step 수)
  * Hallucination rate (잘못된 arg 호출 비율)
  * Recovery rate (error 후 복구 성공 비율)

### Step 2: Small Model Fine-tuning (Week 2)

* **Base 모델**: Qwen2.5-Coder 시리즈 (0.5B / 1.5B / 7B) — 한국어·코드 친화
* **방법**:
  * **SFT (Supervised Fine-Tuning)**: 본인 sft 페어 (2,388개)로 trajectory 학습
  * **DPO (Direct Preference Optimization)**: 본인 dpo 페어 (179개) 활용
  * **QLoRA 4-bit**: RTX 4090 가능
* **Targeted fine-tuning** (Amazon 2512.15943 방식): 일반 데이터 X, 본인 task만 학습
* **Ablation**:
  * SFT only vs SFT+DPO
  * 모델 크기별 (0.5B / 1.5B / 7B)
  * Targeted vs general data

### Step 3: Compute-Matched 비교 실험 (Week 3)

* **비교군**:
  * Fine-tuned 0.5B / 1.5B / 7B (우리 모델)
  * Base 0.5B / 1.5B / 7B (fine-tuning 없음)
  * GPT-4 / Claude (API)
* **평가**:
  * 본인 personal task test set
  * BFCL (Berkeley Function Calling Leaderboard) general task — 일반 능력 손상 측정
  * HumanEval — 코딩 능력 손상 측정
* **Compute-matched**:
  * 동일 token budget 하에서 비교
  * Cost ($/correct trajectory) 정량 측정
* **시각화**:
  * Pareto frontier: cost vs accuracy
  * Trajectory length 분포 histogram
  * Recovery rate 비교

### Step 4: (옵션) Mechanistic 분석 — Tool Selection의 내부 표현 (Week 4)

본 단계는 1차 후보(Mechanistic Interpretability)와의 narrative 시너지를 위한 옵션. 시간 여유 시 진행.

* **2605.07990 ("Tool Calling is Linearly Readable and Steerable")** 방법론 적용
* 본인 fine-tuned 모델 vs base 모델의 tool selection direction 비교
* "Personal fine-tuning이 hidden state를 어떻게 변경하는가" 분석
* Per-tool mean activation 추출 + linear probing
* (가능하다면) Tool selection 회로의 specific attention head 식별

### 최종 산출물 + 발표 준비 (Week 4)

* Trajectory length 비교 histogram
* Cost vs Accuracy Pareto frontier (한 슬라이드 결론)
* Personal vs General capability trade-off 그래프
* (옵션) Tool selection direction 시각화
* Live demo: 본인 모델 vs base 모델 vs GPT-4 동일 task 비교 영상
* 최종 리포트

---

## 실제 활용 데이터셋

### 주 데이터셋

* **본인 Claude Code 실사용 데이터** (HuggingFace `SuperPowerMz/claude-code-personal-v1`, private)
  * **sft**: 2,388 페어 (ChatML format)
  * **guppy**: 7,352 페어 (raw chunk for Guppy LM)
  * **pretrain**: 2,388 페어
  * **dpo**: 179 페어 (chosen/rejected)
  * 4대 호스트(Mac, 홈서버, 회사 서버 2대)에서 매일 03:00 cron으로 자동 수집·MinIO에 저장·PII 마스킹 후 HF push 자동화
  * 데이터 무료 + 자동 증가 중

### Test 벤치마크

* **본인 데이터의 시간 기반 split test set** (최근 trajectory)
* **BFCL** (Berkeley Function Calling Leaderboard, V4) — general tool calling 평가
* **HumanEval** — 코딩 능력 손상 측정
* **TRAJECT-Bench** (옵션) — trajectory-aware 평가 비교

### 보조 데이터셋 (옵션)

* **xLAM Function Calling 60k** (Salesforce) — general baseline 비교용
* **ToolBench** — general tool calling 데이터

### 모델 (HuggingFace 공개)

* **Qwen2.5-Coder-0.5B-Instruct** (350M~)
* **Qwen2.5-Coder-1.5B-Instruct** (1.5B)
* **Qwen2.5-Coder-7B-Instruct** (7B)
* (비교) GPT-4, Claude (API, 비용 발생)

### 데이터셋 선정 이유

* **본인 실사용 데이터 = 분야 #1 challenge 해결**: "Real function-calling data scarcity" (industrial survey 결론) 의 unique 해결책
* **자동 수집 인프라**: 이미 구축, 실험 중에도 데이터 누적 증가
* **PII 마스킹 완료**: Email, API key 등 자동 마스킹된 안전한 데이터
* **Trajectory 완전 포함**: tool call sequence, results, 최종 답변 모두 보존

---

## 예상 결과

### 1. 정량적 결과

```text
[Personal Domain Test (본인 task)]
                  Accuracy    Avg Trajectory    Recovery Rate
Base 1.5B:        35%         5.2 steps          22%
Fine-tuned 1.5B:  72%         2.8 steps          61%   ← 우리 모델
Base 7B:          48%         4.1 steps          35%
Fine-tuned 7B:    81%         2.4 steps          74%   ← 우리 모델
GPT-4:            78%         3.1 steps          68%
Claude Opus:      82%         2.9 steps          76%

→ "1.5B fine-tuned가 GPT-4 92% 도달, trajectory 길이는 90% 짧음"

[General Capability Test (BFCL, HumanEval)]
                  BFCL        HumanEval
Base 1.5B:        62%         48%
Fine-tuned 1.5B:  55% (-7%)   42% (-6%)
                                  ↑ trade-off 측정

[Compute-Matched 비교]
Cost ($) per correct trajectory:
  Fine-tuned 1.5B:  $0.001
  GPT-4:            $0.05
  → 50배 비용 절감
```

### 2. 시각화 결과물

* **Trajectory length 비교 histogram** (모델별)
* **Cost vs Accuracy Pareto frontier**
* **Personal vs General trade-off scatter plot**
* **Step별 tool 사용 빈도 비교**
* **Recovery 영상**: error 후 복구 trajectory 시각화
* (옵션) **Tool selection direction 시각화**: fine-tuning 전후 비교

### 3. 발표 임팩트 포인트

* **"개인 데이터 7천 페어로 1.5B 모델이 GPT-4 92% 도달"** — 한 줄 결론
* **Trajectory 효율성**: "Base는 5.2 step, 우리 모델은 2.8 step" — 시각적 임팩트
* **Cost 비교**: "GPT-4 대비 50배 비용 절감"
* (옵션) **Mechanistic**: "Tool selection을 위한 linear direction이 fine-tuning으로 강화됨"

---

## 이 실험으로 알게 되는 것

### Layer 1: 실험이 직접 답하는 질문

**"Personal Domain에서 Small Targeted Model이 Large General Model을 trajectory 효율성과 정확도에서 능가하는가?"**

세 시나리오 (A/B/C) 중 어느 것이 성립하는지 명확한 답.

### Layer 2: 구체적으로 손에 잡히는 발견

1. **Personal task accuracy/efficiency 정량 수치**: 모델 크기별 trade-off curve
2. **Trajectory 효율성**: fine-tuning이 trajectory를 얼마나 단축하는지
3. **Compute efficiency**: $/correct trajectory 정확한 비교
4. **General capability 손상량**: targeted fine-tuning의 cost
5. **DPO의 효과**: 179 페어로 trajectory가 얼마나 개선되는지
6. (옵션) **Tool selection direction**: fine-tuning이 model 내부에 미치는 영향

→ 이는 **누구도 보유하지 못한 personal data 기반의 새로운 데이터 포인트**.

### Layer 3: 학계·실용 측면 함의

**학계 측면**:
* Amazon 2512.15943 ("Small targeted wins") 결과를 **personal real-world data로 확장 검증**
* TRAJECT-Bench/TRACE framework의 personal domain 적용 첫 시도
* "Real-world function-calling data scarcity" 분야 #1 challenge에 unique 해결책 제시

**실용 측면**:
* **개인화 코딩 비서**: Edge device에서 작동 가능한 personal model 가능성
* **비용 절감**: GPT-4 대비 50배 cost 감소 가능성 입증
* **Privacy**: Personal data를 외부 API로 보내지 않고 로컬 처리
* **Trajectory efficiency**: 단순 정확도 너머의 새 평가 axis 제시

### Layer 4: 한계 (정직하게)

본 실험이 **답할 수 없는 것**:
* **일반화**: 한 사람의 데이터 결과가 다른 사람에게도 적용되는가 (single-subject limitation)
* **확장성**: 7B 너머 큰 모델에서도 같은 패턴 유지되는가
* **장기 효과**: Fine-tuning 후 다른 task로 transfer learning 시 어떻게 되는가
* **Personal task 정의**: "Personal task"의 정확한 정의·경계

답할 수 있는 것: **"한 학생의 1년치 Claude Code 데이터로 1.5B 모델 fine-tuning 시 personal domain에서 GPT-4 대비 trajectory 효율성과 정확도가 어떻게 변하는가"** 까지.

→ 작은 발견이지만 **명확하고 측정 가능한 발견**.

---

## 강의 개념 연결

| 강의 개념 | 본 프로젝트에서 다루는 방식 |
|----------|---------------------------|
| Transformer architecture | Qwen2.5-Coder의 decoder-only transformer 활용 |
| Attention mechanism | Tool selection 시 attention pattern 분석 (Step 4) |
| Fine-tuning / Transfer Learning | QLoRA를 활용한 효율적 fine-tuning |
| RLHF / DPO | Direct Preference Optimization으로 trajectory 효율 학습 |
| Pretrained model 활용 | Qwen2.5-Coder base model 사용 |
| Trajectory / Sequence modeling | 시간 sequence 기반 행동 학습 |
| Evaluation metric design | TRACE-style multi-dimensional 평가 metric 직접 정의 |
| 연구 방법론 | 가설 → 실험 → 결과 해석 + ablation |

---

## 교수님 평가 기준 대응

| 예상 질문 | 답변 |
|----------|------|
| 왜 이 모델? | Qwen2.5-Coder는 코드·한국어 친화적이고 0.5B~7B 크기 옵션 제공 → compute-matched ablation 가능. 너무 큰 모델은 RTX 4090에서 불가. |
| 왜 이 데이터셋? | 본인 실사용 7천 페어는 **real-world function-calling 데이터로, 분야 #1 challenge에 unique 해결책**. Synthetic 데이터의 한계 (coverage, accuracy) 극복. 자동 수집·PII 마스킹 인프라 완비. |
| 왜 이 평가지표? | 단순 accuracy가 아닌 **TRACE-style multi-dimensional metric** (trajectory length, efficiency, recovery 등). 2026 트렌드 (TRAJECT-Bench, "Beyond the Final Answer")와 일치. |
| Compute-matched가 왜 중요? | "Small Targeted vs Large General" 비교에서 token budget 통제 필수. 그렇지 않으면 fair comparison 아님. (2604.02460 single vs multi-agent 결과와 같은 맥락) |
| 결과 해석 논리? | Personal task accuracy 측정 → Trajectory 효율성 측정 → General capability trade-off 측정 → Compute cost 비교. **각 단계가 다음의 가설 검증**. |
| 일반화 가능성? | 정직히 단일 사용자 데이터 결과라 일반화 한계 인정. 하지만 framework 자체는 다른 사용자에게도 적용 가능. (future work) |

---

## 위험 요소 + 대응

| 위험 | 가능성 | 대응 |
|------|--------|------|
| Test data leakage (본인 데이터 자체 평가) | 중간 | 시간 기반 split (최근 trajectory만 test), task category 다양화 |
| GPT-4 API 비용 | 중간 | 작은 test set (200~500 trajectory)로 제한, $30~50 예산 |
| Fine-tuning 시간 초과 | 낮음 | QLoRA 4-bit, RTX 4090 단일 GPU, 1.5B 모델 기준 약 6~12시간 |
| Trajectory parsing 복잡함 | 중간 | HF dataset이 이미 ChatML format으로 정리되어 있음, 추가 파싱 최소화 |
| General capability 손상 너무 큼 | 중간 | 그 자체가 발견 (Case C). 정량 측정으로 trade-off 입증. |
| Mechanistic 분석 (Step 4) 시간 부족 | 높음 | Step 4는 옵션. Step 1~3만 완성해도 핵심 결과 확보. |
| 본인 데이터의 카테고리 편중 | 중간 | 데이터 분석으로 카테고리 분포 명시 + 한정된 task에서의 결론만 주장 |

**핵심 안전망**: Step 1~3까지 완성하면 기본 점수 확보. Step 4 (mechanistic)은 차별화 가산점.

---

## 1차 후보 (MI)와의 관계

| 차원 | 1차 (MI) | 본 제안 (Tool Calling) |
|------|---------|---------------------|
| 모델 | 작은 toy + GPT-2 small | Production-scale (Qwen2.5-Coder) |
| 데이터 | 합성 (modular arithmetic, code binding) | **본인 real-world 7천 페어** |
| Method | Mechanistic 분석 (회로 발견) | Fine-tuning + Trajectory 평가 |
| 결 | 분석·연구형 | **응용·실용형** |
| 발표 narrative | "Transformer 내부 동작 reverse-engineering" | **"개인 데이터로 만든 작고 효율적인 비서"** |
| Step 4 (옵션) | - | Tool selection mechanistic 분석 → **1차 narrative와 연결** |

→ 두 후보가 함께 채택되면 **"분석 + 응용 양쪽 cover"** 라는 강한 voting 메시지 형성.

---

## 참조

### 핵심 페이퍼

* [Amazon, "Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning" (2025.12, arXiv:2512.15943)](https://arxiv.org/abs/2512.15943) — 본 제안의 직접 기반
* [BFCL: From Tool Use to Agentic Evaluation (OpenReview)](https://openreview.net/forum?id=2GmDdhBdDk) — 표준 벤치마크
* [TRAJECT-Bench: Trajectory-Aware Benchmark (arXiv:2510.04550)](https://arxiv.org/html/2510.04550v1) — Trajectory 평가 표준
* [TRACE: "Beyond the Final Answer" (arXiv:2510.02837)](https://arxiv.org/abs/2510.02837) — Multi-dimensional trajectory 평가
* [SE-Agent: Self-Evolution Trajectory Optimization (arXiv:2508.02085)](https://arxiv.org/html/2508.02085v6)
* [ToolACE: Winning the Points of LLM Function Calling (OpenReview)](https://openreview.net/forum?id=8EB8k6DdCU)
* [Function Calling Industrial Practices and Challenges](https://openreview.net/pdf/d01d50e27f7636724789f2aad6f4ac378749a0e1.pdf) — 분야 #1 challenge "data scarcity" 명시

### Mechanistic 분석 (Step 4 옵션)

* [Tool Calling is Linearly Readable and Steerable in Language Models (arXiv:2605.07990, 2026.05)](https://arxiv.org/html/2605.07990)
* [Beyond the Black Box: Interpretability of Agentic AI Tool Use (arXiv:2605.06890, 2026.05)](https://arxiv.org/html/2605.06890)
* [LLM Agents Already Know When to Call Tools (arXiv:2605.09252, 2026.05)](https://arxiv.org/html/2605.09252v1)

### 라이브러리 + 모델

* [Qwen2.5-Coder (HuggingFace)](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
* [QLoRA / PEFT](https://github.com/huggingface/peft)
* [TRL (Transformer RL, DPO 지원)](https://github.com/huggingface/trl)
* [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)

### 본인 데이터셋

* **HuggingFace**: `SuperPowerMz/claude-code-personal-v1` (private)
* 자동 수집 인프라: 홈서버 MinIO + 4 클라이언트 sync + 일일 03:00 cron PII masking + HF push
* 2026-05-12 기준: sft 2,388 / guppy 7,352 / dpo 179 / pretrain 2,388

### 분야 위상 (제안 정당화)

* [ICLR 2026 Trends: Agentic AI](https://encord.com/iclr-2026/) — "Agent" 가 가장 많은 키워드
* [Gartner: Multi-agent system inquiries 1,445% growth (Q1 2024 → Q2 2025)](https://www.gartner.com/en/articles/hype-cycle-for-agentic-ai)
* [Tool Use and Function Calling Standards (Zylos Research, 2026.04)](https://zylos.ai/research/2026-04-07-tool-use-function-calling-standards-benchmarks)
