# Term Project 제안: MoE Expert는 한국어와 영어를 다르게 처리하는가? — Multilingual Specialization의 Mechanistic 분석

> **분야**: Mixture of Experts (MoE) + Mechanistic Interpretability + Multilingual NLP
> **제안자**: 손성준
> **작성일**: 2026-05-12
> **관련 후보**: 1차 제안 `sonsj-proposal.md` (Grokking + Code Variable Binding)와 결은 비슷하나 **production-scale 모델 + 다국어 차원**으로 차별화

---

## 주제

> **"Mixtral 같은 Mixture-of-Experts(MoE) 모델 안에서, 다른 언어(한국어 vs 영어)를 다른 expert가 전담 처리하는가? 아니면 모든 expert가 언어 무관하게 협업하는가?"**

본 프로젝트는 production-scale MoE 모델의 expert routing을 **언어별로 분석**하여, **multilingual specialization 패턴을 mechanistic하게 해부**한다.

본 분야는 2026년 ICLR에서 가장 핫한 키워드 중 하나인 **MoE 아키텍처**와 **Mechanistic Interpretability**의 교차점이다. Mixtral 8×7B, DeepSeek-MoE-16B, OLMoE 등이 production 환경에서 빠르게 채택되고 있지만, **"expert가 언어별로 specialize되는가"는 거의 연구되지 않은 영역**이다.

### 19개 기존 후보 + 1차 제안과의 차별점

* 19개 기존 후보: 모두 응용 앱 (이미지·텍스트·오디오 처리 서비스).
* 1차 제안 (MI 후보): 작은 toy transformer + GPT-2 small의 **회로 분석**.
* **본 제안**: Production-scale MoE 모델의 **expert routing × 언어** 분석.
* → 1차와 같이 mechanistic 분석이지만, **모델 scale·아키텍처·핵심 질문 모두 다름**. "트랜스포머의 모든 scale에서 내부 동작을 본다"는 narrative 형성 가능.

---

## 배경: MoE와 Expert Specialization이란?

### Mixture of Experts (MoE)란?

**한 줄 정의**: 하나의 모델 안에 여러 "전문가(expert)" sub-network를 두고, 입력 token마다 **router가 어떤 expert로 보낼지 결정**하는 아키텍처.

```
입력 token → [Router] → Expert 1, Expert 2, ..., Expert N 중 top-K 선택
              ↓
           선택된 expert들이 처리 → 출력
```

대표 모델:
* **Mixtral 8×7B** (Mistral): 8 experts, top-2 활성화, 47B 총 파라미터지만 13B만 활성
* **DeepSeek-MoE-16B**: 64 experts + shared experts
* **OLMoE-1B-7B** (Allen AI): 64 experts, top-8 활성

**왜 MoE인가**: 총 모델 크기는 크지만 token당 활성 파라미터는 작아서 **computation 효율적**. GPT-4, Gemini도 MoE로 알려짐.

### Expert Specialization이란?

가설: "Expert들이 특정 패턴·도메인·언어에 specialize되어 있다."

기존 연구:
* Mixtral 논문은 **expert가 도메인별 specialization 약함** 보고 (syntactic 패턴만 약하게 분리)
* DeepSeek-MoE는 fine-grained expert로 specialization 강화
* **언어별 specialization은 거의 안 다뤄짐**

### 한국어가 왜 흥미로운가

* 한국어는 영어와 **형태론적·통사론적으로 매우 다름** (교착어, SOV, 조사 등)
* 모델이 한국어를 **별도 expert에 분리**하면 → "MoE가 typology를 학습한다"는 증거
* 분리 **안 되면** → "Expert는 언어 추상 표현을 공유한다"는 증거
* 두 결론 모두 흥미롭고 publishable한 발견

---

## 핵심 연구 질문

> **"Mixtral의 8 expert는 한국어와 영어 처리를 어떻게 분담하는가?"**

세 가지 가능한 결론:

### Case A — Language-Specialized Experts

**발견**: Expert별로 한국어/영어 routing 빈도가 명확히 갈림.

**무슨 뜻인가**: MoE 모델이 학습 중 언어를 **분리해서 모듈화**. Routing이 언어 detection 역할.

**일상 비유**: 다국어 회사에서 "한국어 팀"과 "영어 팀"이 분리되어 있고, 들어오는 문서 언어에 따라 다른 팀이 처리.

**실용적 함의**:
* 특정 expert를 ablate하면 한 언어만 망가짐 → 모듈성
* Multilingual fine-tuning이 expert 단위로 가능
* 저자원 언어 추가 시 새 expert로 처리 가능

### Case B — Shared Experts

**발견**: 모든 expert가 한국어/영어 거의 동일 빈도로 routing.

**무슨 뜻인가**: MoE는 언어를 **추상화된 표현으로 통합 처리**. Expert는 언어 무관한 추상 기능 담당.

**일상 비유**: 회사 직원들이 다국어 가능해서, 언어와 무관하게 task별로 분담.

**실용적 함의**:
* Cross-lingual transfer가 자연스럽게 됨
* Expert는 syntactic/semantic 추상 기능 단위
* 새 언어 추가 시 추가 학습 필요

### Case C — Partial Specialization (가장 가능성 높음)

**발견**: 일부 expert는 언어 specialized, 일부는 shared.

**무슨 뜻인가**: 표면적 처리 (tokenization 인접)는 language-specific, 깊은 추상 처리는 shared.

**실용적 함의**: Layer 별로 specialization 정도가 다를 가능성. 초기 layer는 specialized, 후기 layer는 shared.

→ **세 결론 모두 학술적·실용적 가치 있음**. **실패 없는 실험 디자인**.

---

## 대략적인 계획

### Step 1: 환경 셋업 + Expert Routing 추출 (Week 1)

* Mixtral-8×7B (4-bit 양자화, RTX 4090 추론 가능) 또는 OLMoE-1B-7B 로컬 셋업.
* HuggingFace `output_router_logits=True` 옵션으로 routing 정보 추출 코드 작성.
* 각 token마다:
  * 어느 expert가 선택되었는지 (top-K)
  * Routing weight 값
* Layer × token × expert 3D tensor 형태로 저장.

### Step 2: 한국어/영어 페어 데이터 수집 (Week 1~2)

* **KLUE 벤치마크**에서 한국어 task 1000개
* 동일 의미의 영어 페어 1000개 (직접 번역 또는 기존 multilingual 데이터셋 활용)
* 도메인 분포:
  * 일반 텍스트 (KLUE-NLI, KorNLI)
  * 코드·수학 (한국어 vs 영어 prompt)
  * 일상 대화

### Step 3: Specialization 정량 측정 (Week 2~3)

#### 3-1. Expert × Language 활성화 빈도 매트릭스

* 각 layer × 각 expert가 한국어/영어에 활성화된 비율 계산.
* Heatmap 시각화: 8 expert × N layer × 2 language.

#### 3-2. Specialization Score 정의

$$\text{Spec}(e, l) = \frac{P(\text{expert } e | \text{language } l)}{P(\text{expert } e)} - 1$$

* $> 0$: 해당 expert가 그 언어 선호
* $< 0$: 회피
* $= 0$: 무관

#### 3-3. KL Divergence 측정

* $D_{KL}(P(\text{expert} | \text{한국어}) \| P(\text{expert} | \text{영어}))$
* Layer별 KL 값 → 어느 layer가 가장 language-specific 처리하나

### Step 4: Causal Intervention + 시각화 (Week 3~4)

#### 4-1. Expert Ablation 실험

* 특정 expert를 강제 비활성화 (routing weight 0)
* 한국어/영어 생성 능력 변화 측정 (perplexity, fluency)
* **"L3 Expert 5를 끄면 한국어 perplexity가 ×5 증가, 영어는 변화 없음"** 같은 결과 추구

#### 4-2. Expert Forcing 실험

* 영어 입력에 한국어-선호 expert를 강제 활성화
* 모델 행동 변화 관찰
* "Golden Gate Claude"의 MoE 버전 같은 데모

#### 4-3. 발표 시각화 결과물

* Expert × Language 활성화 heatmap (한 장 슬라이드)
* Layer별 KL divergence 추이 그래프
* Ablation 비디오 ("이 expert 끄니 한국어 망가짐")
* (옵션) Streamlit 인터랙티브 데모

---

## 실제 활용 데이터셋

### 주 데이터셋

* **KLUE benchmark** (한국어)
  * KLUE-NLI, KLUE-STS, KLUE-MRC 등 8개 task에서 텍스트 sample 추출
  * 약 1000~3000개 sample

* **영어 페어 데이터**
  * KLUE 대응되는 영어 task: SuperGLUE, GLUE, SQuAD
  * 또는 KLUE를 직접 영어 번역
  * 1000~3000개 sample

* **합성 코드 데이터** (Step 2 확장, 1차 제안과 연결)
  * 한국어 주석 + 영어 코드 vs 영어 주석 + 코드
  * 1000개 sample

### 보조 데이터셋

* **FLORES-200**: 200+ 언어 평행 corpus, 한국어/영어 페어 확보 가능
* **xP3** (BigScience): multilingual instruction 데이터

### 모델

* **주력**: Mixtral-8×7B 4-bit (Hugging Face)
* **대안 1**: OLMoE-1B-7B (Allen AI, 더 작음)
* **대안 2**: DeepSeek-MoE-16B (한국어 학습 데이터 포함 추정)

### 데이터셋 선정 이유

* **공개 + 표준 벤치마크 사용**: 재현 가능성 보장
* **동일 의미 페어**: 언어가 유일한 변수가 되도록 controlled
* **추론만 사용 (학습 X)**: 컴퓨트 부담 최소화

---

## 예상 결과

### 1. 정량적 결과

```text
[Expert × Language 활성화 빈도 (Mixtral-8×7B 가상 예시)]

Layer 5:
        Expert 0  Expert 1  Expert 2  Expert 3  Expert 4  Expert 5  Expert 6  Expert 7
한국어:    8%       45%        7%       18%        5%        9%        4%        4%   ← Expert 1 한국어 편향
영어:      12%       6%       30%        8%       25%        9%        6%        4%   ← Expert 2,4 영어 편향

[KL Divergence by Layer]
Layer 0:  D_KL = 0.02 (거의 무관)
Layer 5:  D_KL = 0.58 (강한 specialization) ← 중간 layer가 language-specific
Layer 30: D_KL = 0.08 (다시 추상화)

[Ablation 효과]
L5 Expert 1 ablated:
  - 한국어 perplexity: 12.3 → 47.6 (+287%)
  - 영어 perplexity: 8.1 → 9.2 (+14%)
  → L5 Expert 1은 한국어 전담 가능성
```

### 2. 시각화 결과물

* **Expert × Language activation heatmap** (메인 결과)
* **Layer별 KL divergence 추이** 그래프
* **Ablation 비교 영상** ("이 expert 끄면 한국어만 망가짐")
* **Expert specialization tree** (계층적 클러스터링)
* (옵션) Streamlit 인터랙티브 데모

### 3. 발표 임팩트 포인트

* "Mixtral의 8 expert 중 어느 것이 한국어 전담인지 한 슬라이드로 보임" 시각화
* Live ablation 데모: "Expert 끄니 한국어 못 함"
* "한국어 → 영어로 생성 강제 전환" expert forcing 데모

---

## 이 실험으로 알게 되는 것

### Layer 1: 실험이 직접 답하는 질문

**"Mixtral의 8 expert는 한국어와 영어를 어떻게 분담하는가?"**

세 결론 (A 분리 / B 공유 / C 부분 specialization) 중 하나로 답.

### Layer 2: 구체적으로 손에 잡히는 발견

1. **Expert별 언어 선호도 정량 수치** — "Expert 1은 한국어 5.6× 선호"
2. **Layer별 specialization 패턴** — "초기 layer는 language-specific, 후기는 shared"
3. **인과적 중요도** — Ablation effect size, "어떤 expert 끄면 어떤 언어 망가지는지"
4. **Cross-lingual transfer 메커니즘** — 영어 학습 후 한국어 성능에 어떤 expert가 기여

→ 이는 **Mixtral 모델에 대한 새로운 데이터**. 거의 없는 영역.

### Layer 3: 학계·실용 측면 함의

**학계 측면**:
* MoE expert specialization 연구에 multilingual 차원 추가.
* 한국어 + MoE 분석의 첫 체계적 시도.
* "Language module emerge in MoE training" 가설 검증.

**실용 측면**:
* **저자원 언어 추가**: 새 언어를 새 expert로 통합 가능?
* **Multilingual fine-tuning 전략**: Expert별 LoRA 적용 가능
* **모델 압축**: 특정 언어만 필요하면 일부 expert만 사용 (메모리 절감)
* **Bias 진단**: 언어별 expert 자원 할당 불평등 진단

### Layer 4: 한계 (정직하게)

본 실험이 **답할 수 없는 것**:
* "다른 MoE 모델(GPT-4, Gemini)도 같은가" — 비공개 모델이라 검증 불가
* "Specialization이 학습 과정 중 언제 발생했나" — 학습 trace 필요
* "다른 언어쌍 (한국어-일본어, 영어-중국어)도 같은 패턴인가" — 확장 실험 필요

답할 수 있는 것: **"Mixtral-8×7B(또는 선택한 MoE) 수준에서 한국어 vs 영어 expert 분담"** 까지.

→ 작은 발견이지만 명확하고 측정 가능.

---

## 강의 개념 연결

| 강의 개념 | 본 프로젝트에서 다루는 방식 |
|----------|---------------------------|
| Transformer architecture | MoE는 transformer의 자연스러운 확장, 직접 다룸 |
| Attention mechanism | Router의 attention-like 동작 분석 |
| Pretrained model 활용 | Mixtral 사전학습 모델 그대로 사용 |
| Multilingual NLP | 한국어/영어 처리 메커니즘 비교 |
| Causal analysis | Expert ablation으로 인과 검증 |
| 모델 압축·효율 | MoE의 sparsity, 활성 파라미터 개념 |
| 연구 방법론 | 가설 → 측정 → 인과 검증 흐름 |

---

## 교수님 평가 기준 대응

| 예상 질문 | 답변 |
|----------|------|
| 왜 이 모델? | Mixtral은 가장 널리 사용되는 open MoE이고 8개 expert로 분석이 깔끔. 4-bit 양자화로 학생 컴퓨트로도 가능. 너무 큰 GPT-4 같은 모델은 closed라 분석 불가. |
| 왜 이 데이터셋? | KLUE는 한국어 표준 벤치마크, 영어 페어로 GLUE 활용 → 언어가 유일한 변수가 되도록 controlled experiment. 합성 데이터가 아닌 real-world 한국어 사용. |
| 왜 이 평가지표? | Activation 빈도, **Specialization score**, **KL divergence**, **Ablation effect size** — 정보 이론적 + 인과적 검증의 표준 지표. 단순 accuracy가 아닌 메커니즘 검증. |
| 결과 해석 논리? | 빈도 측정 → specialization score → KL로 layer별 분포 → ablation으로 인과 검증. **각 단계가 이전 단계의 가설을 검증**하는 구조. |
| 왜 한국어인가? | 한국어는 영어와 형태론·통사론적으로 매우 다른 언어 → 만약 expert specialization이 일어난다면 가장 명확히 드러날 비교 대상. 한국 팀이라서 할 수 있는 차별점. |

---

## 위험 요소 + 대응

| 위험 | 가능성 | 대응 |
|------|--------|------|
| Mixtral이 너무 무거움 (메모리 부족) | 중간 | 4-bit 양자화 (Mixtral-8x7B-Instruct-v0.1-GPTQ), 또는 OLMoE-1B-7B로 대체 |
| Routing 추출 코드 작성 어려움 | 낮음 | HuggingFace `output_router_logits=True` 옵션 + 기존 MoE 분석 코드 참조 |
| Specialization이 약하게 나옴 (Case B 결과) | 중간 | 그것 자체가 발견 ("MoE는 multilingual을 공유 처리"). Mixtral 논문도 specialization 약함 보고 → 우리 결과로 확장. |
| Ablation 효과가 미미 | 중간 | Ablation 강도 조절 (1개 expert vs N개 동시), forcing 실험으로 보완 |
| 한국어/영어 페어 페차링 부담 | 낮음 | FLORES-200, xP3 같은 기존 multilingual 데이터셋 활용 |
| 컴퓨트 부족 | 낮음 | RTX 4090 1대 + 4-bit 양자화 충분. 대안으로 작은 OLMoE 사용 |

**핵심 안전망**: Step 1~3 (routing 추출 + specialization 측정)만 완성해도 기본 점수 확보. Step 4 (ablation + 시각화)는 차별화 가산점.

---

## 참조

### 핵심 페이퍼

* [Jiang et al., "Mixtral of Experts" (Mistral, 2024, arXiv:2401.04088)](https://arxiv.org/abs/2401.04088) — 본 프로젝트의 주 모델
* [Joint MoE Scaling Laws (2025, arXiv:2502.05172)](https://arxiv.org/abs/2502.05172) — MoE scaling 이론
* [Towards a Comprehensive Scaling Law of MoE (2025, arXiv:2509.23678)](https://arxiv.org/abs/2509.23678)
* [OLMoE: Open Mixture-of-Experts Language Models (Allen AI, 2024)](https://arxiv.org/abs/2409.02060) — 대안 모델
* [DeepSeek-MoE: Fine-grained Expert Specialization (2024)](https://arxiv.org/abs/2401.06066) — Specialization 강화 기법

### 라이브러리 + 코드

* [HuggingFace Mixtral 모델 (4-bit)](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
* [TransformerLens (mechanistic 분석 도구)](https://github.com/TransformerLensOrg/TransformerLens)
* [bitsandbytes 4-bit 양자화](https://github.com/TimDettmers/bitsandbytes)
* [KLUE 벤치마크](https://klue-benchmark.com)
* [FLORES-200 평행 corpus](https://github.com/facebookresearch/flores)

### 분야 위상 (제안 정당화)

* [Mechanistic Interpretability — MIT Tech Review 2026 Top 10 Breakthrough Technologies](https://www.technologyreview.com/2026/01/12/1130003/mechanistic-interpretability-ai-research-models-2026-breakthrough-technologies/)
* [ICLR 2026 Trends: Agentic AI, Multimodal Models](https://encord.com/iclr-2026/) — MoE는 ICLR 2026 핵심 키워드
* [Apple — Parameters vs FLOPs Scaling Laws for MoE](https://machinelearning.apple.com/research/parameters-flops-scaling)
* [Transformer Circuits Thread (Anthropic)](https://transformer-circuits.pub/) — mechanistic 분석 방법론

### 시각화 영감

* [Anthropic — Scaling Monosemanticity (interactive features)](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
* [Anthropic — Golden Gate Claude (feature steering demo)](https://www.anthropic.com/news/golden-gate-claude) — Ablation/Forcing 데모 영감
* [Mixtral Expert Routing Visualization (community)](https://huggingface.co/blog/moe)

### 관련 한국어 자료

* [KLUE 벤치마크 논문](https://arxiv.org/abs/2105.09680)
* [한국어 LLM 평가 (HRM8K, KMMLU 등)](https://huggingface.co/datasets/HAERAE-HUB)
