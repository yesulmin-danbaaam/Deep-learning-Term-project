# 📄 Term Project 후보 주제 통합 보고서

---

# 🥇 1. CommCoach AI

## (상황 인식 기반 커뮤니케이션 코칭 시스템)

### 📌 주제

공격적이거나 비효율적인 문장을 상황에 맞는 공손하고 효과적인 표현으로 변환하고, 적절한 커뮤니케이션 전략까지 추천하는 AI 시스템

---

### 🧠 대략적인 계획

* Step 1: 문장 공격성/감정 분류 모델 학습
* Step 2: Transformer 기반 문장 재작성 모델 구축
* Step 3: 상황 기반 전략 분류 모델 추가
* Step 4: 채팅 이미지 입력 시 CNN + OCR 적용 (확장)
* Step 5: 전체 파이프라인 통합 및 데모 제작

---

### 📊 실제 활용 데이터셋

* Korean Toxic Comment Dataset (KOTOX)
* Korean Hate Speech Dataset
* Jigsaw Toxic Comment Dataset
* 자체 제작 데이터 (공손 변환 문장 쌍)

---

### 📈 예상 결과

* 공격성 분류 정확도: 85% 이상
* 문장 변환 자연스러움 향상 (정성 평가)
* 실제 커뮤니케이션 개선 효과 데모 가능

---

---

# 🥈 2. SLLM Tool Calling Agent

## (자연어 → Tool/API 호출 모델)

### 📌 주제

사용자의 자연어 요청을 적절한 API 또는 Tool 호출(JSON 형태)로 변환하는 경량 SLLM 모델

---

### 🧠 대략적인 계획

* Step 1: Tool schema 정의 (6~8개)
* Step 2: tool-calling 데이터셋 정리
* Step 3: SLLM (Qwen, Llama 등) QLoRA fine-tuning
* Step 4: JSON parsing + tool executor 구현
* Step 5: UI 기반 실제 실행 데모

---

### 📊 실제 활용 데이터셋

* xLAM Function Calling Dataset
* ToolBench
* Berkeley Function Calling Leaderboard Dataset
* delivery-app-function-calling-datasets-korean

---

### 📈 예상 결과

* Tool selection accuracy: 80% 이상
* JSON valid output 비율 증가
* 실제 API 호출 성공률 개선

---

---

# 🥉 3. AI 카톡 답장 클론

## (개인화 대화 생성 모델)

### 📌 주제

사용자의 말투를 학습하여 자동으로 개인 스타일의 답장을 생성하는 AI

---

### 🧠 대략적인 계획

* Step 1: 대화 데이터 전처리 (입력-답장 구조)
* Step 2: Transformer 기반 생성 모델 학습
* Step 3: 개인 데이터로 fine-tuning
* Step 4: 실시간 채팅 데모 제작

---

### 📊 실제 활용 데이터셋

* AI Hub 한국어 대화 데이터셋
* DailyDialog Dataset
* 개인 카톡 데이터 (익명화)

---

### 📈 예상 결과

* 자연스러운 답장 생성
* 개인 말투 유사도 향상
* 높은 데모 임팩트

---

---

# 🎧 4. 감정 기반 음악 추천 AI

### 📌 주제

사용자의 얼굴 표정 또는 음성을 분석하여 감정 상태에 맞는 음악을 추천하는 시스템

---

### 🧠 대략적인 계획

* Step 1: 얼굴 감정 인식 CNN 학습
* Step 2: 감정 → 음악 매핑 로직 설계
* Step 3: 실시간 카메라 입력 처리
* Step 4: 추천 결과 UI 구현

---

### 📊 실제 활용 데이터셋

* FER2013
* RAVDESS

---

### 📈 예상 결과

* 감정 분류 정확도 70~80%
* 실시간 추천 데모 가능
* 사용자 체감 만족도 높음

---

---

# 🧍 5. 집중력 예측 AI

### 📌 주제

사용자의 얼굴 및 행동 패턴을 기반으로 집중력 저하 상태를 예측하는 모델

---

### 🧠 대략적인 계획

* Step 1: 영상 데이터 수집
* Step 2: CNN + RNN 모델 학습
* Step 3: 집중도 scoring 시스템 설계
* Step 4: 실시간 예측 데모

---

### 📊 실제 활용 데이터셋

* DAiSEE Dataset

---

### 📈 예상 결과

* 집중 상태 분류 가능
* 실시간 피드백 시스템 구현
* 교육/업무 활용 가능성

---

---

# 🧠 6. 기억 왜곡 탐지 AI

### 📌 주제

동일 질문에 대한 답변의 일관성을 분석하여 기억 왜곡 가능성을 탐지

---

### 🧠 대략적인 계획

* Step 1: 인터뷰 데이터 직접 수집
* Step 2: 텍스트 유사도 분석
* Step 3: 시간 흐름 기반 RNN 모델 적용
* Step 4: consistency score 계산

---

### 📊 실제 활용 데이터셋

* Persona-Chat Dataset
* 자체 인터뷰 데이터

---

### 📈 예상 결과

* 답변 일관성 수치화
* 새로운 유형의 분석 결과 제시
* 연구형 프로젝트로 적합

---

---

# 🧠 7. AI 환각 탐지기

### 📌 주제

AI가 생성한 “그럴듯한 거짓 정보”를 탐지하는 시스템

---

### 🧠 대략적인 계획

* Step 1: 사실/허구 데이터 구축
* Step 2: Transformer 기반 분류 모델 학습
* Step 3: hallucination score 정의
* Step 4: 모델 비교 실험

---

### 📊 실제 활용 데이터셋

* FEVER Dataset
* TruthfulQA

---

### 📈 예상 결과

* hallucination detection 정확도 향상
* 최신 AI 문제 해결 접근
* 연구 가치 높음

---

---

# 🎬 8. 밈 생성 AI

### 📌 주제

사용자가 입력한 상황을 기반으로 적절한 밈 이미지를 생성하는 시스템

---

### 🧠 대략적인 계획

* Step 1: 밈 템플릿 수집
* Step 2: 상황 문장 생성
* Step 3: 텍스트 → 이미지 매핑
* Step 4: UI 기반 출력

---

### 📊 실제 활용 데이터셋

* Meme Dataset (Facebook AI)
* 직접 수집 데이터

---

### 📈 예상 결과

* 재미 중심 결과
* 높은 발표 임팩트
* 사용자 참여 유도

---

---

# 🎬 9. 이상 행동 탐지 AI

### 📌 주제

영상에서 비정상적인 행동을 자동으로 탐지하는 시스템

---

### 🧠 대략적인 계획

* Step 1: 영상 데이터 수집
* Step 2: CNN + LSTM 모델 학습
* Step 3: anomaly detection 모델 적용
* Step 4: 영상 데모 제작

---

### 📊 실제 활용 데이터셋

* UCF Crime Dataset

---

### 📈 예상 결과

* 이상 행동 탐지 가능
* 보안/감시 시스템 응용

---

---

# 📊 최종 정리 및 추천

## 🎯 종합 난이도 / 점수 / 현실성 기준

| 주제           | 난이도 | 구현 가능성 | 점수    |
| ------------ | --- | ------ | ----- |
| CommCoach    | 중상  | 높음     | ⭐⭐⭐⭐⭐ |
| Tool Calling | 중상  | 높음     | ⭐⭐⭐⭐⭐ |
| 카톡 클론        | 중   | 높음     | ⭐⭐⭐⭐  |
| 감정 음악        | 중   | 매우 높음  | ⭐⭐⭐⭐  |
| 집중력          | 중   | 중      | ⭐⭐⭐⭐  |
| 기억 왜곡        | 상   | 낮음     | ⭐⭐⭐⭐⭐ |
| 환각 탐지        | 상   | 낮음     | ⭐⭐⭐⭐  |

---

# 🔥 최종 추천

## 🥇 1순위

👉 **CommCoach AI**

* 가장 안정적으로 A+
* 데이터 확보 가능
* 강의 내용 완벽 반영

---

## 🥈 2순위

👉 **SLLM Tool Calling Agent**

* 최신 트렌드
* 기술적으로 매우 강함

---

## 🥉 3순위

👉 **AI 카톡 클론**

* 재미 + 데모 임팩트 최고

---

# 🧠 최종 한 줄

👉
**“CommCoach 또는 Tool Calling이 가장 현실적으로 고득점 가능한 선택”**


