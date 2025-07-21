# Layer-wise Evolutionary Optimization for Cross-Domain Model Merging in CJK and Beyond

## 📖 개요 (Overview)
다국어(중국어, 일본어, 한국어) 언어 모델의 병합 및 성능 평가를 위한 프로젝트입니다.
CMA-ES 최적화를 통해 모델 병합 가중치를 자동으로 찾고, 각 언어별 MGSM 벤치마크에서 성능을 평가합니다.

## 🎯 주요 기능 (Features)
- **모델 병합**: 선형 SLERP 방법을 사용한 언어별 모델 병합
- **자동 최적화**: CMA-ES를 통한 병합 가중치 자동 탐색
- **다국어 평가**: 중국어, 일본어, 한국어 벤치마크 평가
- **성능 추적**: 실시간 성능 모니터링 및 로깅

## 🛠️ 설치 (Installation)

### 요구사항 (Requirements)
- Python 3.8+
- CUDA-enabled GPU
- PyTorch 2.1.2+

### 의존성 설치
```bash
pip install -r requirements.txt
```

## 🚀 사용법 (Usage)

### 1. 모델 평가만 실행
각 언어별로 미리 병합된 모델을 평가:

```bash
# 한국어 모델 평가
python 한국어_evaluate.py

# 일본어 모델 평가
python 일본어_evaluate.py

# 중국어 모델 평가
python 중국어_evaluate.py
```

### 2. CMA-ES 최적화를 통한 병합 및 평가
최적 병합 가중치를 자동으로 찾아 모델을 병합하고 평가:

```bash
# 한국어 최적화
python 한국어_cma_back.py

# 일본어 최적화
python 일본어_cma_back.py

# 중국어 최적화
python 중국어_cma_back.py
```

## 🏗️ 프로젝트 구조 (Project Structure)

```
├── 한국어_evaluate.py      # 한국어 모델 평가
├── 일본어_evaluate.py      # 일본어 모델 평가  
├── 중국어_evaluate.py      # 중국어 모델 평가
├── 한국어_cma_back.py      # 한국어 CMA-ES 최적화
├── 일본어_cma_back.py      # 일본어 CMA-ES 최적화
├── 중국어_cma_back.py      # 중국어 CMA-ES 최적화
├── 한국어_cma.py           # 한국어 모델 병합 및 평가
├── 일본어_cma.py           # 일본어 모델 병합 및 평가
├── 중국어_cma.py           # 중국어 모델 병합 및 평가
├── configs/              # YAML 설정 파일들
├── results/              # 결과 저장 폴더
├── merged_model/         # 병합된 모델 저장 폴더
└── requirements.txt      # 의존성 목록
```

## 🔧 설정 (Configuration)

### GPU 설정
기본적으로 GPU 4번을 사용하도록 설정되어 있습니다:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
```

다른 GPU를 사용하려면 코드에서 해당 부분을 수정하세요.

### 모델 설정
각 언어별로 사용하는 베이스 모델:
- **한국어**: `davidkim205/komt-mistral-7b-v1` + `WizardLM/WizardMath-7B-V1.1`
- **일본어**: `augmxnt/shisa-gamma-7b-v1` + `WizardLM/WizardMath-7B-V1.1`
- **중국어**: `lchakkei/Mistral-7B-V2-Traditional-Chinese` + `WizardLM/WizardMath-7B-V1.1`

## 📈 결과 (Results)

결과는 `results/` 폴더에 언어별로 저장됩니다:
- `results/한국어/`: 한국어 평가 결과
- `results/일본어/`: 일본어 평가 결과  
- `results/중국어/`: 중국어 평가 결과


## 🔍 상세 정보 (Detailed Information)

### CMA-ES 최적화
- **초기값**: 모든 레이어에 대해 0.5 (한국어는 0.33)
- **표준편차**: 0.1
- **반복횟수**: 100회
- **레이어 수**: 291개

### 병합 방법
- **SLERP**: Spherical Linear Interpolation
- **가중치**: 레이어별로 다른 가중치 적용
- **저장**: 최고 성능 달성 시 자동 저장