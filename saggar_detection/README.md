# Saggar Detection using Mask R-CNN

측면 사진에서 팔렛 위에 쌓여있는 Saggar의 배열 및 층수를 자동으로 감지하고 분석하는 딥러닝 시스템입니다.

## 주요 기능

- **Saggar 감지**: Mask R-CNN을 활용한 정확한 Saggar 객체 감지 및 세그멘테이션
- **층수 계산**: 측면 사진에서 Y축 좌표 기반 자동 층 분류
- **배열 분석**: 각 층의 Saggar 배열 패턴 자동 분석
- **패턴 인식**: 균일 배열, 피라미드형, 교차 배열 등 쌓임 패턴 분류
- **안정성 평가**: 쌓임 구조의 안정성 자동 평가
- **시각화**: 감지 결과 및 층별 구분 시각화

## 시스템 구조

```
saggar_detection/
├── models/
│   ├── __init__.py
│   └── maskrcnn_model.py      # Mask R-CNN 모델 정의
├── utils/
│   ├── __init__.py
│   ├── layer_counter.py       # 층수 계산 알고리즘
│   ├── pattern_analyzer.py    # 배열 패턴 분석
│   └── image_utils.py         # 이미지 처리 유틸리티
├── configs/
│   └── config.yaml            # 설정 파일
├── data/                      # 데이터 디렉토리
├── detector.py                # 메인 감지 파이프라인
├── train.py                   # 모델 학습 스크립트
├── example_usage.py           # 사용 예제
└── requirements.txt           # 패키지 의존성
```

## 설치 방법

### 1. 필요 패키지 설치

```bash
cd saggar_detection
pip install -r requirements.txt
```

### 2. PyTorch 설치 (CUDA 사용 시)

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision
```

## 사용 방법

### 기본 사용법

```python
from detector import SaggarDetector

# Detector 초기화
detector = SaggarDetector(
    confidence_threshold=0.5,
    device='cuda',  # 'cuda' 또는 'cpu'
    vertical_tolerance=20,
    horizontal_tolerance=30
)

# 이미지 처리
result = detector.detect_from_image(
    image_path='path/to/image.jpg',
    output_dir='output/'
)

# 결과 출력
detector.print_summary(result)
```

### 명령줄 사용

```bash
python detector.py \
    --image path/to/image.jpg \
    --output ./output \
    --confidence 0.5 \
    --device cuda
```

### 일괄 처리

```python
# 여러 이미지 동시 처리
results = detector.batch_process(
    image_dir='data/test/',
    output_dir='output/batch/',
    pattern='*.jpg'
)
```

## 알고리즘 설명

### 1. Saggar 감지 (Mask R-CNN)

- **백본**: ResNet-50 with FPN (Feature Pyramid Network)
- **출력**:
  - 바운딩 박스 (x1, y1, x2, y2)
  - 신뢰도 점수
  - 세그멘테이션 마스크

```python
# 모델 예측
predictions = model.predict(image, confidence_threshold=0.5)

# 결과
boxes = predictions['boxes']      # [N, 4]
scores = predictions['scores']    # [N]
masks = predictions['masks']      # [N, H, W]
```

### 2. 층수 계산

측면 사진에서 Y축 좌표를 기반으로 층을 구분합니다.

**알고리즘**:
1. 각 Saggar의 중심점 계산: `center_y = (y1 + y2) / 2`
2. Y 좌표 기반 계층적 클러스터링 수행
3. 허용 오차 내의 Saggar들을 동일 층으로 그룹화
4. 위에서 아래로 층 번호 할당 (1, 2, 3, ...)

```python
from utils import SaggarLayerCounter

counter = SaggarLayerCounter(
    vertical_tolerance=20,    # 같은 층 판단 기준 (픽셀)
    horizontal_tolerance=30   # 같은 열 판단 기준 (픽셀)
)

layer_analysis = counter.analyze_stacking_pattern(boxes, scores)
```

**출력 예시**:
```python
{
    'total_saggars': 36,
    'num_layers': 3,
    'layer_info': {
        1: {'saggar_count': 12, 'num_columns': 4},
        2: {'saggar_count': 12, 'num_columns': 4},
        3: {'saggar_count': 12, 'num_columns': 4}
    }
}
```

### 3. 배열 패턴 분석

각 층의 Saggar 배열 패턴을 분석합니다.

**패턴 유형**:
- **균일 배열 (Uniform)**: 모든 층의 Saggar 개수 동일
- **피라미드형 (Pyramid)**: 위로 갈수록 개수 감소
- **교차 배열 (Alternating)**: 홀수/짝수 층이 교차 패턴
- **불규칙 (Irregular)**: 위 패턴에 해당하지 않음

```python
from utils import SaggarPatternAnalyzer

analyzer = SaggarPatternAnalyzer()

# 패턴 분석
pattern_type = analyzer.detect_pattern_type(layer_info)
grid_info = analyzer.calculate_grid_arrangement(layer_info)
stability = analyzer.analyze_stability(layer_info)
```

### 4. 안정성 평가

쌓임 구조의 안정성을 평가합니다.

**평가 기준**:
- 위층이 아래층보다 많으면 불안정
- 층간 개수 편차가 크면 경고
- 안정성 점수: 0.0 (불안정) ~ 1.0 (안정)

## 결과 예시

```
=============================================================
SAGGAR DETECTION SUMMARY
=============================================================

Image: test_image.jpg
Total Saggars Detected: 36

Number of Layers: 3

  Layer 1:
    - Saggars: 12
    - Columns: 4
    - Avg Confidence: 0.932

  Layer 2:
    - Saggars: 12
    - Columns: 4
    - Avg Confidence: 0.918

  Layer 3:
    - Saggars: 12
    - Columns: 4
    - Avg Confidence: 0.895

Pattern Type: 균일 배열

Stability Score: 1.00

=============================================================
```

## 고급 기능

### 1. 기대 구성과 비교

```python
expected_config = {
    'num_layers': 5,
    'saggars_per_layer': 12,
    'total_saggars': 60
}

comparison = analyzer.compare_with_expected(
    layer_info,
    expected_config
)

if not comparison['matches_expected']:
    for diff in comparison['differences']:
        print(f"차이: {diff['category']} - "
              f"예상 {diff['expected']}, 실제 {diff['actual']}")
```

### 2. 누락 감지

```python
gaps_info = analyzer.detect_gaps_or_missing(layer_info)

if gaps_info['has_gaps']:
    print(f"누락된 층: {gaps_info['missing_layers']}")
    print(f"불완전한 층: {len(gaps_info['incomplete_layers'])}")
```

### 3. 격자 배열 분석

```python
grid_info = analyzer.calculate_grid_arrangement(layer_info)

for layer, info in grid_info.items():
    print(f"Layer {layer}: {info['columns']}x{info['max_rows']} grid")
    print(f"  완전한 격자: {info['is_complete_grid']}")
```

## 모델 학습

자체 데이터로 모델을 fine-tuning할 수 있습니다.

### 1. 데이터 준비

COCO 또는 VOC 형식으로 어노테이션된 데이터 필요:

```
data/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    └── images/
```

### 2. 학습 실행

```python
from train import SaggarTrainer
from models import SaggarMaskRCNN

# 모델 초기화
model = SaggarMaskRCNN(num_classes=2)

# 학습
trainer = SaggarTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=0.005,
    num_epochs=50
)

trainer.train()
```

## 설정 커스터마이징

`configs/config.yaml` 파일에서 설정 변경 가능:

```yaml
detection:
  confidence_threshold: 0.5

layer_analysis:
  vertical_tolerance: 20
  horizontal_tolerance: 30

training:
  batch_size: 2
  learning_rate: 0.005
  num_epochs: 50
```

## 성능 최적화

### GPU 사용

```python
detector = SaggarDetector(device='cuda')
```

### 이미지 크기 조정

큰 이미지는 자동으로 리사이즈됩니다 (기본 1024px):

```python
from utils.image_utils import resize_image

resized = resize_image(image, max_size=1024)
```

### 배치 처리

여러 이미지를 효율적으로 처리:

```python
results = detector.batch_process(
    image_dir='data/',
    pattern='*.jpg'
)
```

## 문제 해결

### 1. CUDA 메모리 부족

- 이미지 크기 줄이기
- 배치 크기 감소

### 2. 감지 정확도 낮음

- `confidence_threshold` 조정
- 자체 데이터로 fine-tuning
- 이미지 전처리 개선

### 3. 층 구분 오류

- `vertical_tolerance` 조정
- `horizontal_tolerance` 조정
- 이미지 촬영 각도 개선

## 예제 코드

더 많은 예제는 `example_usage.py` 참조:

```bash
python example_usage.py
```

## 시스템 요구사항

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU 사용 시)
- RAM: 8GB 이상 권장
- GPU: 4GB VRAM 이상 (학습 시)

## 라이선스

MIT License

## 문의

이슈 또는 개선 제안이 있으시면 GitHub Issues를 이용해주세요.
