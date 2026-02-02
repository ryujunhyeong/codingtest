# Saggar Detection ROS Package

ROS1 Noetic 패키지로 .bag 파일에서 측면 Saggar 이미지를 읽고 실시간으로 감지, 층수 분석, 시각화를 수행합니다.

## 주요 기능

### 1. Mask R-CNN 기반 감지
- 딥러닝 기반 정확한 Saggar 감지 및 세그멘테이션
- 중심점 자동 계산
- 내부 테두리(contour) 추출
- 층별 자동 분류
- 실시간 처리 및 시각화

### 2. Edge Detection 기반 전통적 방식
- Canny Edge Detection
- Contour 기반 객체 감지
- 형태학적 연산으로 노이즈 제거
- 적응적 임계값 지원
- 빠른 처리 속도

### 3. ROS 통합 기능
- .bag 파일 자동 재생
- 실시간 토픽 발행
- RViz 마커 시각화
- 두 방식 비교 모드

## 패키지 구조

```
saggar_detection_ros/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── msg/
│   ├── SaggarDetection.msg      # 단일 Saggar 감지 메시지
│   ├── SaggarArray.msg          # Saggar 배열 메시지
│   └── LayerInfo.msg            # 층 정보 메시지
├── nodes/
│   ├── bag_player_node.py       # Bag 파일 재생 노드
│   ├── maskrcnn_detector_node.py # Mask R-CNN 감지 노드
│   ├── edge_detector_node.py    # Edge Detection 노드
│   └── visualization_node.py    # RViz 시각화 노드
├── launch/
│   ├── maskrcnn_detection.launch # Mask R-CNN 실행
│   ├── edge_detection.launch    # Edge Detection 실행
│   └── compare_methods.launch   # 두 방식 비교
├── config/
│   └── detection_params.yaml    # 파라미터 설정
└── rviz/
    └── saggar_detection.rviz    # RViz 설정
```

## 설치 방법

### 1. 의존성 설치

```bash
# ROS Noetic 기본 패키지
sudo apt-get install ros-noetic-cv-bridge ros-noetic-image-transport

# Python 패키지
pip3 install torch torchvision opencv-python numpy scipy
```

### 2. 워크스페이스 설정

```bash
# Catkin 워크스페이스 생성 (이미 있으면 생략)
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# 패키지 복사
cp -r /path/to/saggar_detection_ros .
cp -r /path/to/saggar_detection .

# 빌드
cd ~/catkin_ws
catkin_make

# Source
source devel/setup.bash
```

### 3. 실행 권한 부여

```bash
chmod +x ~/catkin_ws/src/saggar_detection_ros/nodes/*.py
```

## 사용 방법

### 1. Mask R-CNN 방식으로 감지

```bash
roslaunch saggar_detection_ros maskrcnn_detection.launch \
    bag_file:=/path/to/your/file.bag \
    image_topic:=/camera/image_raw \
    confidence_threshold:=0.5
```

**파라미터:**
- `bag_file`: .bag 파일 경로
- `image_topic`: bag 파일 내 이미지 토픽 이름
- `model_path`: 학습된 모델 경로 (선택사항)
- `confidence_threshold`: 감지 신뢰도 임계값 (0.0~1.0)
- `device`: 'cuda' 또는 'cpu'

### 2. Edge Detection 방식으로 감지

```bash
roslaunch saggar_detection_ros edge_detection.launch \
    bag_file:=/path/to/your/file.bag \
    image_topic:=/camera/image_raw \
    canny_low:=50 \
    canny_high:=150
```

**파라미터:**
- `canny_low`: Canny edge 하위 임계값
- `canny_high`: Canny edge 상위 임계값
- `min_contour_area`: 최소 contour 면적
- `max_contour_area`: 최대 contour 면적

### 3. 두 방식 비교

```bash
roslaunch saggar_detection_ros compare_methods.launch \
    bag_file:=/path/to/your/file.bag \
    image_topic:=/camera/image_raw
```

이 모드는 Mask R-CNN과 Edge Detection을 동시에 실행하여 결과를 비교할 수 있습니다.

### 4. 실시간 카메라 사용

bag 파일 대신 실시간 카메라를 사용하려면:

```bash
# bag_player를 사용하지 않고 직접 토픽 구독
roslaunch saggar_detection_ros maskrcnn_detection.launch \
    use_bag_player:=false \
    input_topic:=/your/camera/topic
```

## ROS 토픽

### 발행 토픽

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/saggar/image_raw` | sensor_msgs/Image | Bag에서 읽은 원본 이미지 |
| `/saggar/detections/maskrcnn` | SaggarArray | Mask R-CNN 감지 결과 |
| `/saggar/detections/edge` | SaggarArray | Edge Detection 감지 결과 |
| `/saggar/visualization/maskrcnn` | sensor_msgs/Image | Mask R-CNN 시각화 이미지 |
| `/saggar/visualization/edge` | sensor_msgs/Image | Edge Detection 시각화 이미지 |
| `/saggar/markers` | visualization_msgs/MarkerArray | RViz 바운딩 박스 마커 |
| `/saggar/text_markers` | visualization_msgs/MarkerArray | RViz 텍스트 마커 |
| `/saggar/contour_markers` | visualization_msgs/MarkerArray | RViz Contour 마커 |

### 구독 토픽

감지 노드는 `/saggar/image_raw` 토픽을 구독합니다.

## 메시지 구조

### SaggarDetection.msg
```
Header header
float32 x                    # 바운딩 박스 x
float32 y                    # 바운딩 박스 y
float32 width                # 바운딩 박스 너비
float32 height               # 바운딩 박스 높이
geometry_msgs/Point center   # 중심점
float32 confidence           # 신뢰도
int32 layer_id               # 층 번호
int32 column_id              # 열 번호
geometry_msgs/Point[] contour_points  # 내부 테두리 점들
sensor_msgs/Image mask       # 세그멘테이션 마스크
```

### SaggarArray.msg
```
Header header
string method                # "maskrcnn" or "edge_detection"
SaggarDetection[] detections # 감지된 Saggar 배열
int32 total_count            # 총 개수
int32 num_layers             # 층수
float32 processing_time      # 처리 시간 (ms)
```

## 시각화

### RViz에서 확인
RViz가 자동으로 실행되며 다음을 표시합니다:
- 원본 이미지
- 감지 결과 이미지 (마스크/contour 포함)
- 3D 바운딩 박스 마커
- 중심점 마커
- 층/열 정보 텍스트
- Contour 라인

### rqt_image_view로 이미지 비교
```bash
rqt_image_view
```
여러 이미지 토픽을 동시에 보며 비교할 수 있습니다.

## 파라미터 조정

### Mask R-CNN 파라미터
`config/detection_params.yaml` 또는 launch 파일에서 조정:
```yaml
maskrcnn:
  confidence_threshold: 0.5    # 신뢰도 임계값
  vertical_tolerance: 20       # 층 구분 허용 오차 (픽셀)
  horizontal_tolerance: 30     # 열 구분 허용 오차 (픽셀)
```

### Edge Detection 파라미터
```yaml
edge_detection:
  canny_low: 50               # Canny 하위 임계값
  canny_high: 150             # Canny 상위 임계값
  min_contour_area: 500       # 최소 contour 면적
  max_contour_area: 50000     # 최대 contour 면적
  morph_kernel_size: 3        # 형태학적 연산 커널 크기
  use_adaptive_threshold: false  # 적응적 임계값 사용 여부
```

## 성능 비교

| 방식 | 정확도 | 속도 | GPU 필요 | 장점 | 단점 |
|------|--------|------|----------|------|------|
| **Mask R-CNN** | 높음 | 중간 | 권장 | 정확한 세그멘테이션, 신뢰도 점수 | 느림, GPU 필요 |
| **Edge Detection** | 중간 | 빠름 | 불필요 | 빠른 처리, 가벼움 | 노이즈에 민감, 복잡한 환경에서 부정확 |

## 문제 해결

### 1. "No module named 'saggar_detection'" 오류
```bash
# 상위 saggar_detection 패키지의 경로를 PYTHONPATH에 추가
export PYTHONPATH=$PYTHONPATH:/path/to/saggar_detection
```

### 2. CUDA 메모리 부족
```bash
# CPU 모드로 실행
roslaunch saggar_detection_ros maskrcnn_detection.launch device:=cpu
```

### 3. Bag 파일에서 토픽을 찾을 수 없음
```bash
# Bag 파일의 토픽 확인
rosbag info your_file.bag

# 올바른 토픽 이름을 image_topic 파라미터로 전달
roslaunch saggar_detection_ros maskrcnn_detection.launch \
    bag_file:=your_file.bag \
    image_topic:=/correct/topic/name
```

### 4. Edge Detection 감지 안됨
- `canny_low`, `canny_high` 값 조정
- `min_contour_area`, `max_contour_area` 범위 조정
- `use_adaptive_threshold:=true` 시도

## 예제 사용

### 기본 사용 (Mask R-CNN)
```bash
roslaunch saggar_detection_ros maskrcnn_detection.launch \
    bag_file:=~/data/saggar_test.bag
```

### Edge Detection으로 빠른 처리
```bash
roslaunch saggar_detection_ros edge_detection.launch \
    bag_file:=~/data/saggar_test.bag \
    canny_low:=30 \
    canny_high:=100
```

### 두 방식 동시 비교
```bash
roslaunch saggar_detection_ros compare_methods.launch \
    bag_file:=~/data/saggar_test.bag
```

### Python에서 결과 구독
```python
#!/usr/bin/env python3
import rospy
from saggar_detection_ros.msg import SaggarArray

def callback(msg):
    print(f"Method: {msg.method}")
    print(f"Total: {msg.total_count} Saggars")
    print(f"Layers: {msg.num_layers}")
    print(f"Processing time: {msg.processing_time:.1f}ms")

    for det in msg.detections:
        print(f"  Layer {det.layer_id}, Center: ({det.center.x:.1f}, {det.center.y:.1f})")

rospy.init_node('listener')
rospy.Subscriber('/saggar/detections/maskrcnn', SaggarArray, callback)
rospy.spin()
```

## 고급 기능

### 1. 감지 결과를 파일로 저장
```python
#!/usr/bin/env python3
import rospy
import json
from saggar_detection_ros.msg import SaggarArray

results = []

def callback(msg):
    result = {
        'timestamp': msg.header.stamp.to_sec(),
        'method': msg.method,
        'total_count': msg.total_count,
        'num_layers': msg.num_layers,
        'detections': [
            {
                'layer_id': det.layer_id,
                'center_x': det.center.x,
                'center_y': det.center.y,
                'confidence': det.confidence
            }
            for det in msg.detections
        ]
    }
    results.append(result)

rospy.init_node('saver')
rospy.Subscriber('/saggar/detections/maskrcnn', SaggarArray, callback)

rospy.spin()

# Save results
with open('detection_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### 2. 커스텀 파라미터로 실행
`my_params.yaml` 생성:
```yaml
maskrcnn:
  confidence_threshold: 0.6
  vertical_tolerance: 25
```

실행:
```bash
rosparam load my_params.yaml
roslaunch saggar_detection_ros maskrcnn_detection.launch
```

## 라이선스
MIT License

## 참고
- [ROS Noetic Documentation](http://wiki.ros.org/noetic)
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- Canny Edge Detection: J. Canny, "A Computational Approach to Edge Detection", IEEE Trans. Pattern Analysis and Machine Intelligence, 1986
