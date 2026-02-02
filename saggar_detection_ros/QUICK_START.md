# Saggar Detection ROS - Quick Start Guide

ROS1 Noetic을 사용한 실시간 Saggar 감지 시스템 빠른 시작 가이드

## 1분 요약

```bash
# 1. 워크스페이스 생성
mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src

# 2. 패키지 복사
cp -r /path/to/saggar_detection .
cp -r /path/to/saggar_detection_ros .

# 3. 빌드
cd ~/catkin_ws && catkin_make && source devel/setup.bash

# 4. 실행 (Mask R-CNN)
roslaunch saggar_detection_ros maskrcnn_detection.launch \
    bag_file:=/path/to/your/file.bag \
    image_topic:=/camera/image_raw
```

## 설치

### 필수 요구사항
- Ubuntu 20.04
- ROS Noetic
- Python 3.8+
- CUDA 11.x (GPU 사용 시)

### 1단계: ROS 의존성 설치

```bash
sudo apt-get update
sudo apt-get install -y \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-vision-msgs \
    ros-noetic-rviz \
    ros-noetic-rqt-image-view
```

### 2단계: Python 패키지 설치

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install opencv-python numpy scipy pillow matplotlib
```

### 3단계: 워크스페이스 설정

```bash
# Catkin 워크스페이스 생성
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# 패키지 복사 (실제 경로로 변경)
cp -r /path/to/saggar_detection .
cp -r /path/to/saggar_detection_ros .

# PYTHONPATH 설정 (선택사항이지만 권장)
echo "export PYTHONPATH=\$PYTHONPATH:~/catkin_ws/src/saggar_detection" >> ~/.bashrc
source ~/.bashrc

# 빌드
cd ~/catkin_ws
catkin_make

# Source
source devel/setup.bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

## 사용 방법

### 방법 1: Mask R-CNN (딥러닝 기반)

**특징:**
- ✅ 높은 정확도
- ✅ 정확한 세그멘테이션
- ✅ 신뢰도 점수 제공
- ⚠️ GPU 권장 (느림)

**실행:**
```bash
roslaunch saggar_detection_ros maskrcnn_detection.launch \
    bag_file:=/path/to/your/file.bag \
    image_topic:=/camera/image_raw \
    confidence_threshold:=0.5 \
    device:=cuda
```

### 방법 2: Edge Detection (전통적 방식)

**특징:**
- ✅ 빠른 처리 속도
- ✅ GPU 불필요
- ✅ 가벼움
- ⚠️ 노이즈에 민감

**실행:**
```bash
roslaunch saggar_detection_ros edge_detection.launch \
    bag_file:=/path/to/your/file.bag \
    image_topic:=/camera/image_raw \
    canny_low:=50 \
    canny_high:=150
```

### 방법 3: 두 방식 비교

```bash
roslaunch saggar_detection_ros compare_methods.launch \
    bag_file:=/path/to/your/file.bag \
    image_topic:=/camera/image_raw
```

## Bag 파일 준비

### Bag 파일 확인

```bash
# Bag 파일 정보 확인
rosbag info your_file.bag

# 출력 예시:
# topics:      /camera/image_raw    1000 msgs    : sensor_msgs/Image
#              /camera/camera_info  1000 msgs    : sensor_msgs/CameraInfo
```

### 이미지 토픽이 압축된 경우

```bash
# compressed 토픽을 raw로 변환하는 노드 실행
rosrun image_transport republish compressed in:=/camera/image_raw/compressed raw out:=/camera/image_raw
```

## 결과 확인

### 1. RViz에서 시각화
Launch 파일 실행 시 RViz가 자동으로 열립니다.

**표시 내용:**
- 원본 이미지
- 감지 결과 (마스크/contour 표시)
- 3D 바운딩 박스 마커
- 중심점 마커
- 층/열 정보 텍스트

### 2. 터미널 로그
```
[ INFO] Detected 36 Saggars in 3 layers (125.3ms)
[ INFO] Layer 1: 12 Saggars in 4 columns
[ INFO] Layer 2: 12 Saggars in 4 columns
[ INFO] Layer 3: 12 Saggars in 4 columns
```

### 3. rqt_image_view로 이미지 비교
```bash
rqt_image_view
```

여러 토픽 선택:
- `/saggar/image_raw` - 원본
- `/saggar/visualization/maskrcnn` - Mask R-CNN 결과
- `/saggar/visualization/edge` - Edge Detection 결과

### 4. 토픽 확인
```bash
# 발행되는 토픽 확인
rostopic list

# 감지 결과 실시간 확인
rostopic echo /saggar/detections/maskrcnn

# 메시지 빈도 확인
rostopic hz /saggar/detections/maskrcnn
```

## 파라미터 조정

### Mask R-CNN 감지 향상

```bash
roslaunch saggar_detection_ros maskrcnn_detection.launch \
    bag_file:=/path/to/file.bag \
    confidence_threshold:=0.3 \    # 더 낮은 임계값 (더 많이 감지)
    device:=cuda
```

### Edge Detection 감도 조정

```bash
roslaunch saggar_detection_ros edge_detection.launch \
    bag_file:=/path/to/file.bag \
    canny_low:=30 \               # 더 민감하게
    canny_high:=100 \
    min_contour_area:=300 \       # 더 작은 객체도 감지
    max_contour_area:=80000
```

## 문제 해결

### ❌ "No module named 'saggar_detection'"

**해결:**
```bash
export PYTHONPATH=$PYTHONPATH:~/catkin_ws/src/saggar_detection
source ~/.bashrc
```

### ❌ CUDA 메모리 부족

**해결 1:** CPU 모드 사용
```bash
roslaunch saggar_detection_ros maskrcnn_detection.launch device:=cpu
```

**해결 2:** Edge Detection 사용
```bash
roslaunch saggar_detection_ros edge_detection.launch
```

### ❌ "Topic /camera/image_raw not found in bag"

**해결:**
```bash
# Bag 파일의 실제 토픽 확인
rosbag info your_file.bag

# 올바른 토픽 이름 사용
roslaunch saggar_detection_ros maskrcnn_detection.launch \
    image_topic:=/your/actual/topic
```

### ❌ Edge Detection이 아무것도 감지 못함

**해결:** 파라미터 조정
```bash
roslaunch saggar_detection_ros edge_detection.launch \
    canny_low:=20 \
    canny_high:=80 \
    min_contour_area:=200
```

또는 adaptive threshold 사용:
```bash
# edge_detector_node.py의 파라미터 변경
rosparam set /edge_detector/use_adaptive_threshold true
```

## 고급 사용

### 실시간 카메라 사용

```bash
# USB 카메라
rosrun usb_cam usb_cam_node

# 별도 터미널에서 감지 실행
roslaunch saggar_detection_ros maskrcnn_detection.launch \
    use_bag_player:=false \
    input_topic:=/usb_cam/image_raw
```

### Python 스크립트로 결과 받기

```python
#!/usr/bin/env python3
import rospy
from saggar_detection_ros.msg import SaggarArray

def callback(msg):
    print(f"감지됨: {msg.total_count}개, {msg.num_layers}층")
    for det in msg.detections:
        print(f"  중심: ({det.center.x:.0f}, {det.center.y:.0f}), "
              f"층: {det.layer_id}, 신뢰도: {det.confidence:.2f}")

rospy.init_node('result_listener')
rospy.Subscriber('/saggar/detections/maskrcnn', SaggarArray, callback)
rospy.spin()
```

### 결과를 JSON으로 저장

```bash
# 별도 터미널에서
rostopic echo -p /saggar/detections/maskrcnn > results.csv
```

## 성능 벤치마크

| 방식 | 처리 시간 (GPU) | 처리 시간 (CPU) | 메모리 사용 |
|------|----------------|----------------|------------|
| Mask R-CNN | 100-200ms | 800-1500ms | ~2GB |
| Edge Detection | 10-30ms | 10-30ms | ~100MB |

**권장 사용:**
- **정확도 중요** → Mask R-CNN + GPU
- **속도 중요** → Edge Detection
- **GPU 없음** → Edge Detection

## 추가 리소스

- 상세 문서: `saggar_detection_ros/README.md`
- Mask R-CNN 알고리즘: `saggar_detection/README.md`
- ROS Wiki: http://wiki.ros.org/noetic
- 문제 보고: GitHub Issues

## 다음 단계

1. ✅ 기본 감지 실행 완료
2. 📊 결과 분석 및 파라미터 최적화
3. 🎯 자체 데이터로 Mask R-CNN 모델 fine-tuning
4. 🔧 특정 환경에 맞춘 파라미터 튜닝
5. 📈 성능 측정 및 개선

## 요약

```bash
# 🚀 가장 빠른 시작 (Edge Detection)
roslaunch saggar_detection_ros edge_detection.launch \
    bag_file:=/path/to/file.bag

# 🎯 가장 정확한 감지 (Mask R-CNN + GPU)
roslaunch saggar_detection_ros maskrcnn_detection.launch \
    bag_file:=/path/to/file.bag \
    device:=cuda

# 🔬 두 방식 비교
roslaunch saggar_detection_ros compare_methods.launch \
    bag_file:=/path/to/file.bag
```

성공적인 감지를 기원합니다! 🎉
