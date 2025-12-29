"""
Quick demo script for Saggar Detection
간단한 데모 실행 스크립트
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw

from detector import SaggarDetector


def create_sample_image():
    """
    테스트용 샘플 이미지 생성 (실제 사진이 없을 경우)
    """
    # 800x600 샘플 이미지 생성
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(image)

    # 팔렛 그리기
    pallet_y = height - 50
    draw.rectangle([50, pallet_y, width-50, height-20], fill=(139, 90, 43))

    # Saggar 그리기 (3층, 각 층에 4개씩)
    saggar_width = 80
    saggar_height = 100
    spacing = 20

    colors = [(180, 100, 100), (100, 180, 100), (100, 100, 180)]

    for layer in range(3):
        for col in range(4):
            x = 100 + col * (saggar_width + spacing)
            y = pallet_y - (layer + 1) * (saggar_height + 10)

            # Saggar 본체
            draw.rectangle(
                [x, y, x + saggar_width, y + saggar_height],
                fill=colors[layer],
                outline=(0, 0, 0),
                width=2
            )

            # 뚜껑
            draw.ellipse(
                [x - 5, y - 10, x + saggar_width + 5, y + 10],
                fill=colors[layer],
                outline=(0, 0, 0),
                width=2
            )

    # 저장
    os.makedirs('data/test', exist_ok=True)
    sample_path = 'data/test/sample_saggar.jpg'
    image.save(sample_path)
    print(f"Sample image created: {sample_path}")

    return sample_path


def run_demo():
    """
    데모 실행
    """
    print("=" * 60)
    print("Saggar Detection Demo")
    print("=" * 60)

    # 샘플 이미지 생성
    print("\n1. Creating sample image...")
    sample_image = create_sample_image()

    # Detector 초기화
    print("\n2. Initializing Saggar Detector...")
    detector = SaggarDetector(
        confidence_threshold=0.3,  # 샘플 이미지이므로 낮은 임계값
        device='cpu',  # CPU 사용 (빠른 테스트)
        vertical_tolerance=30,
        horizontal_tolerance=40
    )

    # 감지 수행
    print("\n3. Running detection...")
    try:
        result = detector.detect_from_image(
            sample_image,
            output_dir='output/demo'
        )

        # 결과 출력
        print("\n4. Results:")
        detector.print_summary(result)

        if result.get('num_detected', 0) == 0:
            print("\n주의: 샘플 이미지에서 Saggar가 감지되지 않았습니다.")
            print("이는 모델이 실제 Saggar 데이터로 학습되지 않았기 때문입니다.")
            print("\n실제 사용을 위해서는:")
            print("1. 실제 Saggar 사진 데이터 수집")
            print("2. 데이터 어노테이션 (COCO 또는 VOC 형식)")
            print("3. train.py를 사용하여 모델 fine-tuning")
            print("4. 학습된 모델로 detector 초기화")

    except Exception as e:
        print(f"\nError during detection: {e}")
        print("\n이는 정상적인 동작입니다. 모델이 실제 데이터로 학습되지 않았습니다.")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print(f"Check the output directory: output/demo/")
    print("=" * 60)


def show_usage():
    """
    사용 방법 표시
    """
    print("""
Saggar Detection System

사용 방법:

1. 데모 실행:
   python demo.py

2. 실제 이미지 처리:
   python detector.py --image path/to/image.jpg --output ./output

3. Python 코드에서 사용:
   from detector import SaggarDetector

   detector = SaggarDetector()
   result = detector.detect_from_image('image.jpg', 'output/')
   detector.print_summary(result)

4. 모델 학습:
   python train.py

5. 예제 코드 실행:
   python example_usage.py

더 자세한 정보는 README.md를 참조하세요.
    """)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_usage()
    else:
        run_demo()
