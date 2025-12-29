"""
Main Saggar Detection Pipeline
측면 사진에서 Saggar를 감지하고 층수 및 배열을 분석하는 통합 파이프라인
"""

import os
import torch
import numpy as np
from pathlib import Path

from models import SaggarMaskRCNN
from utils import SaggarLayerCounter, SaggarPatternAnalyzer
from utils.image_utils import (
    load_image,
    prepare_image_for_model,
    draw_detection_results,
    save_image,
    resize_image
)


class SaggarDetector:
    """
    Saggar 감지 및 분석을 위한 통합 클래스
    """

    def __init__(
        self,
        model_path=None,
        confidence_threshold=0.5,
        device=None,
        vertical_tolerance=20,
        horizontal_tolerance=30
    ):
        """
        Args:
            model_path (str, optional): 학습된 모델 경로
            confidence_threshold (float): 감지 신뢰도 임계값
            device (str, optional): 사용할 디바이스
            vertical_tolerance (int): 같은 층 판단 허용 오차
            horizontal_tolerance (int): 같은 열 판단 허용 오차
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Mask R-CNN 모델 초기화
        self.model = SaggarMaskRCNN(num_classes=2, device=self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print("Using pretrained Mask R-CNN (not fine-tuned for Saggar)")

        self.model.eval_mode()

        # 층수 계산기 초기화
        self.layer_counter = SaggarLayerCounter(
            vertical_tolerance=vertical_tolerance,
            horizontal_tolerance=horizontal_tolerance
        )

        # 패턴 분석기 초기화
        self.pattern_analyzer = SaggarPatternAnalyzer()

    def detect_from_image(self, image_path, output_dir=None):
        """
        이미지에서 Saggar 감지 및 분석

        Args:
            image_path (str): 입력 이미지 경로
            output_dir (str, optional): 결과 저장 디렉토리

        Returns:
            dict: 감지 및 분석 결과
        """
        print(f"Processing image: {image_path}")

        # 이미지 로드
        pil_image, np_image = load_image(image_path)
        original_size = np_image.shape[:2]

        # 이미지 크기 조정 (처리 속도 향상)
        resized_image = resize_image(np_image, max_size=1024)

        # 모델 입력 준비
        image_tensor = prepare_image_for_model(resized_image, device=self.device)

        # Saggar 감지
        print("Detecting Saggars...")
        predictions = self.model.predict(
            image_tensor,
            confidence_threshold=self.confidence_threshold
        )

        num_detected = len(predictions['boxes'])
        print(f"Detected {num_detected} Saggars")

        if num_detected == 0:
            return {
                'image_path': image_path,
                'num_detected': 0,
                'message': 'No Saggars detected'
            }

        # 층수 및 배열 분석
        print("Analyzing layers and arrangement...")
        layer_analysis = self.layer_counter.analyze_stacking_pattern(
            predictions['boxes'],
            predictions['scores']
        )

        # 패턴 분석
        print("Analyzing pattern...")
        pattern_report = self.pattern_analyzer.generate_comprehensive_report(
            layer_analysis['layer_info']
        )

        # 결과 시각화
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # 감지 결과 시각화
            detection_vis = draw_detection_results(
                resized_image,
                predictions['boxes'],
                predictions['masks'],
                predictions['labels'],
                predictions['scores'],
                show_masks=True
            )

            detection_output = os.path.join(
                output_dir,
                f"{Path(image_path).stem}_detection.jpg"
            )
            save_image(detection_vis, detection_output)

            # 층별 시각화
            layer_vis = self.layer_counter.visualize_layers(
                resized_image,
                predictions['boxes'].numpy(),
                np.array(layer_analysis['layer_labels'])
            )

            layer_output = os.path.join(
                output_dir,
                f"{Path(image_path).stem}_layers.jpg"
            )
            save_image(layer_vis, layer_output)

            print(f"Results saved to {output_dir}")

        # 최종 결과 구성
        result = {
            'image_path': image_path,
            'image_size': original_size,
            'num_detected': num_detected,
            'detection_info': {
                'boxes': predictions['boxes'].tolist(),
                'scores': predictions['scores'].tolist(),
                'confidence_threshold': self.confidence_threshold
            },
            'layer_analysis': layer_analysis,
            'pattern_analysis': pattern_report
        }

        return result

    def print_summary(self, result):
        """
        분석 결과 요약 출력

        Args:
            result (dict): detect_from_image 결과
        """
        if result.get('num_detected', 0) == 0:
            print(result.get('message', 'No results'))
            return

        print("\n" + "=" * 60)
        print("SAGGAR DETECTION SUMMARY")
        print("=" * 60)

        # 기본 정보
        print(f"\nImage: {result['image_path']}")
        print(f"Total Saggars Detected: {result['num_detected']}")

        # 층 정보
        layer_info = result['layer_analysis']['layer_info']
        print(f"\nNumber of Layers: {result['layer_analysis']['num_layers']}")

        for layer in sorted(layer_info.keys()):
            info = layer_info[layer]
            print(f"\n  Layer {layer}:")
            print(f"    - Saggars: {info['saggar_count']}")
            print(f"    - Columns: {info['num_columns']}")
            if 'avg_confidence' in info:
                print(f"    - Avg Confidence: {info['avg_confidence']:.3f}")

        # 패턴 정보
        pattern = result['pattern_analysis']
        print(f"\nPattern Type: {pattern['summary']['pattern_description']}")

        # 안정성 정보
        stability = pattern['stability_analysis']
        print(f"\nStability Score: {stability['stability_score']:.2f}")
        if stability['warnings']:
            print("Stability Warnings:")
            for warning in stability['warnings']:
                print(f"  - {warning}")

        # 누락 정보
        gaps = pattern['gaps_analysis']
        if gaps['has_gaps']:
            print("\nGaps Detected:")
            if gaps['missing_layers']:
                print(f"  - Missing layers: {gaps['missing_layers']}")
            if gaps['incomplete_layers']:
                print(f"  - Incomplete layers: {len(gaps['incomplete_layers'])}")

        print("\n" + "=" * 60)

    def batch_process(self, image_dir, output_dir=None, pattern="*.jpg"):
        """
        디렉토리 내 모든 이미지 일괄 처리

        Args:
            image_dir (str): 이미지 디렉토리 경로
            output_dir (str, optional): 결과 저장 디렉토리
            pattern (str): 이미지 파일 패턴

        Returns:
            list: 각 이미지의 분석 결과 리스트
        """
        image_paths = list(Path(image_dir).glob(pattern))
        results = []

        print(f"Found {len(image_paths)} images to process")

        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing {image_path.name}")

            try:
                result = self.detect_from_image(str(image_path), output_dir)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        print(f"\nCompleted processing {len(results)} images")
        return results


def main():
    """
    사용 예제
    """
    import argparse

    parser = argparse.ArgumentParser(description='Saggar Detection using Mask R-CNN')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Detector 초기화
    detector = SaggarDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device
    )

    # 이미지 처리
    result = detector.detect_from_image(args.image, args.output)

    # 결과 출력
    detector.print_summary(result)


if __name__ == '__main__':
    main()
