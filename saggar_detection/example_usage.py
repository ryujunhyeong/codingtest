"""
Saggar Detection 사용 예제
"""

import os
from detector import SaggarDetector


def example_single_image():
    """
    단일 이미지 처리 예제
    """
    print("=" * 60)
    print("Example 1: Single Image Detection")
    print("=" * 60)

    # Detector 초기화
    detector = SaggarDetector(
        model_path=None,  # 사전 학습된 모델 사용 (또는 학습된 모델 경로 지정)
        confidence_threshold=0.5,
        device='cuda',  # 또는 'cpu'
        vertical_tolerance=20,
        horizontal_tolerance=30
    )

    # 이미지 경로
    image_path = "data/test/sample_image.jpg"

    # 출력 디렉토리
    output_dir = "output/single_image"

    # 감지 및 분석
    result = detector.detect_from_image(image_path, output_dir)

    # 결과 출력
    detector.print_summary(result)

    return result


def example_batch_processing():
    """
    여러 이미지 일괄 처리 예제
    """
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)

    # Detector 초기화
    detector = SaggarDetector(
        confidence_threshold=0.6,
        device='cuda'
    )

    # 이미지 디렉토리
    image_dir = "data/test"
    output_dir = "output/batch"

    # 일괄 처리
    results = detector.batch_process(
        image_dir=image_dir,
        output_dir=output_dir,
        pattern="*.jpg"
    )

    # 통계 출력
    print(f"\nProcessed {len(results)} images")

    total_saggars = sum(r['num_detected'] for r in results)
    avg_layers = sum(
        r.get('layer_analysis', {}).get('num_layers', 0)
        for r in results
    ) / len(results) if results else 0

    print(f"Total Saggars detected: {total_saggars}")
    print(f"Average layers per image: {avg_layers:.1f}")

    return results


def example_with_expected_config():
    """
    기대 구성과 비교하는 예제
    """
    print("\n" + "=" * 60)
    print("Example 3: Comparison with Expected Configuration")
    print("=" * 60)

    detector = SaggarDetector(confidence_threshold=0.5)

    image_path = "data/test/sample_image.jpg"
    result = detector.detect_from_image(image_path)

    # 기대하는 구성 설정
    expected_config = {
        'num_layers': 5,
        'saggars_per_layer': 12,
        'total_saggars': 60
    }

    # 비교
    comparison = detector.pattern_analyzer.compare_with_expected(
        result['layer_analysis']['layer_info'],
        expected_config
    )

    print("\nComparison Results:")
    print(f"Matches Expected: {comparison['matches_expected']}")

    if comparison['differences']:
        print("\nDifferences:")
        for diff in comparison['differences']:
            print(f"  {diff['category']}: Expected {diff['expected']}, "
                  f"Got {diff['actual']} (Δ {diff['difference']:+d})")

    return comparison


def example_custom_analysis():
    """
    커스텀 분석 예제
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Analysis")
    print("=" * 60)

    detector = SaggarDetector(
        confidence_threshold=0.5,
        vertical_tolerance=25,  # 더 넓은 허용 오차
        horizontal_tolerance=35
    )

    image_path = "data/test/sample_image.jpg"
    result = detector.detect_from_image(image_path, output_dir="output/custom")

    # 층별 상세 분석
    layer_info = result['layer_analysis']['layer_info']

    print("\nDetailed Layer Analysis:")
    for layer, info in sorted(layer_info.items()):
        print(f"\nLayer {layer}:")
        print(f"  Total Saggars: {info['saggar_count']}")
        print(f"  Columns: {info['num_columns']}")
        print(f"  Average Y Position: {info['avg_y_position']:.1f}")
        print(f"  Average Height: {info['avg_height']:.1f} pixels")

        if 'avg_confidence' in info:
            print(f"  Detection Confidence: {info['avg_confidence']:.3f}")

    # 패턴 분석 상세
    pattern = result['pattern_analysis']

    print(f"\nPattern Analysis:")
    print(f"  Type: {pattern['summary']['pattern_description']}")

    # 격자 배열 정보
    grid = pattern['grid_arrangement']
    print(f"\nGrid Arrangement:")
    for layer, grid_info in sorted(grid.items()):
        print(f"  Layer {layer}: {grid_info['columns']}x{grid_info['max_rows']} "
              f"({'Complete' if grid_info['is_complete_grid'] else 'Incomplete'})")

    # 안정성 분석
    stability = pattern['stability_analysis']
    print(f"\nStability Analysis:")
    print(f"  Stability Score: {stability['stability_score']:.2f}/1.0")
    print(f"  Status: {'STABLE' if stability['is_stable'] else 'UNSTABLE'}")

    if stability['warnings']:
        print("  Warnings:")
        for warning in stability['warnings']:
            print(f"    - {warning}")

    return result


def example_visualization_only():
    """
    시각화만 수행하는 예제
    """
    print("\n" + "=" * 60)
    print("Example 5: Visualization Only")
    print("=" * 60)

    detector = SaggarDetector(confidence_threshold=0.7)

    image_path = "data/test/sample_image.jpg"

    # 감지만 수행 (시각화 없음)
    result = detector.detect_from_image(image_path, output_dir=None)

    # 나중에 시각화
    from utils.image_utils import load_image, draw_detection_results, save_image
    import numpy as np

    _, np_image = load_image(image_path)

    # 감지 결과 시각화
    vis_image = draw_detection_results(
        np_image,
        np.array(result['detection_info']['boxes']),
        show_masks=False
    )

    os.makedirs("output/viz_only", exist_ok=True)
    save_image(vis_image, "output/viz_only/visualization.jpg")

    print("Visualization saved")


def main():
    """
    모든 예제 실행
    """
    # 주의: 실제 이미지 파일이 필요합니다
    # 아래 예제들은 data/test/ 디렉토리에 이미지가 있다고 가정합니다

    print("Saggar Detection Examples\n")

    # 예제 실행을 위해서는 실제 이미지 파일이 필요합니다
    # 각 예제를 개별적으로 실행할 수 있습니다

    print("\nAvailable examples:")
    print("1. example_single_image() - 단일 이미지 처리")
    print("2. example_batch_processing() - 여러 이미지 일괄 처리")
    print("3. example_with_expected_config() - 기대 구성과 비교")
    print("4. example_custom_analysis() - 커스텀 분석")
    print("5. example_visualization_only() - 시각화만 수행")

    print("\n사용법:")
    print("from example_usage import example_single_image")
    print("result = example_single_image()")


if __name__ == '__main__':
    main()
