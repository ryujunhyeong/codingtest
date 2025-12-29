"""
Saggar Layer Counting Algorithm
측면 사진에서 Saggar의 층수를 계산하는 알고리즘
"""

import numpy as np
from scipy.cluster.hierarchy import fclusterdata
import cv2


class SaggarLayerCounter:
    """
    측면에서 본 Saggar 사진에서 층수와 배열을 분석하는 클래스
    """

    def __init__(self, vertical_tolerance=20, horizontal_tolerance=30):
        """
        Args:
            vertical_tolerance (int): 같은 층으로 판단하는 Y축 허용 오차 (픽셀)
            horizontal_tolerance (int): 같은 열로 판단하는 X축 허용 오차 (픽셀)
        """
        self.vertical_tolerance = vertical_tolerance
        self.horizontal_tolerance = horizontal_tolerance

    def get_saggar_centers(self, boxes):
        """
        바운딩 박스에서 Saggar의 중심점 계산

        Args:
            boxes (np.ndarray): 바운딩 박스 배열 [N, 4] (x1, y1, x2, y2)

        Returns:
            np.ndarray: 중심점 배열 [N, 2] (center_x, center_y)
        """
        centers = np.zeros((len(boxes), 2))
        centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # center_x
        centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # center_y
        return centers

    def cluster_vertical_positions(self, centers):
        """
        Y축 좌표를 기반으로 층 클러스터링

        Args:
            centers (np.ndarray): Saggar 중심점 [N, 2]

        Returns:
            np.ndarray: 각 Saggar의 층 번호 [N]
        """
        if len(centers) == 0:
            return np.array([])

        # Y 좌표만 추출 (측면 사진이므로 Y축이 높이를 나타냄)
        y_coords = centers[:, 1].reshape(-1, 1)

        # 계층적 클러스터링으로 층 구분
        # 위에서 아래로 정렬하기 위해 Y 좌표 사용
        layer_labels = fclusterdata(
            y_coords,
            t=self.vertical_tolerance,
            criterion='distance',
            method='complete'
        )

        # 층 번호를 위에서 아래로 재정렬 (Y값이 작을수록 위)
        unique_layers = np.unique(layer_labels)
        layer_avg_y = {layer: np.mean(y_coords[layer_labels == layer]) for layer in unique_layers}
        sorted_layers = sorted(layer_avg_y.items(), key=lambda x: x[1])

        # 층 번호 재매핑 (1부터 시작)
        layer_mapping = {old: new for new, (old, _) in enumerate(sorted_layers, 1)}
        remapped_layers = np.array([layer_mapping[label] for label in layer_labels])

        return remapped_layers

    def cluster_horizontal_positions(self, centers, layer_labels):
        """
        각 층에서 X축 좌표를 기반으로 열 클러스터링

        Args:
            centers (np.ndarray): Saggar 중심점 [N, 2]
            layer_labels (np.ndarray): 각 Saggar의 층 번호 [N]

        Returns:
            dict: {층 번호: 열 배열 정보}
        """
        layer_columns = {}

        unique_layers = np.unique(layer_labels)
        for layer in unique_layers:
            layer_mask = layer_labels == layer
            layer_centers = centers[layer_mask]

            if len(layer_centers) == 0:
                continue

            # X 좌표만 추출
            x_coords = layer_centers[:, 0].reshape(-1, 1)

            # 열 클러스터링
            if len(x_coords) > 1:
                column_labels = fclusterdata(
                    x_coords,
                    t=self.horizontal_tolerance,
                    criterion='distance',
                    method='complete'
                )
            else:
                column_labels = np.array([1])

            # 열 번호를 왼쪽에서 오른쪽으로 재정렬
            unique_cols = np.unique(column_labels)
            col_avg_x = {col: np.mean(x_coords[column_labels == col]) for col in unique_cols}
            sorted_cols = sorted(col_avg_x.items(), key=lambda x: x[1])
            col_mapping = {old: new for new, (old, _) in enumerate(sorted_cols, 1)}
            remapped_cols = np.array([col_mapping[label] for label in column_labels])

            layer_columns[layer] = {
                'num_columns': len(unique_cols),
                'column_labels': remapped_cols,
                'saggar_count': len(layer_centers)
            }

        return layer_columns

    def analyze_stacking_pattern(self, boxes, scores=None):
        """
        Saggar 쌓임 패턴 분석

        Args:
            boxes (np.ndarray or torch.Tensor): 바운딩 박스 [N, 4]
            scores (np.ndarray or torch.Tensor, optional): 신뢰도 점수 [N]

        Returns:
            dict: 층수, 배열 패턴 등의 분석 결과
        """
        # Tensor를 numpy로 변환
        if hasattr(boxes, 'numpy'):
            boxes = boxes.numpy()
        if scores is not None and hasattr(scores, 'numpy'):
            scores = scores.numpy()

        if len(boxes) == 0:
            return {
                'total_saggars': 0,
                'num_layers': 0,
                'layer_info': {},
                'pattern_description': 'No Saggars detected'
            }

        # Saggar 중심점 계산
        centers = self.get_saggar_centers(boxes)

        # 층 클러스터링
        layer_labels = self.cluster_vertical_positions(centers)
        num_layers = len(np.unique(layer_labels))

        # 각 층의 열 정보 계산
        layer_columns = self.cluster_horizontal_positions(centers, layer_labels)

        # 각 층의 상세 정보 생성
        layer_info = {}
        for layer in range(1, num_layers + 1):
            layer_mask = layer_labels == layer
            layer_boxes = boxes[layer_mask]
            layer_centers = centers[layer_mask]

            # 층의 평균 높이 계산
            avg_height = np.mean(layer_boxes[:, 3] - layer_boxes[:, 1])
            avg_y = np.mean(layer_centers[:, 1])

            layer_info[layer] = {
                'saggar_count': int(np.sum(layer_mask)),
                'avg_y_position': float(avg_y),
                'avg_height': float(avg_height),
                'num_columns': layer_columns[layer]['num_columns'],
                'column_labels': layer_columns[layer]['column_labels'].tolist()
            }

            if scores is not None:
                layer_scores = scores[layer_mask]
                layer_info[layer]['avg_confidence'] = float(np.mean(layer_scores))

        # 패턴 설명 생성
        pattern_desc = self._generate_pattern_description(num_layers, layer_info)

        return {
            'total_saggars': len(boxes),
            'num_layers': num_layers,
            'layer_labels': layer_labels.tolist(),
            'layer_info': layer_info,
            'pattern_description': pattern_desc,
            'centers': centers.tolist()
        }

    def _generate_pattern_description(self, num_layers, layer_info):
        """
        쌓임 패턴에 대한 텍스트 설명 생성

        Args:
            num_layers (int): 총 층수
            layer_info (dict): 각 층의 상세 정보

        Returns:
            str: 패턴 설명
        """
        desc_parts = [f"Total {num_layers} layers detected"]

        for layer in sorted(layer_info.keys()):
            info = layer_info[layer]
            desc_parts.append(
                f"Layer {layer}: {info['saggar_count']} Saggars "
                f"in {info['num_columns']} column(s)"
            )

        return "; ".join(desc_parts)

    def visualize_layers(self, image, boxes, layer_labels, output_path=None):
        """
        층별로 색상을 다르게 하여 시각화

        Args:
            image (np.ndarray): 원본 이미지
            boxes (np.ndarray): 바운딩 박스 [N, 4]
            layer_labels (np.ndarray): 층 레이블 [N]
            output_path (str, optional): 저장 경로

        Returns:
            np.ndarray: 시각화된 이미지
        """
        # 이미지 복사
        vis_image = image.copy()

        # 층별 색상 정의 (HSV 색공간 활용)
        num_layers = len(np.unique(layer_labels))
        colors = []
        for i in range(num_layers):
            hue = int(180 * i / num_layers)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))

        # 각 Saggar 그리기
        for box, layer in zip(boxes, layer_labels):
            x1, y1, x2, y2 = map(int, box)
            color = colors[layer - 1]

            # 바운딩 박스 그리기
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 층 번호 표시
            cv2.putText(
                vis_image,
                f"L{layer}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        if output_path:
            cv2.imwrite(output_path, vis_image)

        return vis_image
