"""
Saggar Array Pattern Detection
Saggar의 배열 패턴을 분석하고 분류하는 알고리즘
"""

import numpy as np
from collections import Counter


class SaggarPatternAnalyzer:
    """
    Saggar 배열 패턴을 분석하는 클래스
    """

    def __init__(self):
        self.pattern_types = {
            'uniform': '균일 배열',
            'pyramid': '피라미드형',
            'irregular': '불규칙 배열',
            'alternating': '교차 배열'
        }

    def detect_pattern_type(self, layer_info):
        """
        Saggar 쌓임 패턴 유형 감지

        Args:
            layer_info (dict): 각 층의 정보

        Returns:
            str: 패턴 유형
        """
        if not layer_info:
            return 'empty'

        num_layers = len(layer_info)
        if num_layers == 1:
            return 'single_layer'

        # 각 층의 Saggar 개수 추출
        layer_counts = [info['saggar_count'] for info in layer_info.values()]

        # 균일 배열 체크
        if len(set(layer_counts)) == 1:
            return 'uniform'

        # 피라미드형 체크 (위로 갈수록 감소)
        if self._is_pyramid_pattern(layer_counts):
            return 'pyramid'

        # 교차 배열 체크
        if self._is_alternating_pattern(layer_counts):
            return 'alternating'

        # 기타는 불규칙 배열
        return 'irregular'

    def _is_pyramid_pattern(self, counts):
        """
        피라미드 패턴 여부 확인 (아래층이 위층보다 많거나 같음)

        Args:
            counts (list): 각 층의 Saggar 개수

        Returns:
            bool: 피라미드 패턴 여부
        """
        for i in range(len(counts) - 1):
            if counts[i] > counts[i + 1]:  # 위층이 아래층보다 많으면 피라미드 아님
                return False
        return True

    def _is_alternating_pattern(self, counts):
        """
        교차 배열 패턴 확인 (홀수/짝수 층의 개수가 교차)

        Args:
            counts (list): 각 층의 Saggar 개수

        Returns:
            bool: 교차 패턴 여부
        """
        if len(counts) < 3:
            return False

        # 홀수 인덱스와 짝수 인덱스의 패턴 확인
        even_counts = [counts[i] for i in range(0, len(counts), 2)]
        odd_counts = [counts[i] for i in range(1, len(counts), 2)]

        # 각 그룹 내에서 값이 동일하고, 두 그룹이 다른 경우
        even_uniform = len(set(even_counts)) == 1
        odd_uniform = len(set(odd_counts)) == 1

        return even_uniform and odd_uniform and even_counts[0] != odd_counts[0]

    def calculate_grid_arrangement(self, layer_info):
        """
        각 층의 격자 배열 계산

        Args:
            layer_info (dict): 각 층의 정보

        Returns:
            dict: 격자 배열 정보
        """
        grid_info = {}

        for layer, info in layer_info.items():
            num_cols = info['num_columns']
            num_saggars = info['saggar_count']

            # 각 열의 Saggar 개수 계산
            column_labels = info['column_labels']
            col_counts = Counter(column_labels)

            # 행 개수 추정 (가장 많은 열 기준)
            max_rows = max(col_counts.values())

            grid_info[layer] = {
                'columns': num_cols,
                'max_rows': max_rows,
                'column_distribution': dict(col_counts),
                'total': num_saggars,
                'is_complete_grid': num_saggars == num_cols * max_rows
            }

        return grid_info

    def detect_gaps_or_missing(self, layer_info, expected_pattern=None):
        """
        누락되거나 빈 공간 감지

        Args:
            layer_info (dict): 각 층의 정보
            expected_pattern (str, optional): 예상 패턴

        Returns:
            dict: 누락 정보
        """
        gaps_info = {
            'has_gaps': False,
            'missing_layers': [],
            'incomplete_layers': []
        }

        if not layer_info:
            return gaps_info

        # 층 번호 연속성 체크
        layer_numbers = sorted(layer_info.keys())
        expected_layers = list(range(1, max(layer_numbers) + 1))

        for expected in expected_layers:
            if expected not in layer_numbers:
                gaps_info['missing_layers'].append(expected)
                gaps_info['has_gaps'] = True

        # 각 층의 완전성 체크
        for layer, info in layer_info.items():
            # 열별 Saggar 개수가 일정하지 않으면 불완전
            col_counts = Counter(info['column_labels'])
            if len(set(col_counts.values())) > 1:
                gaps_info['incomplete_layers'].append({
                    'layer': layer,
                    'reason': 'uneven_column_distribution',
                    'distribution': dict(col_counts)
                })
                gaps_info['has_gaps'] = True

        return gaps_info

    def analyze_stability(self, layer_info):
        """
        쌓임 안정성 분석

        Args:
            layer_info (dict): 각 층의 정보

        Returns:
            dict: 안정성 분석 결과
        """
        stability = {
            'is_stable': True,
            'warnings': [],
            'stability_score': 1.0
        }

        if len(layer_info) <= 1:
            return stability

        layer_numbers = sorted(layer_info.keys())
        layer_counts = [layer_info[i]['saggar_count'] for i in layer_numbers]

        # 위층이 아래층보다 많으면 불안정
        for i in range(len(layer_counts) - 1):
            upper_layer = layer_numbers[i]
            lower_layer = layer_numbers[i + 1]

            if layer_counts[i] > layer_counts[i + 1]:
                stability['is_stable'] = False
                stability['warnings'].append(
                    f"Layer {upper_layer} has more Saggars than layer {lower_layer} below it"
                )
                stability['stability_score'] -= 0.2

        # 큰 불균형 체크
        if len(layer_counts) > 2:
            avg_count = np.mean(layer_counts)
            std_count = np.std(layer_counts)

            if std_count > avg_count * 0.5:  # 표준편차가 평균의 50% 이상
                stability['warnings'].append(
                    f"Large variation in layer sizes (std={std_count:.1f})"
                )
                stability['stability_score'] -= 0.1

        stability['stability_score'] = max(0.0, stability['stability_score'])

        return stability

    def generate_comprehensive_report(self, layer_info):
        """
        종합 분석 보고서 생성

        Args:
            layer_info (dict): 각 층의 정보

        Returns:
            dict: 종합 분석 결과
        """
        pattern_type = self.detect_pattern_type(layer_info)
        grid_arrangement = self.calculate_grid_arrangement(layer_info)
        gaps_info = self.detect_gaps_or_missing(layer_info)
        stability = self.analyze_stability(layer_info)

        total_saggars = sum(info['saggar_count'] for info in layer_info.values())

        report = {
            'summary': {
                'total_saggars': total_saggars,
                'num_layers': len(layer_info),
                'pattern_type': pattern_type,
                'pattern_description': self.pattern_types.get(pattern_type, pattern_type)
            },
            'layer_details': layer_info,
            'grid_arrangement': grid_arrangement,
            'gaps_analysis': gaps_info,
            'stability_analysis': stability
        }

        return report

    def compare_with_expected(self, layer_info, expected_config):
        """
        기대 구성과 실제 감지 결과 비교

        Args:
            layer_info (dict): 실제 감지된 층 정보
            expected_config (dict): 기대하는 구성 정보
                예: {'num_layers': 5, 'saggars_per_layer': 12}

        Returns:
            dict: 비교 결과
        """
        comparison = {
            'matches_expected': True,
            'differences': []
        }

        actual_layers = len(layer_info)
        actual_total = sum(info['saggar_count'] for info in layer_info.values())

        # 층수 비교
        if 'num_layers' in expected_config:
            expected_layers = expected_config['num_layers']
            if actual_layers != expected_layers:
                comparison['matches_expected'] = False
                comparison['differences'].append({
                    'category': 'num_layers',
                    'expected': expected_layers,
                    'actual': actual_layers,
                    'difference': actual_layers - expected_layers
                })

        # 층별 Saggar 개수 비교
        if 'saggars_per_layer' in expected_config:
            expected_per_layer = expected_config['saggars_per_layer']

            for layer, info in layer_info.items():
                if info['saggar_count'] != expected_per_layer:
                    comparison['matches_expected'] = False
                    comparison['differences'].append({
                        'category': 'layer_count',
                        'layer': layer,
                        'expected': expected_per_layer,
                        'actual': info['saggar_count'],
                        'difference': info['saggar_count'] - expected_per_layer
                    })

        # 전체 개수 비교
        if 'total_saggars' in expected_config:
            expected_total = expected_config['total_saggars']
            if actual_total != expected_total:
                comparison['matches_expected'] = False
                comparison['differences'].append({
                    'category': 'total_saggars',
                    'expected': expected_total,
                    'actual': actual_total,
                    'difference': actual_total - expected_total
                })

        return comparison
