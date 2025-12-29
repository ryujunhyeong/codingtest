"""
Mask R-CNN Model for Saggar Detection
팔렛 위의 Saggar를 감지하고 세그멘테이션하는 Mask R-CNN 모델
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class SaggarMaskRCNN:
    """
    Saggar 감지를 위한 Mask R-CNN 모델
    """

    def __init__(self, num_classes=2, pretrained=True, device=None):
        """
        Args:
            num_classes (int): 클래스 수 (배경 포함). Saggar의 경우 2 (배경 + Saggar)
            pretrained (bool): 사전 학습된 가중치 사용 여부
            device (str): 사용할 디바이스 ('cuda' 또는 'cpu')
        """
        self.num_classes = num_classes
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Mask R-CNN 모델 로드 (ResNet-50 FPN 백본)
        self.model = maskrcnn_resnet50_fpn(pretrained=pretrained)

        # Box predictor 교체
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Mask predictor 교체
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        self.model.to(self.device)

    def train_mode(self):
        """모델을 학습 모드로 전환"""
        self.model.train()

    def eval_mode(self):
        """모델을 평가 모드로 전환"""
        self.model.eval()

    def predict(self, image, confidence_threshold=0.5):
        """
        이미지에서 Saggar 감지 및 세그멘테이션

        Args:
            image (torch.Tensor): 입력 이미지 [C, H, W]
            confidence_threshold (float): 신뢰도 임계값

        Returns:
            dict: 감지 결과 (boxes, labels, scores, masks)
        """
        self.model.eval()

        with torch.no_grad():
            if image.device != self.device:
                image = image.to(self.device)

            # 배치 차원 추가
            if len(image.shape) == 3:
                image = image.unsqueeze(0)

            predictions = self.model(image)[0]

            # 신뢰도 임계값 적용
            keep = predictions['scores'] > confidence_threshold

            result = {
                'boxes': predictions['boxes'][keep].cpu(),
                'labels': predictions['labels'][keep].cpu(),
                'scores': predictions['scores'][keep].cpu(),
                'masks': predictions['masks'][keep].cpu()
            }

            return result

    def save_model(self, path):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

    def get_model(self):
        """PyTorch 모델 반환"""
        return self.model


def get_transform(train=False):
    """
    이미지 전처리 변환

    Args:
        train (bool): 학습용 변환 여부

    Returns:
        transforms: 변환 함수
    """
    transforms = []
    # 이미지를 텐서로 변환
    transforms.append(torchvision.transforms.ToTensor())

    if train:
        # 학습 시 데이터 증강 추가 가능
        pass

    return torchvision.transforms.Compose(transforms)
