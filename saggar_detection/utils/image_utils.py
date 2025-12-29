"""
Image processing utilities for Saggar detection
"""

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T


def load_image(image_path):
    """
    이미지 로드

    Args:
        image_path (str): 이미지 파일 경로

    Returns:
        tuple: (PIL Image, numpy array)
    """
    # PIL로 로드
    pil_image = Image.open(image_path).convert('RGB')

    # Numpy로 변환 (OpenCV 형식)
    np_image = cv2.imread(image_path)
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    return pil_image, np_image


def prepare_image_for_model(image, device='cpu'):
    """
    모델 입력을 위한 이미지 전처리

    Args:
        image (PIL.Image or np.ndarray): 입력 이미지
        device (str): 디바이스 ('cpu' or 'cuda')

    Returns:
        torch.Tensor: 전처리된 이미지 텐서 [C, H, W]
    """
    if isinstance(image, np.ndarray):
        # Numpy array를 PIL Image로 변환
        image = Image.fromarray(image)

    # Tensor로 변환
    transform = T.ToTensor()
    image_tensor = transform(image)

    return image_tensor.to(device)


def draw_detection_results(image, boxes, masks=None, labels=None, scores=None, show_masks=True):
    """
    감지 결과를 이미지에 시각화

    Args:
        image (np.ndarray): 원본 이미지
        boxes (np.ndarray): 바운딩 박스 [N, 4]
        masks (np.ndarray, optional): 마스크 [N, H, W]
        labels (np.ndarray, optional): 레이블 [N]
        scores (np.ndarray, optional): 신뢰도 점수 [N]
        show_masks (bool): 마스크 표시 여부

    Returns:
        np.ndarray: 시각화된 이미지
    """
    vis_image = image.copy()

    if len(boxes) == 0:
        return vis_image

    # Tensor를 numpy로 변환
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()
    if masks is not None and torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    if labels is not None and torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if scores is not None and torch.is_tensor(scores):
        scores = scores.cpu().numpy()

    # 마스크 오버레이
    if show_masks and masks is not None:
        for i, mask in enumerate(masks):
            # 마스크 이진화 (threshold = 0.5)
            mask = mask.squeeze()
            mask = (mask > 0.5).astype(np.uint8)

            # 마스크 크기를 이미지 크기에 맞춤
            if mask.shape != vis_image.shape[:2]:
                mask = cv2.resize(mask, (vis_image.shape[1], vis_image.shape[0]))

            # 랜덤 색상 생성
            color = np.random.randint(0, 255, 3).tolist()

            # 마스크 오버레이
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask == 1] = color

            vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, 0.5, 0)

    # 바운딩 박스 및 텍스트
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # 바운딩 박스 그리기
        color = (0, 255, 0)  # 초록색
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # 텍스트 준비
        text_parts = []
        if labels is not None:
            text_parts.append(f"Saggar")
        if scores is not None:
            text_parts.append(f"{scores[i]:.2f}")

        if text_parts:
            text = " ".join(text_parts)

            # 텍스트 배경
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_h - 10),
                (x1 + text_w, y1),
                color,
                -1
            )

            # 텍스트
            cv2.putText(
                vis_image,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    return vis_image


def save_image(image, output_path):
    """
    이미지 저장

    Args:
        image (np.ndarray): 저장할 이미지
        output_path (str): 저장 경로
    """
    # RGB를 BGR로 변환 (OpenCV는 BGR 사용)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    cv2.imwrite(output_path, image_bgr)
    print(f"Image saved to {output_path}")


def resize_image(image, max_size=1024):
    """
    이미지 크기 조정 (종횡비 유지)

    Args:
        image (np.ndarray or PIL.Image): 입력 이미지
        max_size (int): 최대 크기 (긴 변 기준)

    Returns:
        이미지와 동일한 타입: 크기 조정된 이미지
    """
    is_numpy = isinstance(image, np.ndarray)

    if is_numpy:
        h, w = image.shape[:2]
    else:
        w, h = image.size

    # 긴 변 기준으로 스케일 계산
    scale = max_size / max(h, w)

    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)

        if is_numpy:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = image.resize((new_w, new_h), Image.LANCZOS)

        return resized
    else:
        return image


def enhance_image(image):
    """
    이미지 품질 향상 (대비 증가, 노이즈 제거 등)

    Args:
        image (np.ndarray): 입력 이미지

    Returns:
        np.ndarray: 향상된 이미지
    """
    # 노이즈 제거
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # 대비 향상 (CLAHE)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    enhanced_lab = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return enhanced
