#!/usr/bin/env python3
"""
Mask R-CNN Detector Node
Mask R-CNN을 사용한 실시간 Saggar 감지 노드
"""

import rospy
import cv2
import numpy as np
import torch
import sys
import os
import time
from pathlib import Path

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from saggar_detection.models.maskrcnn_model import SaggarMaskRCNN
from saggar_detection.utils.layer_counter import SaggarLayerCounter
from saggar_detection.utils.pattern_analyzer import SaggarPatternAnalyzer

# Import custom messages
from saggar_detection_ros.msg import SaggarDetection, SaggarArray, LayerInfo


class MaskRCNNDetectorNode:
    """
    Mask R-CNN 기반 Saggar 감지 ROS 노드
    """

    def __init__(self):
        rospy.init_node('maskrcnn_detector_node', anonymous=False)

        # Parameters
        self.model_path = rospy.get_param('~model_path', None)
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)
        self.device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.vertical_tolerance = rospy.get_param('~vertical_tolerance', 20)
        self.horizontal_tolerance = rospy.get_param('~horizontal_tolerance', 30)
        self.publish_visualization = rospy.get_param('~publish_visualization', True)
        self.input_topic = rospy.get_param('~input_topic', '/saggar/image_raw')
        self.output_topic = rospy.get_param('~output_topic', '/saggar/detections/maskrcnn')
        self.viz_topic = rospy.get_param('~viz_topic', '/saggar/visualization/maskrcnn')

        # Initialize Mask R-CNN model
        rospy.loginfo("Loading Mask R-CNN model...")
        self.model = SaggarMaskRCNN(
            num_classes=2,
            device=self.device
        )

        if self.model_path and os.path.exists(self.model_path):
            self.model.load_model(self.model_path)
            rospy.loginfo(f"Loaded model from {self.model_path}")
        else:
            rospy.logwarn("Using pretrained Mask R-CNN (not fine-tuned for Saggar)")

        self.model.eval_mode()

        # Initialize layer counter and pattern analyzer
        self.layer_counter = SaggarLayerCounter(
            vertical_tolerance=self.vertical_tolerance,
            horizontal_tolerance=self.horizontal_tolerance
        )
        self.pattern_analyzer = SaggarPatternAnalyzer()

        # CV Bridge
        self.bridge = CvBridge()

        # Publishers
        self.detection_pub = rospy.Publisher(
            self.output_topic,
            SaggarArray,
            queue_size=10
        )

        if self.publish_visualization:
            self.viz_pub = rospy.Publisher(
                self.viz_topic,
                Image,
                queue_size=10
            )

        # Subscriber
        self.image_sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

        rospy.loginfo("Mask R-CNN Detector Node initialized")
        rospy.loginfo(f"Device: {self.device}")
        rospy.loginfo(f"Confidence threshold: {self.confidence_threshold}")
        rospy.loginfo(f"Subscribing to: {self.input_topic}")
        rospy.loginfo(f"Publishing detections to: {self.output_topic}")

    def get_contour_points(self, mask):
        """
        마스크에서 내부 테두리(contour) 추출

        Args:
            mask: Binary mask [H, W]

        Returns:
            List of Point messages
        """
        # 마스크를 uint8로 변환
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Contour 찾기
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 가장 큰 contour 선택
        if len(contours) == 0:
            return []

        largest_contour = max(contours, key=cv2.contourArea)

        # Contour를 Point 메시지로 변환
        contour_points = []
        for point in largest_contour:
            pt = Point()
            pt.x = float(point[0][0])
            pt.y = float(point[0][1])
            pt.z = 0.0
            contour_points.append(pt)

        return contour_points

    def image_callback(self, msg):
        """
        이미지 콜백 함수
        """
        start_time = time.time()

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        # Convert to tensor
        image_tensor = torch.from_numpy(cv_image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        # Run detection
        predictions = self.model.predict(
            image_tensor,
            confidence_threshold=self.confidence_threshold
        )

        num_detected = len(predictions['boxes'])

        if num_detected == 0:
            rospy.loginfo("No Saggars detected")
            # Publish empty detection
            detection_msg = SaggarArray()
            detection_msg.header = msg.header
            detection_msg.method = "maskrcnn"
            detection_msg.total_count = 0
            detection_msg.num_layers = 0
            detection_msg.processing_time = (time.time() - start_time) * 1000
            self.detection_pub.publish(detection_msg)
            return

        # Analyze layers
        layer_analysis = self.layer_counter.analyze_stacking_pattern(
            predictions['boxes'],
            predictions['scores']
        )

        # Create detection messages
        detection_array = SaggarArray()
        detection_array.header = msg.header
        detection_array.method = "maskrcnn"
        detection_array.total_count = num_detected
        detection_array.num_layers = layer_analysis['num_layers']

        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions['masks'].cpu().numpy()
        layer_labels = layer_analysis['layer_labels']
        centers = layer_analysis['centers']

        # Get column information per layer
        layer_info = layer_analysis['layer_info']

        for i in range(num_detected):
            detection = SaggarDetection()
            detection.header = msg.header

            # Bounding box
            x1, y1, x2, y2 = boxes[i]
            detection.x = float(x1)
            detection.y = float(y1)
            detection.width = float(x2 - x1)
            detection.height = float(y2 - y1)

            # Center point
            detection.center.x = float(centers[i][0])
            detection.center.y = float(centers[i][1])
            detection.center.z = 0.0

            # Confidence
            detection.confidence = float(scores[i])

            # Layer and column info
            detection.layer_id = int(layer_labels[i])

            # Get column id from layer info
            layer_id = layer_labels[i]
            if layer_id in layer_info:
                # Find which column this detection belongs to
                col_labels = layer_info[layer_id]['column_labels']
                layer_mask = np.array(layer_labels) == layer_id
                layer_indices = np.where(layer_mask)[0]
                idx_in_layer = np.where(layer_indices == i)[0][0]
                detection.column_id = int(col_labels[idx_in_layer])
            else:
                detection.column_id = 0

            # Contour points
            mask = masks[i].squeeze()
            mask = (mask > 0.5).astype(np.uint8)

            # Resize mask to image size if needed
            if mask.shape != cv_image.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (cv_image.shape[1], cv_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            contour_points = self.get_contour_points(mask)
            detection.contour_points = contour_points

            # Mask as image (optional, can be disabled for performance)
            try:
                mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                detection.mask = mask_msg
            except CvBridgeError as e:
                rospy.logwarn(f"Failed to convert mask: {e}")

            detection_array.detections.append(detection)

        # Processing time
        processing_time = (time.time() - start_time) * 1000
        detection_array.processing_time = processing_time

        # Publish detections
        self.detection_pub.publish(detection_array)

        rospy.loginfo(f"Detected {num_detected} Saggars in {layer_analysis['num_layers']} layers "
                     f"({processing_time:.1f}ms)")

        # Visualization
        if self.publish_visualization:
            self.publish_visualization_image(cv_image, boxes, masks, scores, layer_labels, msg.header)

    def publish_visualization_image(self, image, boxes, masks, scores, layer_labels, header):
        """
        시각화 이미지 발행
        """
        vis_image = image.copy()

        # 층별 색상 생성
        num_layers = len(np.unique(layer_labels))
        colors = []
        for i in range(num_layers):
            hue = int(180 * i / max(num_layers, 1))
            color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            color_rgb = tuple(map(int, [color_bgr[2], color_bgr[1], color_bgr[0]]))
            colors.append(color_rgb)

        # Draw masks
        for i, mask in enumerate(masks):
            mask = mask.squeeze()
            mask = (mask > 0.5).astype(np.uint8)

            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

            layer_id = layer_labels[i]
            color = colors[layer_id - 1]

            # Overlay mask
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask == 1] = color
            vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, 0.4, 0)

            # Draw contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, color, 2)

        # Draw boxes and labels
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            layer_id = layer_labels[i]
            color = colors[layer_id - 1]

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(vis_image, (cx, cy), 5, color, -1)

            # Label
            label = f"L{layer_id} {scores[i]:.2f}"
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Publish
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='rgb8')
            viz_msg.header = header
            self.viz_pub.publish(viz_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to publish visualization: {e}")

    def run(self):
        """
        노드 실행
        """
        rospy.spin()


def main():
    try:
        node = MaskRCNNDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
