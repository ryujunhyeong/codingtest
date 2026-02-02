#!/usr/bin/env python3
"""
Edge Detection Based Saggar Detector Node
전통적인 Edge Detection 방식을 사용한 Saggar 감지 노드
"""

import rospy
import cv2
import numpy as np
import sys
import os
import time
from scipy.cluster.hierarchy import fclusterdata

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from saggar_detection.utils.layer_counter import SaggarLayerCounter

# Import custom messages
from saggar_detection_ros.msg import SaggarDetection, SaggarArray


class EdgeDetectorNode:
    """
    Edge Detection 기반 Saggar 감지 ROS 노드
    """

    def __init__(self):
        rospy.init_node('edge_detector_node', anonymous=False)

        # Parameters
        self.canny_low = rospy.get_param('~canny_low', 50)
        self.canny_high = rospy.get_param('~canny_high', 150)
        self.blur_kernel = rospy.get_param('~blur_kernel', 5)
        self.min_contour_area = rospy.get_param('~min_contour_area', 500)
        self.max_contour_area = rospy.get_param('~max_contour_area', 50000)
        self.vertical_tolerance = rospy.get_param('~vertical_tolerance', 20)
        self.horizontal_tolerance = rospy.get_param('~horizontal_tolerance', 30)
        self.publish_visualization = rospy.get_param('~publish_visualization', True)
        self.input_topic = rospy.get_param('~input_topic', '/saggar/image_raw')
        self.output_topic = rospy.get_param('~output_topic', '/saggar/detections/edge')
        self.viz_topic = rospy.get_param('~viz_topic', '/saggar/visualization/edge')

        # Morphological operation parameters
        self.morph_kernel_size = rospy.get_param('~morph_kernel_size', 3)
        self.use_adaptive_threshold = rospy.get_param('~use_adaptive_threshold', False)

        # Initialize layer counter
        self.layer_counter = SaggarLayerCounter(
            vertical_tolerance=self.vertical_tolerance,
            horizontal_tolerance=self.horizontal_tolerance
        )

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

        rospy.loginfo("Edge Detector Node initialized")
        rospy.loginfo(f"Canny thresholds: ({self.canny_low}, {self.canny_high})")
        rospy.loginfo(f"Min/Max contour area: ({self.min_contour_area}, {self.max_contour_area})")
        rospy.loginfo(f"Subscribing to: {self.input_topic}")
        rospy.loginfo(f"Publishing detections to: {self.output_topic}")

    def preprocess_image(self, image):
        """
        이미지 전처리

        Args:
            image: RGB image

        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        return enhanced

    def detect_edges(self, image):
        """
        Edge detection using Canny

        Args:
            image: Preprocessed grayscale image

        Returns:
            Edge image
        """
        if self.use_adaptive_threshold:
            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
        else:
            # Canny edge detection
            edges = cv2.Canny(image, self.canny_low, self.canny_high)
            binary = edges

        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=1)

        return dilated

    def find_saggars(self, edge_image):
        """
        Contour 기반 Saggar 감지

        Args:
            edge_image: Edge detection 결과

        Returns:
            List of (bbox, contour, center)
        """
        # Find contours
        contours, hierarchy = cv2.findContours(
            edge_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_contour_area or area > self.max_contour_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio (Saggar는 일반적으로 세로가 더 김)
            aspect_ratio = h / max(w, 1)
            if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                continue

            # Calculate center
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx = x + w // 2
                cy = y + h // 2

            center = (cx, cy)

            detections.append({
                'bbox': (x, y, w, h),
                'contour': contour,
                'center': center,
                'area': area
            })

        return detections

    def cluster_layers(self, detections):
        """
        감지된 Saggar를 층별로 클러스터링

        Args:
            detections: List of detection dicts

        Returns:
            detections with layer_id added
        """
        if len(detections) == 0:
            return detections

        # Extract centers
        centers = np.array([d['center'] for d in detections])

        # Y 좌표로 클러스터링
        y_coords = centers[:, 1].reshape(-1, 1)

        if len(y_coords) > 1:
            layer_labels = fclusterdata(
                y_coords,
                t=self.vertical_tolerance,
                criterion='distance',
                method='complete'
            )
        else:
            layer_labels = np.array([1])

        # 층 번호를 위에서 아래로 재정렬
        unique_layers = np.unique(layer_labels)
        layer_avg_y = {layer: np.mean(y_coords[layer_labels == layer]) for layer in unique_layers}
        sorted_layers = sorted(layer_avg_y.items(), key=lambda x: x[1])
        layer_mapping = {old: new for new, (old, _) in enumerate(sorted_layers, 1)}
        remapped_layers = np.array([layer_mapping[label] for label in layer_labels])

        # Add layer info to detections
        for i, detection in enumerate(detections):
            detection['layer_id'] = int(remapped_layers[i])

        return detections

    def get_contour_points(self, contour):
        """
        Contour를 Point 메시지 리스트로 변환

        Args:
            contour: OpenCV contour

        Returns:
            List of Point messages
        """
        contour_points = []
        for point in contour:
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

        # Preprocess
        preprocessed = self.preprocess_image(cv_image)

        # Edge detection
        edges = self.detect_edges(preprocessed)

        # Find Saggars
        detections = self.find_saggars(edges)

        # Cluster into layers
        detections = self.cluster_layers(detections)

        num_detected = len(detections)

        if num_detected == 0:
            rospy.loginfo("No Saggars detected")
            # Publish empty detection
            detection_msg = SaggarArray()
            detection_msg.header = msg.header
            detection_msg.method = "edge_detection"
            detection_msg.total_count = 0
            detection_msg.num_layers = 0
            detection_msg.processing_time = (time.time() - start_time) * 1000
            self.detection_pub.publish(detection_msg)
            return

        # Calculate number of layers
        layer_ids = [d['layer_id'] for d in detections]
        num_layers = len(set(layer_ids))

        # Create detection messages
        detection_array = SaggarArray()
        detection_array.header = msg.header
        detection_array.method = "edge_detection"
        detection_array.total_count = num_detected
        detection_array.num_layers = num_layers

        for detection in detections:
            det_msg = SaggarDetection()
            det_msg.header = msg.header

            # Bounding box
            x, y, w, h = detection['bbox']
            det_msg.x = float(x)
            det_msg.y = float(y)
            det_msg.width = float(w)
            det_msg.height = float(h)

            # Center point
            cx, cy = detection['center']
            det_msg.center.x = float(cx)
            det_msg.center.y = float(cy)
            det_msg.center.z = 0.0

            # Confidence (edge detection doesn't have confidence, use 1.0)
            det_msg.confidence = 1.0

            # Layer info
            det_msg.layer_id = detection['layer_id']
            det_msg.column_id = 0  # Will be calculated later if needed

            # Contour points
            contour_points = self.get_contour_points(detection['contour'])
            det_msg.contour_points = contour_points

            # Create mask from contour
            mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [detection['contour']], -1, 255, -1)

            try:
                mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                det_msg.mask = mask_msg
            except CvBridgeError as e:
                rospy.logwarn(f"Failed to convert mask: {e}")

            detection_array.detections.append(det_msg)

        # Processing time
        processing_time = (time.time() - start_time) * 1000
        detection_array.processing_time = processing_time

        # Publish detections
        self.detection_pub.publish(detection_array)

        rospy.loginfo(f"Detected {num_detected} Saggars in {num_layers} layers "
                     f"({processing_time:.1f}ms)")

        # Visualization
        if self.publish_visualization:
            self.publish_visualization_image(
                cv_image,
                detections,
                edges,
                msg.header
            )

    def publish_visualization_image(self, image, detections, edges, header):
        """
        시각화 이미지 발행
        """
        vis_image = image.copy()

        # 층별 색상 생성
        layer_ids = [d['layer_id'] for d in detections]
        num_layers = len(set(layer_ids))

        colors = []
        for i in range(num_layers):
            hue = int(180 * i / max(num_layers, 1))
            color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            color_rgb = tuple(map(int, [color_bgr[2], color_bgr[1], color_bgr[0]]))
            colors.append(color_rgb)

        # Draw detections
        for detection in detections:
            layer_id = detection['layer_id']
            color = colors[layer_id - 1]

            # Draw contour
            cv2.drawContours(vis_image, [detection['contour']], -1, color, 2)

            # Draw bounding box
            x, y, w, h = detection['bbox']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

            # Draw center
            cx, cy = detection['center']
            cv2.circle(vis_image, (cx, cy), 5, color, -1)

            # Label
            label = f"L{layer_id}"
            cv2.putText(
                vis_image,
                label,
                (x, y - 5),
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
        node = EdgeDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
