#!/usr/bin/env python3
"""
Visualization Node
RViz를 위한 마커 발행 및 추가 시각화 노드
"""

import rospy
import cv2
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from saggar_detection_ros.msg import SaggarArray, SaggarDetection


class VisualizationNode:
    """
    감지 결과를 RViz 마커로 시각화하는 노드
    """

    def __init__(self):
        rospy.init_node('visualization_node', anonymous=False)

        # Parameters
        self.input_topic = rospy.get_param('~input_topic', '/saggar/detections/maskrcnn')
        self.marker_topic = rospy.get_param('~marker_topic', '/saggar/markers')
        self.text_topic = rospy.get_param('~text_topic', '/saggar/text_markers')
        self.contour_topic = rospy.get_param('~contour_topic', '/saggar/contour_markers')
        self.frame_id = rospy.get_param('~frame_id', 'camera_frame')
        self.marker_scale = rospy.get_param('~marker_scale', 0.1)
        self.z_offset = rospy.get_param('~z_offset', 0.05)  # Z offset per layer

        # Publishers
        self.marker_pub = rospy.Publisher(
            self.marker_topic,
            MarkerArray,
            queue_size=10
        )

        self.text_pub = rospy.Publisher(
            self.text_topic,
            MarkerArray,
            queue_size=10
        )

        self.contour_pub = rospy.Publisher(
            self.contour_topic,
            MarkerArray,
            queue_size=10
        )

        # Subscriber
        self.detection_sub = rospy.Subscriber(
            self.input_topic,
            SaggarArray,
            self.detection_callback,
            queue_size=10
        )

        rospy.loginfo("Visualization Node initialized")
        rospy.loginfo(f"Subscribing to: {self.input_topic}")
        rospy.loginfo(f"Publishing markers to: {self.marker_topic}")

    def get_color_for_layer(self, layer_id, num_layers):
        """
        층별 색상 생성

        Args:
            layer_id: Layer number (1-indexed)
            num_layers: Total number of layers

        Returns:
            ColorRGBA message
        """
        hue = int(180 * (layer_id - 1) / max(num_layers, 1))
        color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]

        color = ColorRGBA()
        color.r = float(color_bgr[2]) / 255.0
        color.g = float(color_bgr[1]) / 255.0
        color.b = float(color_bgr[0]) / 255.0
        color.a = 0.8

        return color

    def create_box_marker(self, detection, marker_id, color, frame_id):
        """
        바운딩 박스 마커 생성

        Args:
            detection: SaggarDetection message
            marker_id: Marker ID
            color: ColorRGBA
            frame_id: Frame ID

        Returns:
            Marker message
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bounding_boxes"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Position (center of box)
        marker.pose.position.x = detection.center.x / 1000.0  # Convert to meters
        marker.pose.position.y = detection.center.y / 1000.0
        marker.pose.position.z = detection.layer_id * self.z_offset

        marker.pose.orientation.w = 1.0

        # Scale
        marker.scale.x = detection.width / 1000.0
        marker.scale.y = detection.height / 1000.0
        marker.scale.z = 0.01

        # Color
        marker.color = color

        marker.lifetime = rospy.Duration(1.0)

        return marker

    def create_center_marker(self, detection, marker_id, color, frame_id):
        """
        중심점 마커 생성

        Args:
            detection: SaggarDetection message
            marker_id: Marker ID
            color: ColorRGBA
            frame_id: Frame ID

        Returns:
            Marker message
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "center_points"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position
        marker.pose.position.x = detection.center.x / 1000.0
        marker.pose.position.y = detection.center.y / 1000.0
        marker.pose.position.z = detection.layer_id * self.z_offset

        marker.pose.orientation.w = 1.0

        # Scale
        marker.scale.x = self.marker_scale
        marker.scale.y = self.marker_scale
        marker.scale.z = self.marker_scale

        # Color (brighter)
        bright_color = ColorRGBA()
        bright_color.r = min(color.r * 1.5, 1.0)
        bright_color.g = min(color.g * 1.5, 1.0)
        bright_color.b = min(color.b * 1.5, 1.0)
        bright_color.a = 1.0
        marker.color = bright_color

        marker.lifetime = rospy.Duration(1.0)

        return marker

    def create_text_marker(self, detection, marker_id, frame_id):
        """
        텍스트 마커 생성

        Args:
            detection: SaggarDetection message
            marker_id: Marker ID
            frame_id: Frame ID

        Returns:
            Marker message
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "text_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        # Position (above center)
        marker.pose.position.x = detection.center.x / 1000.0
        marker.pose.position.y = detection.center.y / 1000.0
        marker.pose.position.z = detection.layer_id * self.z_offset + 0.1

        marker.pose.orientation.w = 1.0

        # Text
        marker.text = f"L{detection.layer_id} C{detection.column_id}\n{detection.confidence:.2f}"

        # Scale
        marker.scale.z = 0.05

        # Color (white)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(1.0)

        return marker

    def create_contour_marker(self, detection, marker_id, color, frame_id):
        """
        Contour 라인 마커 생성

        Args:
            detection: SaggarDetection message
            marker_id: Marker ID
            color: ColorRGBA
            frame_id: Frame ID

        Returns:
            Marker message
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "contours"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Points
        z = detection.layer_id * self.z_offset

        for contour_pt in detection.contour_points:
            pt = Point()
            pt.x = contour_pt.x / 1000.0
            pt.y = contour_pt.y / 1000.0
            pt.z = z
            marker.points.append(pt)

        # Close the contour
        if len(detection.contour_points) > 0:
            pt = Point()
            pt.x = detection.contour_points[0].x / 1000.0
            pt.y = detection.contour_points[0].y / 1000.0
            pt.z = z
            marker.points.append(pt)

        # Scale (line width)
        marker.scale.x = 0.005

        # Color
        marker.color = color

        marker.lifetime = rospy.Duration(1.0)

        return marker

    def detection_callback(self, msg):
        """
        감지 결과 콜백
        """
        if msg.total_count == 0:
            # Clear markers
            self.clear_markers()
            return

        # Create marker arrays
        box_markers = MarkerArray()
        text_markers = MarkerArray()
        contour_markers = MarkerArray()

        for i, detection in enumerate(msg.detections):
            # Get color for this layer
            color = self.get_color_for_layer(detection.layer_id, msg.num_layers)

            # Bounding box marker
            box_marker = self.create_box_marker(
                detection,
                i,
                color,
                self.frame_id
            )
            box_markers.markers.append(box_marker)

            # Center point marker
            center_marker = self.create_center_marker(
                detection,
                i,
                color,
                self.frame_id
            )
            box_markers.markers.append(center_marker)

            # Text marker
            text_marker = self.create_text_marker(
                detection,
                i,
                self.frame_id
            )
            text_markers.markers.append(text_marker)

            # Contour marker
            if len(detection.contour_points) > 0:
                contour_marker = self.create_contour_marker(
                    detection,
                    i,
                    color,
                    self.frame_id
                )
                contour_markers.markers.append(contour_marker)

        # Publish markers
        self.marker_pub.publish(box_markers)
        self.text_pub.publish(text_markers)
        self.contour_pub.publish(contour_markers)

        rospy.loginfo(f"Published {len(box_markers.markers)} markers for {msg.total_count} detections")

    def clear_markers(self):
        """
        모든 마커 제거
        """
        # Delete all markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL

        marker_array = MarkerArray()
        marker_array.markers.append(delete_marker)

        self.marker_pub.publish(marker_array)
        self.text_pub.publish(marker_array)
        self.contour_pub.publish(marker_array)

    def run(self):
        """
        노드 실행
        """
        rospy.spin()


def main():
    try:
        node = VisualizationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
