#!/usr/bin/env python3
"""
Bag Player Node
.bag 파일에서 이미지를 읽어 토픽으로 발행하는 노드
"""

import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys


class BagPlayerNode:
    """
    ROS bag 파일에서 이미지를 읽어 발행하는 노드
    """

    def __init__(self):
        rospy.init_node('bag_player_node', anonymous=False)

        # Parameters
        self.bag_file = rospy.get_param('~bag_file', '')
        self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.output_topic = rospy.get_param('~output_topic', '/saggar/image_raw')
        self.loop = rospy.get_param('~loop', True)
        self.rate = rospy.get_param('~rate', 10.0)  # Hz
        self.start_time = rospy.get_param('~start_time', 0.0)  # seconds
        self.end_time = rospy.get_param('~end_time', -1.0)  # -1 means until end

        # Publisher
        self.image_pub = rospy.Publisher(
            self.output_topic,
            Image,
            queue_size=10
        )

        self.bridge = CvBridge()

        # Check if bag file is specified
        if not self.bag_file:
            rospy.logerr("No bag file specified! Use ~bag_file parameter")
            sys.exit(1)

        rospy.loginfo(f"Bag Player Node initialized")
        rospy.loginfo(f"Bag file: {self.bag_file}")
        rospy.loginfo(f"Reading from topic: {self.image_topic}")
        rospy.loginfo(f"Publishing to: {self.output_topic}")
        rospy.loginfo(f"Rate: {self.rate} Hz")
        rospy.loginfo(f"Loop: {self.loop}")

    def play_bag(self):
        """
        Bag 파일을 재생하여 이미지 발행
        """
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            try:
                with rosbag.Bag(self.bag_file, 'r') as bag:
                    # Get bag info
                    info = bag.get_type_and_topic_info()
                    topics = info.topics

                    if self.image_topic not in topics:
                        rospy.logerr(f"Topic {self.image_topic} not found in bag file!")
                        rospy.loginfo(f"Available topics: {list(topics.keys())}")
                        return

                    # Get start and end times
                    bag_start_time = bag.get_start_time()
                    bag_end_time = bag.get_end_time()

                    start_t = bag_start_time + self.start_time
                    end_t = bag_end_time if self.end_time < 0 else bag_start_time + self.end_time

                    rospy.loginfo(f"Playing bag from {start_t - bag_start_time:.2f}s to {end_t - bag_start_time:.2f}s")

                    # Read messages
                    for topic, msg, t in bag.read_messages(
                        topics=[self.image_topic],
                        start_time=rospy.Time.from_sec(start_t),
                        end_time=rospy.Time.from_sec(end_t)
                    ):
                        if rospy.is_shutdown():
                            break

                        # Update timestamp to current time
                        msg.header.stamp = rospy.Time.now()

                        # Publish image
                        self.image_pub.publish(msg)

                        rate.sleep()

                    rospy.loginfo("Finished playing bag file")

                    if not self.loop:
                        break

                    rospy.loginfo("Looping bag file...")

            except Exception as e:
                rospy.logerr(f"Error reading bag file: {e}")
                break

        rospy.loginfo("Bag player node shutting down")

    def run(self):
        """
        노드 실행
        """
        self.play_bag()


def main():
    try:
        node = BagPlayerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
