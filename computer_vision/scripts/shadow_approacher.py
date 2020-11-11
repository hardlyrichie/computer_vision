#!/usr/bin/env python3

""" Neato robot that uses computer vision to look for and hide in shadows """

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time
from PIL import Image as Im

class ShadowApproacher:
    """" Class that provides the functionality to find and move toward shadows """
    def __init__(self):
        rospy.init_node('ShadowApproacher')
        
        self.bridge = CvBridge()

        rospy.Subscriber('/camera/image_raw', Image, self.process_raw_image) # topic that gets raw image from camera
        rospy.Subscriber('frame_mask', Image, self.process_mask) # topic that gets shadow mask image
        self.pub = rospy.Publisher('video_frame', Image, queue_size=10) # publish to topic the camera image

        self.last_shadow_time = rospy.Time.now()

        self.mask = None

    def process_raw_image(self, msg):
        """ Recives images from the /camera/image_raw topic and processes it into
            an openCV image.

            msg: the data from the rostopic
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except CvBridgeError as e:
            print(e)

        # (600, 600, 3)
        rows, cols, channels = cv_image.shape
        cv2.imshow('Neato Camera', cv_image)
        if self.mask is not None:
            cv2.imshow('Shadow mask', self.mask)

        # Send image to ShadowMask node every 2 seconds
        current_time = rospy.Time.now()
        if (current_time - self.last_shadow_time).to_sec() > 2:
            mask_image = self.pub.publish(self.bridge.cv2_to_imgmsg(cv_image))
            self.last_shadow_time = current_time

        cv2.waitKey(3)

    def process_mask(self, msg):
        """ Recives images from the /camera/image_raw topic and processes it into
            an openCV image.

            msg: the data from the rostopic
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except CvBridgeError as e:
            print(e)

        self.mask = cv_image

    def run(self):
        r = rospy.Rate(5)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    node = ShadowApproacher()
    node.run()