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
            shadow_mask = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except CvBridgeError as e:
            print(e)

        # Apply morphological transformations and thresholding to get rid of noise
        kernel = np.ones((5,5), np.uint8)
        shadow_mask_grey = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)
        ret, shadow_mask_grey = cv2.threshold(shadow_mask_grey, 127, 255, cv2.THRESH_TOZERO)
        shadow_mask_grey = cv2.erode(shadow_mask_grey, kernel, iterations=2)
        shadow_mask_grey = cv2.dilate(shadow_mask_grey, kernel, iterations=1)
        ret, shadow_mask_grey = cv2.threshold(shadow_mask_grey, 230, 255, cv2.THRESH_TOZERO)

        # Draw contour around shadows
        contours, hierarchy = cv2.findContours(shadow_mask_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnt = contours[4]
        # cv.drawContours(shadow_mask, [cnt], 0, (0,255,0), 3)
        cv2.drawContours(shadow_mask, contours, -1, (0,255,0), 3)

        self.mask = shadow_mask

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