#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import cv2

class NeatoCam:
    def __init__(self):
        rospy.init_node('NeatoCam')

        self.bridge = CvBridge()
        rospy.Subscriber('/camera/image_raw', Image, self.process_image)

    def process_image(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except CvBridgeError as e:
            print(e)

        rows, cols, channels = cv_image.shape
        cv2.imshow("Neato Camera", cv_image)
        cv2.waitKey(3)

    def run(self):
        r = rospy.Rate(5)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")
        cv2.destroyAllWindows()
         
if __name__ == '__main__':
    node = NeatoCam()
    node.run()