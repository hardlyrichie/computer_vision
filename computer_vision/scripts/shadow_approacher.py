#!/usr/bin/env python3

""" Neato robot that uses computer vision to look for and hide in shadows. """

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time
from PIL import Image as Im
import random

class ShadowApproacher:
    """" ROS node that functions as the controller, moving the robot to the nearest shadow """
    def __init__(self):
        rospy.init_node('ShadowApproacher')
        
        self.bridge = CvBridge()

        rospy.Subscriber('/camera/image_raw', Image, self.process_raw_image) # topic that gets raw image from camera
        rospy.Subscriber('frame_mask', Image, self.process_mask) # topic that gets shadow mask image
        self.video_pub = rospy.Publisher('video_frame', Image, queue_size=10) # publish to topic the camera image
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.last_shadow_time = rospy.Time.now()

        # Holds the most recent shadow mask
        self.mask = None

        self.width, self.height = 600, 600

        # Proportional control scaling factor
        self.k = 0.3

        # Flags
        self.within_shadow = False # Checks if neato is currently within a shadow
        self.seeking = True # Seeking mode: looks for nearest shadow ignoring neato area visible to the camera
        self.e_stop = False 
        self.helping_push = True # Gives neato an extra push once in a shadow

    def process_raw_image(self, msg):
        """ 
        Receives images from the /camera/image_raw topic and processes it into an openCV image

        Args:
            msg: the data from the rostopic
        """
        # Convert ros image to opencv image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except CvBridgeError as e:
            print(e)

        # Display camera image and shadow mask
        self.width, self.height, channels = cv_image.shape
        cv2.imshow('Neato Camera', cv_image)
        if self.mask is not None:
            cv2.imshow('Shadow Mask', self.mask)
            # Setup mouse event to start seeking mode upon mouse click
            cv2.setMouseCallback('Shadow Mask', self.process_mouse_event)

        # Send image to ShadowMask node every 2 seconds
        current_time = rospy.Time.now()
        if (current_time - self.last_shadow_time).to_sec() > 2:
            mask_image = self.video_pub.publish(self.bridge.cv2_to_imgmsg(cv_image))
            self.last_shadow_time = current_time

        key = cv2.waitKey(3)

        # Setup e-stop key event
        if key == ord('s'):
            self.toggle_e_stop()

    def process_mask(self, msg):
        """ 
        Receives images from the frame_mask topic and ShadowMask node and performs 
        image processing on it to create a mask and isolate a shadow

        Args:
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
        # shadow_mask_grey = cv2.erode(shadow_mask_grey, kernel, iterations=1)
        shadow_mask_grey = cv2.dilate(shadow_mask_grey, kernel, iterations=1)
        ret, shadow_mask_grey = cv2.threshold(shadow_mask_grey, 230, 255, cv2.THRESH_TOZERO)

        # Draw contour around shadows
        contours, hierarchy = cv2.findContours(shadow_mask_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(shadow_mask, contours, -1, (0,255,0), 3)

        # Neato bounding box, this defines the "neato area" (the part of the neato visible in the camera)
        neato_box = self.create_bbox(93, 567, 417, 33)
        cv2.rectangle(shadow_mask, (neato_box['x'], neato_box['y']), 
                    (neato_box['x'] + neato_box['w'], neato_box['y'] + neato_box['h']), (255, 0, 0), 1)

        # Horizon bounding box
        horizon_box = self.create_bbox(0, 0, self.width, self.height/2)

        # Find COM for all contours and choose the optimal shadow 
        # (largest weighted sum of closeness to neato & area of shadow)
        optimal_shadow = None
        highest_score = 0
        shadow_box = None # Bounding box of the optimal shadow
        shadows_in_neato = 0 # Counter that holds the number of shadows in neato area
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue

            # COM
            cx, cy = self.center_of_mass(M)
            cv2.circle(shadow_mask, (cx, cy), 5, (0, 0, 255), 1, cv2.LINE_AA)

            # Find bounding box for current contour
            contour_box = self.create_bbox(*cv2.boundingRect(contour)) 

            # Most optimal shadow to move toward
            closeness_weight = 0.6
            weighted_sum = (cy / self.height) * closeness_weight + \
                            (cv2.contourArea(contour) / (self.width * self.height / 2)) * (1 - closeness_weight)            

            # Check if current contour and the area that represents the neato in the camera intersect
            intersection = self.check_intersection(contour_box, neato_box)
            if intersection:
                shadows_in_neato += 1

            # Ignore shadows that are in the neato bounding box area if in seeking mode
            # This allows the neato to find other shadows outside the current one it is in
            if self.seeking and intersection:
                continue

            # Ignore shadows above the horizon line
            if self.check_intersection(contour_box, horizon_box):
                continue

            # Keep most optimal shadow
            if weighted_sum > highest_score:
                optimal_shadow = contour
                highest_score = weighted_sum
                shadow_box = contour_box

        # Turn off seeking flag if there are no shadows in neato bounding box
        if shadows_in_neato == 0:
            self.seeking = False

        # Optimal shadow COM
        cv2.circle(shadow_mask, self.center_of_mass(cv2.moments(optimal_shadow)), 8, (0, 0, 255), -1)
        
        if optimal_shadow is None:
            # If there are no shadows, start hunting mode
            self.hunting()
        else:
            # Draw optimal shadow bounding box
            cv2.rectangle(shadow_mask, (shadow_box['x'], shadow_box['y']), 
                        (shadow_box['x'] + shadow_box['w'], shadow_box['y'] + shadow_box['h']), (0, 0, 255), 2)

            # Check if neato is within the shadow (front of neato will be shaded) 
            self.within_shadow = self.check_intersection(shadow_box, neato_box)
            # Give neato back helping push ability if not in shadow
            if not self.within_shadow:
                self.helping_push = True

            # Move neato if there is an optimal shadow
            self.move_to_shadow(optimal_shadow)

        # Save the shadow mask after processing and visualizations
        self.mask = shadow_mask

    def center_of_mass(self, moment):
        """ 
        Calculate the center of mass given an image moment 
        
        Args:
            moment: image moment returned by cv2.moments(countour)
        """
        if moment['m00'] == 0:
            return 0, 0
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy

    def create_bbox(self, x, y, w, h):
        """
        Creates bounding box dictionary

        Args:
            x: x coord of top left point of box
            y: y coord of top left point of box
            w: width of box
            h: height of box

        Returns:
            A bounding box dictionary
        """
        return {
            'x': x, 
            'y': y, 
            'w': w,
            'h': h, 
        }

    def check_intersection(self, box1, box2):
        """ 
        Checks if an two bounding boxes intersect 
        
        Args:
            box1: a bounding box dictionary
            box2: another bounding box dictionary

        Returns:
            Boolean value for intersection
        """
        # Check if boxes intersect in the x direction
        x_overlap_point, overlap_width = self.find_range_overlap(box1['x'], box1['w'],
                                                            box2['x'], box2['w'])
        # Check if boxes intersect in the y direction                                                     
        y_overlap_point, overlap_height = self.find_range_overlap(box1['y'], box1['h'],
                                                            box2['y'], box2['h'])

        # Check if there is a significantly large enough intersection
        return overlap_width and overlap_height and overlap_width * overlap_height > 50


    def find_range_overlap(self, point1, length1, point2, length2):
        """ 
        Checks for intersection in 1D space 

        Args:
            point1: starting point of a line
            length1: length of a line
            point2: starting point of another line
            length2: length of the other line
        
        Returns:
            Boolean value for intersection
        """
        # Find the highest start point and lowest end point.
        # The highest start point is the start point of the overlap.
        # The lowest end point is the end point of the overlap.
        highest_start_point = max(point1, point2)
        lowest_end_point = min(point1 + length1, point2 + length2)

        # Return null overlap if there is no overlap
        if highest_start_point >= lowest_end_point:
            return (None, None)

        # Compute the overlap length
        overlap_length = lowest_end_point - highest_start_point

        return (highest_start_point, overlap_length)

    def move_to_shadow(self, shadow):
        """
        Send motor commands to neato robot to move to nearest shadow

        Args:
            shadow: contour of shadow area
        """
        m = Twist()
        com = self.center_of_mass(cv2.moments(shadow))

        # Rotate amount based on how close COM is to center of image horizontally, range from -0.5 to -0.5
        rotate = ((self.width / 2) - com[0]) / self.width 

        if self.e_stop:
            rospy.loginfo('Stopping neato')
            m.linear = Vector3(0,0,0)
            m.angular = Vector3(0,0,0) 
        elif abs(rotate) > .1:
            # Use proportional control to rotate neato
            rospy.loginfo('Rotating neato')
            m.angular.z = self.k * rotate
        elif self.within_shadow and not self.seeking:
            # Give neato a helpful push the first time it enters the shadow so that the neato is further inside
            if self.helping_push:
                rospy.loginfo('Give neato a helpful push')
                m.linear.x = .15
                m.angular = Vector3(0,0,0) 
                self.helping_push = False
            else:
                # Stop moving if neato is in a shadow
                rospy.loginfo('Neato is within a shadow, stop moving')
                m.linear = Vector3(0,0,0)
                m.angular = Vector3(0,0,0) 
        else:
            # Move forward if COM is around the center of image horizontally
            m.angular.z = 0
            m.linear.x = .05

        self.vel_pub.publish(m)

    def hunting(self):
        """ Hunting behavior. When there is no shadows, spin and move randomly to look for nearest shadow """
        m = Twist()

        if self.e_stop:
            rospy.loginfo('Stopping neato')
            m.linear = Vector3(0,0,0)
            m.angular = Vector3(0,0,0) 
        else:
            rospy.loginfo('Hunting behavior')
            # Turn right or not randomly to a 6:4
            m.angular.z = random.choice([-.1] * 6 + [0] * 4)
            # Move forward or not randomly to a 6:4
            m.linear.x = random.choice([.05] * 6 + [0] * 4)

        self.vel_pub.publish(m)

    def process_mouse_event(self, event, x, y, flags, param):
        """ Callback that handles mouse events """
        # Activates seeking mode. Tells robot to look for and move to the next shadow
        if event == cv2.EVENT_LBUTTONDOWN:
            rospy.loginfo('Seeking mode on')
            self.seeking = True

    def toggle_e_stop(self):
        rospy.loginfo('Toggle E-Stop')
        self.e_stop = not self.e_stop

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