#!/usr/bin/env python3

""" Uses a shadow detection model to create shadow masks """

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import PIL
import torch
from torchvision import transforms
import pydensecrf.densecrf as dcrf
import numpy as np
import time
from config import MTMT_path

class ShadowMask:
    """" ROS node that processes images recieved to detect shadows and create a shadow mask """
    def __init__(self):
        rospy.init_node('ShadowMask')

        # Path to saved model weights
        self.MODEL_PATH = MTMT_path

        self.bridge = CvBridge()

        # Topic that gets the request image from the ShadowApproacher node
        rospy.Subscriber('video_frame', Image, self.process_image)

        self.pub = rospy.Publisher('frame_mask', Image, queue_size=10)

        # Use cpu for inference
        self.device = torch.device('cpu')

    def process_image(self, msg):
        """ 
        Recives images from the video_frame topic, processes it, and publishes the shadow mask

        Args:
            msg: the data from the rostopic
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except CvBridgeError as e:
            print(e)

        rospy.loginfo('Performing inference on shadow detection net')
        mask_image = self.get_mask(cv_image)

        self.pub.publish(self.bridge.cv2_to_imgmsg(mask_image))

        cv2.waitKey(3)

    @torch.no_grad()
    def get_mask(self, image, trans_scale=416):
        """ 
        Perform inference on shadow dection net using cpu 
        
        Args:
            image: openCV image to detect shadow on
            trans_scale: size to scale image to
        """
        # Load trained model and set to evaluation mode
        net = torch.load(self.MODEL_PATH, map_location=self.device)
        net.eval()

        # Process image to correct format before feeding into the net
        # Use ImageNet mean and std to normalize
        normal = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img_transform = transforms.Compose([
            transforms.Resize((trans_scale, trans_scale)),
            transforms.ToTensor(),
            normal,
        ])
        to_pil = transforms.ToPILImage()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w, h, _ = image.shape
        image = PIL.Image.fromarray(image)
        image_trans = img_transform(image).unsqueeze(0).to(device=self.device)
        
        # Run image through the shadow detection model
        up_edge, up_shadow, up_shadow_final = net(image_trans)
        res = torch.sigmoid(up_shadow_final[-1])
        prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
        prediction = self.crf_refine(np.array(image), prediction)

        # Return mask as a np.array to display with openCV
        return cv2.cvtColor(np.array(prediction), cv2.COLOR_RGB2BGR)

    def crf_refine(self, img, annos):
        """ 
        Helper function defined by MTMT model to refine segmentation results using Conditional Random Fields (CRFs)
        
        Taken from https://github.com/eraserNut/MTMT/blob/master/utils/util.py 

        img: the original image
        annos: segmentation results
        """
        assert img.dtype == np.uint8
        assert annos.dtype == np.uint8
        assert img.shape[:2] == annos.shape

        # img and annos should be np array with data type uint8

        EPSILON = 1e-8

        M = 2  # salient or not
        tau = 1.05
        # Setup the CRF model
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

        anno_norm = annos / 255.

        n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * self.sigmoid(1 - anno_norm))
        p_energy = -np.log(anno_norm + EPSILON) / (tau * self.sigmoid(anno_norm))

        U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
        U[0, :] = n_energy.flatten()
        U[1, :] = p_energy.flatten()

        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

        # Do the inference
        infer = np.array(d.inference(1)).astype('float32')
        res = infer[1, :]

        res = res * 255
        res = res.reshape(img.shape[:2])
        return res.astype('uint8')

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def run(self):
        r = rospy.Rate(5)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")

if __name__ == '__main__':
    node = ShadowMask()
    node.run()