#!/usr/bin/env python

import rospy

import cv_bridge
from sensor_msgs.msg import Image
from openpose_pkg.msg import BodyPart, Human, HumanArray

import os
import time
import numpy as np

rospy.init_node('openpose')

bridge = cv_bridge.CvBridge()


from estimator import TfPoseEstimator

### Load model ###
from estimator import TfPoseEstimator

models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'models')
graph_path = os.path.join(models_path, 'graph_opt.pb')

w, h = 432, 368
openpose = TfPoseEstimator(graph_path, target_size=(w, h))

### Load calibrator ###
from helper import Calibrator

calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'calibration')
K = np.load(os.path.join(calib_path, 'K.npy'))
dist = np.load(os.path.join(calib_path, 'dist.npy'))
calibrator = Calibrator(K, dist)


# Should this be in a helper file? -> Depends on w, h
def humans_to_msg(humans):
    humanArray_msg = HumanArray()
    humanArray_msg.image_w = w
    humanArray_msg.image_h = h

    for human in humans:
        human_msg = Human()

        part_scores = []
        for body_part in human.body_parts.values():
            bodyPart_msg = BodyPart()
            bodyPart_msg.idx = body_part.part_idx
            bodyPart_msg.x_percent = body_part.x
            bodyPart_msg.y_percent = body_part.y
            bodyPart_msg.score = body_part.score
            
            part_scores.append(body_part.score)

            human_msg.parts.append(bodyPart_msg)
        
        human_msg.certainty = np.mean(part_scores)/10
        humanArray_msg.humans.append(human_msg)

    return humanArray_msg


def callback(frame_ros):
    frame_cv = bridge.imgmsg_to_cv2(frame_ros, "bgr8")
    frame_cv = calibrator.undistort(frame_cv)
    
    t = time.time()
    humans = openpose.inference(frame_cv, scales=[None])
    print "Inference time: {}".format(time.time() - t)

    humanArray_msg = humans_to_msg(humans)
    humanArray_msg.header = frame_ros.header

    pub.publish(humanArray_msg)


#sub = rospy.Subscriber('frame', Image, callback)
sub = rospy.Subscriber('usb_cam/image_raw', Image, callback)
pub = rospy.Publisher('humans', HumanArray, queue_size=1)
rospy.spin()

