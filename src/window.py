#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2, cv_bridge

import numpy as np
from robot.msg import HumanArray
from helper import Humans, Calibrator

rospy.init_node('window')

bridge = cv_bridge.CvBridge()

frames_buffer = []

topic = 'frame'
#topic = 'usb_cam/image_raw'

if topic == 'usb_cam/image_raw':
    # TODO: set K and dist as ROS parameters.
    K = np.array(rospy.get_param('/usb_cam/calib/K'))
    dist = np.array(rospy.get_param('/usb_cam/calib/dist'))
    # TODO: numpy load is another option, but I don't know how to deal with paths
    #K = np.load('calibration/K.npy')
    #dist = np.load('calibration/dist.npy')
    calibrator = Calibrator(K, dist)


def callback_buffer(frame_ros):
    global frames_buffer
    frame_cv = bridge.imgmsg_to_cv2(frame_ros, "bgr8")
    if topic == 'usb_cam/image_raw':
        frame_cv = calibrator.undistort(frame_cv)
    frames_buffer.append((frame_ros.header.seq, frame_cv))


def callback_humans(human_array):
    global frames_buffer

    import time
    t = time.time()

    # Remove all frames previous to the header associated with human_array
    frames_buffer = filter(lambda frame: frame[0] >= human_array.header.seq, frames_buffer)
    # The first [0] tuple (seq, frame) of this list is the frame we're looking for. Extract the frame [1].
    frame = frames_buffer[0][1]
    
    humans = Humans(human_array, frame)
    frame_humans = humans.draw(draw_position=False, draw_orientation=True)

    print "Processing time: {:.4f}".format(time.time() - t)

    cv2.imshow('Window', frame_humans)
    cv2.waitKey(3)


sub_frame = rospy.Subscriber(topic, Image, callback_buffer)
sub_humans = rospy.Subscriber('humans', HumanArray, callback_humans)

rospy.spin()

cv2.destroyAllWindows()
