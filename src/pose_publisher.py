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
    frame_cv = calibrator.undistort(frame_cv)
    frames_buffer.append((frame_ros.header, frame_cv))


def callback_humans(human_array):
    global frames_buffer

    import time
    t = time.time()

    # Remove all frames previous to the header associated with human_array
    frames_buffer = filter(lambda frame: frame[0].seq >= human_array.header.seq, frames_buffer)
    # The first [0] tuple (seq, frame) of this list is the frame we're looking for
    header, frame = frames_buffer[0]
    
    humans = Humans(human_array, frame, header, K)

    posearray_msg = humans.get_poses()
    pub_pose.publish(posearray_msg)
    
    #frame_humans = humans.draw(draw_position=False, draw_orientation=False)
    #frame_humans = bridge.cv2_to_imgmsg(frame_humans, "bgr8")
    #pub_frame_humans.publish(frame_humans)

    print "Processing time: {:.4f}".format(time.time() - t)

    #cv2.imshow('Window', frame_humans)
    #cv2.waitKey(3)


sub_frame = rospy.Subscriber('usb_cam/image_raw', Image, callback_buffer)
sub_humans = rospy.Subscriber('humans', HumanArray, callback_humans)

from geometry_msgs.msg import PoseArray
pub_pose = rospy.Publisher('ale/poses', PoseArray, queue_size=100)
#pub_frame_humans = rospy.Publisher('ale/frame_humans', Image, queue_size=1)

rospy.spin()

cv2.destroyAllWindows()
