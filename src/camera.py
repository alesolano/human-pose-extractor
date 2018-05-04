#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2, cv_bridge


rospy.init_node('camera')

pub = rospy.Publisher('frame', Image, queue_size=1)
#pub = rospy.Publisher('usb_cam/image_raw', Image, queue_size=1)

bridge = cv_bridge.CvBridge()

cap = cv2.VideoCapture(0)

while not rospy.is_shutdown():
    ret, frame_cv = cap.read()
    if ret:
        #frame_cv = cv2.resize(frame_cv, (300, 300), interpolation=cv2.INTER_CUBIC)
        frame_ros = bridge.cv2_to_imgmsg(frame_cv, "bgr8")
        pub.publish(frame_ros)
    
cap.release()
