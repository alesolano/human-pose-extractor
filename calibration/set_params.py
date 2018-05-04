import numpy as np
import rospy

K = np.load('K.npy')
dist = np.load('dist.npy')
#error = np.load('error.npy')

rospy.set_param('/usb_cam/calib/K', K.tolist())
rospy.set_param('/usb_cam/calib/dist', dist.tolist())
#rospy.set_param('/usb_cam/calib/error', error)
