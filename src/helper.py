#!/usr/bin/env python

from __future__ import division # division returns a floating point number
import os

import numpy as np
import cv2

from enum import Enum

class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

class CocoPair(Enum):
    LShoulder = 0
    RShoulder = 1
    RArm = 2
    RForearm = 3
    LArm = 4
    LForearm = 5
    RBody = 6
    RThigh = 7
    RCalf = 8
    LBody = 9
    LThigh = 10
    LCalf = 11
    Neck = 12
    RNoseEye = 13
    REyeEar = 14
    LNoseEye = 15
    LEyeEar = 16
    RShoulderEar = 17
    LShoulderEar = 18

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), #10
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17) # 19
]   # = 19
CocoPairsRender = CocoPairs[:-2]
CocoPairsNetwork = [
    (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5), #9
    (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
 ]  # = 19

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


class Humans():

    def __init__(self, humans, frame, header, K=None):
        self.humans = humans.humans # List of Human objects
        self.n_humans = len(self.humans) # Number of detected humans
        
        # Coordinates of all parts detected, ordered by humans.
        # List of humans. Each human is a dictionary of parts. Each part is a tuple of x and y coordinates.
        self.parts_coords = []
        
        # Vector components (magnitude and direction) of all pairs detected, ordered by humans
        # List of humans. Each human is a dictionary of pairs. Each pair is a tuple of magnitude and direction
        self.pairs_components = []
        
        self.image = frame
        self.image_h = frame.shape[0]
        self.image_w = frame.shape[1]

        self.header = header

        self.K = K # Intrinsic matrix of the camera
        
        if self.n_humans != 0:
            self.fill_pairs_components()
        

    def fill_pairs_components(self):
        
        for human_idx, human in enumerate(self.humans):
            
            # Append a dictionary for each human detected
            self.parts_coords.append({})
            self.pairs_components.append({})
            
            for part in human.parts:
                x = int(part.x_percent * self.image_w + 0.5)
                y = int(part.y_percent * self.image_h + 0.5)
                self.parts_coords[human_idx][part.idx] = (x, y)

            for pair_idx, pair in enumerate(CocoPairs):
                # if any of the parts that form the pair has not been detected, continue
                if pair[0] not in self.parts_coords[human_idx].keys() \
                    or pair[1] not in self.parts_coords[human_idx].keys():
                    continue

                x_0, y_0 = self.parts_coords[human_idx][pair[0]]
                x_1, y_1 = self.parts_coords[human_idx][pair[1]]

                magnitude = np.linalg.norm([x_1 - x_0, y_1 - y_0])
                direction = np.arctan2(y_1 - y_0, x_1 - x_0)
                self.pairs_components[human_idx][pair_idx] = (magnitude, direction)
                
    
    def draw(self, draw_position=False, draw_orientation=False):
        image_drawn = np.copy(self.image)
        
        centers = {}
        for human_idx, human in enumerate(self.humans):
            
            # draw point
            for part_idx in range(len(CocoPart)):
                # if the part has not been detected, continue
                if part_idx not in self.parts_coords[human_idx].keys():
                    continue

                center = self.parts_coords[human_idx][part_idx]
                centers[part_idx] = center
                cv2.circle(image_drawn, center, 3, CocoColors[part_idx], thickness=3, lineType=8, shift=0)

            # draw line
            for pair_idx, pair in enumerate(CocoPairsRender):
                # if the pair has not been detected, continue
                if pair_idx not in self.pairs_components[human_idx].keys(): 
                    continue

                image_drawn = cv2.line(image_drawn, centers[pair[0]], centers[pair[1]],
                                       CocoColors[pair_idx], 3)
                
            if draw_position:
                try:
                    distance = self.get_distance_to_camera(human_idx)
                    #distance = self.get_position(human_idx)[2]
                    coords = self.parts_coords[human_idx][CocoPart.Nose.value]
                    image_drawn = cv2.putText(image_drawn, "{:.1f} cm".format(distance),
                        (int(coords[0]-100), int(coords[1]-100)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
                except Exception as e:
                    if e.message == 12:
                        print "Neck undetected for human {}".format(human_idx)
            
            if draw_orientation:
                #coords = self.parts_coords[human_idx][CocoPart.Nose.value]
                #image_drawn = cv2.putText(image_drawn, self.left_or_right(human_idx),
                #    (int(coords[0]-60), int(coords[1]-160)),
                #    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
                try:
                    body_angle = self.get_body_angle(human_idx)
                    head_angle = self.get_head_angle(human_idx)
                    coords = self.parts_coords[human_idx][CocoPart.Nose.value]
                    image_drawn = cv2.putText(image_drawn, "Body: {:.2f}".format(body_angle),
                        (int(coords[0]-120), int(coords[1]-100)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
                    image_drawn = cv2.putText(image_drawn, "Head: {:.2f}".format(head_angle),
                        (int(coords[0]-120), int(coords[1]-60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
                except Exception as e:
                    print e

        return image_drawn
    
    
    def get_distance_to_camera(self, human_idx):
        '''
        distance = k * 1./neck_magnitude_pixels
        Where k is a constant estimated from regression data.
        returns:
            distance in cm
        '''
        k = 14000 # TODO: Get the real 'k' with regression.
        neck_magnitude_pixels = self.pairs_components[human_idx][CocoPair.Neck.value][0]
        
        distance = k * 1./neck_magnitude_pixels
        return distance
    
    
    def left_or_right(self, human_idx):
        detected_parts = self.parts_coords[human_idx].keys()

        REye = CocoPart.REye.value in detected_parts
        LEye = CocoPart.LEye.value in detected_parts
        REar = CocoPart.REar.value in detected_parts
        LEar = CocoPart.LEar.value in detected_parts

        if not REye:
            angle = 80.
        elif not LEye:
            angle = -80.
        elif REye and not REar:
            angle = 30
        elif LEye and not LEar:
            angle = -30
        else:
            angle = 0.

        return angle


    def get_position(self, human_idx):
        '''
        Get the 3D position of a given human's nose,
        transforming a point 'm' from pixel coordinate system [u, v]
        to a point 'M' in camera 3D coordinate system [X, Y, Z].
        '''

        assert self.K is not None, "Error: intrinsic matrix K not defined"

        u, v = self.parts_coords[human_idx][CocoPart.Nose.value]
        m = np.array([u, v, 1]) # Homogeneous coordinates of the nose

        # Extracting camera parameters
        x0 = self.K[0, 2]
        y0 = self.K[1, 2]
        f = self.K[0, 0]

        # Matrix from 'm' to 'M/Z'.
        K_inv = np.array([[0, 1/f, -x0/f],
                        [-1/f, 0, y0/f],
                        [0, 0, 1]])

        # Altenative way of obtaining K_inv
        #u, v = v, u
        #K_inv = np.linalg.inv(K)
        #K_inv[1] *= -1

        Z = self.get_distance_to_camera(human_idx)

        M = Z*K_inv.dot(m)

        return M


    def get_head_angle(self, human_idx):
        # TODO: change angle estimation procedure depending on the distance.
        # If the person is far, shortest links (those located in the head) have very low resolution.
        # If the person is close, longest links are not

        # TODO: get real value of neck_k
        neck_k = 2
        neck_angle = self.pairs_components[human_idx][CocoPair.Neck.value][1]
        neck_angle = -neck_k*np.rad2deg(neck_angle + np.pi/2)
        #print "Neck angle: {}".format(neck_angle)

        bins = np.array([-120, -90, -60, -30, 0, 30, 60, 90, 120])
        neck_angle = bins[np.digitize(neck_angle - 15, bins)]
        return neck_angle


    def get_body_angle(self, human_idx):
        # TODO: change angle estimation procedure depending on the distance.
        # If the person is far, shortest links (those located in the head) have very low resolution.
        # If the person is close, longest links are not

        x_l, y_l = self.parts_coords[human_idx][CocoPart.LShoulder.value]
        x_r, y_r = self.parts_coords[human_idx][CocoPart.RShoulder.value]
        back_mag = np.linalg.norm([x_l - x_r, y_l - y_r])
        neck_mag = self.pairs_components[human_idx][CocoPair.Neck.value][0]

        # TODO: get real value of back_k
        back_k = 1.25
        neck_angle = self.get_head_angle(human_idx)
        back_angle = -np.sign(neck_angle+0.1)*np.rad2deg(back_mag/neck_mag - back_k*np.pi/2)

        bins = np.array([-120, -90, -60, -30, 0, 30, 60, 90, 120])
        back_angle = bins[np.digitize(back_angle - 15, bins)]

        return back_angle


    def get_head_orientation(self, human_idx):
        # Orientation of the human head with respecto to the camera
        H = np.array([0, np.deg2rad(self.get_head_angle(human_idx)), 0])
        return H

    def get_body_orientation(self, human_idx):
        # Orientation of the human body with respecto to the camera
        B = np.array([0, np.deg2rad(self.get_body_angle(human_idx)), 0])
        return B


    def get_poses(self):
        from tf.transformations import quaternion_from_euler
        from geometry_msgs.msg import Pose, PoseArray
        posearray_msg = PoseArray()
        posearray_msg.header = self.header
        for human_idx, human in enumerate(self.humans):
            pose_msg = Pose()

            try:
                P = self.get_position(human_idx)
            except:
                print "Neck undetected for human {}".format(human_idx)
                continue

            #B = self.get_body_orientation(human_idx)
            #QB = quaternion_from_euler(*B)

            H = self.get_head_orientation(human_idx)
            QH = quaternion_from_euler(*H)

            pose_msg.position.x = P[0]
            pose_msg.position.y = P[1]
            pose_msg.position.z = P[2]
            pose_msg.orientation.x = QH[0]
            pose_msg.orientation.y = QH[1]
            pose_msg.orientation.z = QH[2]
            pose_msg.orientation.w = QH[3]

            posearray_msg.poses.append(pose_msg)

        return posearray_msg




class Calibrator():
    
    def __init__(self, K=None, dist=None, error=None):
        '''
        Calibration parameters (self.K and self.dist) could be given when constructing the class.
        Otherwise, calculate these parameters using self.calibrate().
        '''
        self.K = K # 3x3 camera matrix: intrinsic parameters
        self.dist = dist # Distortion coefficients (k1, k2, p1, p2)
        self.error = error # RMS reprojection error of the calibration
    
    @staticmethod
    def load_images(filenames):
        '''Given a list of image filenames, return a list of images in numpy array format.'''
        return [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) for image in filenames]
    

    @staticmethod
    def get_chessboard_corners(images, board_size):
        '''
        Given a list of chessboard pictures and its size, returns the location of the detected inner corners.
        Args:
            images: list of RGB chessboard pictures in numpy array format
            board size: tuple (number of inner corners of width, number of inner corners of height).
                For example a chessboard of 10 squares of width and 7 squares of height has a size of (9, 6)
        '''
        ret_and_corners = [cv2.findChessboardCorners(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), board_size, None) 
                           for image in images]
        valid_corners = [corner for ret, corner in ret_and_corners if ret]
        return valid_corners
    

    @staticmethod
    def get_chessboard_points(board_size, dx=1, dy=1):
        '''
        Given the size of the chessboard and the distances between the corners, returns a 3D grid (z = 0)
        of unprojected corners.
        Args:
            board size: tuple (number of inner corners of width, number of inner corners of height).
                For example a chessboard of 10 squares of width and 7 squares of height has a size of (9, 6)
            dx: real distance between corners in the x direction (inches)
            dy: real distance betwen corners in the y direction (inches)
        '''
        X, Y = np.mgrid[0:board_size[1]*dx:dx, 0:board_size[0]*dy:dy]
        points = np.array([X.flatten(), Y.flatten(), np.zeros(len(X.flatten()))]).T
        return points
    

    def calibrate(self, filenames, board_size, dx=1, dy=1):
        '''
        Given a list of filenames and the chessboard size, calculate the calibration parameters
        (self.K and self.dist)
        '''
        images = self.load_images(filenames)
        
        valid_corners = self.get_chessboard_corners(images, board_size)
        num_valid_images = len(valid_corners)
        
        points = self.get_chessboard_points(board_size, dx, dy)
        
        # 3D points (z=0) with origin in the upper left corner
        objpoints = np.array([points] * num_valid_images, dtype=np.float32)
        # 
        imgpoints = np.array([corner[:,0,:] for corner in valid_corners], dtype=np.float32)
        im_shape = images[0].shape[:2]
        
        self.error, self.K, self.dist, vecR, vecT = cv2.calibrateCamera(objpoints, imgpoints, im_shape,
                                                                      self.K, self.dist,
                                                                      flags=cv2.CALIB_ZERO_TANGENT_DIST)


    def save_paramaters(self, directory='', K_file='K', dist_file='dist', error_file='error'):
        assert self.K is not None and self.dist is not None and self.error is not None, "Error: calibration parameters not found."
        np.save(os.path.join(directory, K_file), self.K)
        np.save(os.path.join(directory, dist_file), self.dist)
        np.save(os.path.join(directory, error_file), self.error)


    def undistort(self, image):
        '''
        Given a distorted (raw) image from the camera, and with the calibration parameters known, returns
        the undistorted image.
        '''
        assert self.K is not None and self.dist is not None, "Error: calibration parameters not found."
        return cv2.undistort(image, self.K, self.dist, None, self.K)

