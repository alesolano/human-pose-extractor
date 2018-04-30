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
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19
CocoPairsRender = CocoPairs[:-2]
CocoPairsNetwork = [
    (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
    (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
 ]  # = 19

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


class Humans():

    def __init__(self, humans, frame):
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
                    coords = self.parts_coords[human_idx][CocoPart.Nose.value]
                    image_drawn = cv2.putText(image_drawn, "{:.1f} cm".format(distance),
                        (int(coords[0]-100), int(coords[1]-100)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
                except Exception as e:
                    if e.message == 12:
                        print "Neck undetected for human {}".format(human_idx)
            
            if draw_orientation:
                coords = self.parts_coords[human_idx][CocoPart.Nose.value]
                image_drawn = cv2.putText(image_drawn, self.left_or_right(human_idx),
                    (int(coords[0]-60), int(coords[1]-160)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)

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
        
        if CocoPart.REar.value not in detected_parts \
            and CocoPart.LEar.value in detected_parts:
            looking_msg = "right"
        elif CocoPart.REar.value  in detected_parts \
            and CocoPart.LEar.value not in detected_parts:
            looking_msg = "left"
        else:
            looking_msg = "front"

        return looking_msg




class Calibrator():
    
    def __init__(self, K=None, dist=None):
        '''
        Calibration parameters (self.K and self.dist) could be given when constructing the class.
        Otherwise, calculate these parameters using self.calibrate().
        '''
        self.K = K # 3x3 camera matrix: intrinsic parameters
        self.dist = dist # Distortion coefficients (k1, k2, p1, p2)
    
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
        
        self.ret, self.K, self.dist, vecR, vecT = cv2.calibrateCamera(objpoints, imgpoints, im_shape,
                                                                      self.K, self.dist,
                                                                      flags=cv2.CALIB_ZERO_TANGENT_DIST)


    def undistort(self, image):
        '''
        Given a distorted (raw) image from the camera, and with the calibration parameters known, returns
        the undistorted image.
        '''
        assert self.K is not None and self.dist is not None, "Error: calibration parameters not found."
        return cv2.undistort(image, self.K, self.dist, None, self.K)

