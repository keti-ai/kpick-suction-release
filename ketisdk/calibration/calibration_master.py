import numpy as np
import cv2
import math

class Calibration():
    def __init__(self):
        self.homo2d = []
        self.robot_pose = np.eye(4)
        self.robot_pose[0][3] = 535.0
        self.robot_pose[1][3] = -111.00
        self.robot_pose[2][3] = 640.00

        self.tcp_to_cam = np.eye(4)
        #self.tcp_to_cam[0][3] = 120.0
        self.tcp_to_cam[0][3] = 140.0
        self.tcp_to_cam[1][3] = 30.00
        self.tcp_to_cam[2][3] = -40.00

        self.cam_to_pick = np.eye(4)
        self.tcp_to_tool = np.eye(4)
        self.tcp_to_tool[0][3] = 0.0
        self.tcp_to_tool[1][3] = 0.0
        self.tcp_to_tool[2][3] = 235.0

        self.ppx = 642.252
        self.ppy = 359.912
        self.focallength = 922.0

        self.set2dhomography()


    def set2dhomography(self, robot_points = [], cam_points = []):
        print('cal::set 2dHomograyphy')
        shape = (1, 4, 2)
        robot_points = np.zeros(shape, np.float32)
        cam_points   = np.zeros(shape, np.float32)
        '''
        /doosan robot
        robot_points[0][0] = [236.850, 948.550]
        robot_points[0][1] = [226.220, 493.980]
        robot_points[0][2] = [-345.050, 516.790]
        robot_points[0][3] = [-332.240, 966.260]

        # [X, Y, Z] robot pose
        cam_points[0][0] = [341, 544]
        cam_points[0][1] = [338, 21]
        cam_points[0][2] = [998, 25]
        cam_points[0][3] = [999, 550]
        '''
        robot_points[0][0] = [-251.92, -556.11]
        robot_points[0][1] = [-251.74, -796.09]
        robot_points[0][2] = [149.62, -799.10]
        robot_points[0][3] = [149.63, -561.09]

        # [X, Y, Z] robot pose
        cam_points[0][0] = [433, 74]
        cam_points[0][1] = [428, 352]
        cam_points[0][2] = [900, 354]
        cam_points[0][3] = [903, 77]

        self.homo2d, status = cv2.findHomography(cam_points, robot_points)
        print(self.homo2d)

    def getXYZ_homo(self, x, y, depth):
        cam_default_height = 500

        # using 2d homography
        rx = (self.homo2d[0][0] * x) + (self.homo2d[0][1] * y) + self.homo2d[0][2]
        ry = (self.homo2d[1][0] * x) + (self.homo2d[1][1] * y) + self.homo2d[1][2]
        rz = cam_default_height - depth
        rz = 130

        print('rx,ry,rz::', rx, ry, rz)
        return rx,ry,rz


    def getXYZ(self, x, y, depth):
        pz = -depth
        px = float((float(x) - self.ppx)) / self.focallength * float(pz)
        py = float((float(y) - self.ppy)) / self.focallength * float(pz)

        self.cam_to_pick[0][3] = py
        self.cam_to_pick[1][3] = px
        self.cam_to_pick[2][3] = pz

        #robot_xyz = self.robot_pose.dot(self.tcp_to_cam).dot(self.cam_to_pick).dot(self.tcp_to_tool)

        rx = self.robot_pose[0][3] + self.cam_to_pick[0][3] + self.tcp_to_cam[0][3] - self.tcp_to_tool[2][3] * math.cos(math.radians(60))
        ry = self.robot_pose[1][3] + self.cam_to_pick[1][3] + self.tcp_to_cam[1][3]
        rz = self.robot_pose[2][3] + self.cam_to_pick[2][3] + self.tcp_to_cam[2][3] + self.tcp_to_tool[2][3] * math.sin(math.radians(60))

        '''            
        rx = robot_xyz[0][3]
        ry = robot_xyz[1][3]
        rz = robot_xyz[2][3]
        '''

        print('rx,ry,rz::', rx, ry, rz)
        return rx,ry,rz

    def set_detect_robot_pose(self, pose):
        self.robot_pose[0][3] = pose[0]
        self.robot_pose[1][3] = pose[1]
        self.robot_pose[2][3] = pose[2]

    def set_cam_to_tcp(self, pose):
        self.cam_to_tcp[0][3] = pose[0]
        self.cam_to_tcp[1][3] = pose[1]
        self.cam_to_tcp[2][3] = pose[2]

    def set_tool_pose(self, pose):
        self.tool_pos[0][3] = pose[0]
        self.tool_pos[1][3] = pose[1]
        self.tool_pos[2][3] = pose[2]

if __name__=='__main__':
    cal = Calibration()
    aa = cal.getXYZ(100,100,100)

    print(aa)

