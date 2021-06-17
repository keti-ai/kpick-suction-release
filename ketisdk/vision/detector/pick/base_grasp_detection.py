import numpy as np
from ketisdk.import_basic_utils import *

from torchvision.transforms import transforms
from ketisdk.vision.detector.classifier.roi_classifier import RoiCifarClassfier
import cv2
import os

class BaseGraspDetector(RoiCifarClassfier):

    def load_params(self, args):
        super().load_params(args=args)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(self.args.net.db_mean, self.args.net.db_std), ])
        if hasattr(self.args.net,'bg_depth_file'):
            assert os.path.exists(self.args.net.bg_depth_file)
            self.bg_depth = cv2.blur(cv2.imread(self.args.net.bg_depth_file, cv2.IMREAD_UNCHANGED),
                                     ksize=self.args.net.depth_blur_ksize).astype('float')
            print('backgpund file loaded...')

    def get_background_depth_map(self, rgbd):
        self.bg_depth = cv2.blur(rgbd.depth, ksize=self.args.net.depth_blur_ksize).astype('float')
        print('Background depth stored ...')

    def get_workspace_corner_depth(self, rgbd):
        # pts_array = np.array(rgbd.workspace.pts).reshape((-1,2))
        rx, ry = self.args.net.depth_blur_ksize[0]//2, self.args.net.depth_blur_ksize[1]//2
        dX, dY = np.arange(-rx,rx+1), np.arange(-ry, ry+1)

        corner_depth_map = []
        for pt in rgbd.workspace.pts:
            X, Y = pt[0]+dX, pt[1]+dY
            X, Y = X[(0<=X)&(X<rgbd.width)], Y[(0<=Y)&(Y<rgbd.height)]
            Y,X = np.meshgrid(Y,X)
            X,Y = X.flatten(), Y.flatten()
            corner_depth_map.append(pt + (np.mean(rgbd.depth[(Y,X)]),))
        corner_depth_map = np.array(corner_depth_map)
        print('Corner depth map stored ...')
        return corner_depth_map

    def get_fg_depth_inds(self, rgbd, locs):
        # zlocs = list(range(len(locs[0])))
        if hasattr(self, 'bg_depth'):
            depth_blur = cv2.blur(rgbd.depth, ksize=self.args.net.depth_blur_ksize)

            Zb = depth_blur[locs].reshape((-1, 1)).astype('float')
            Zbg = self.bg_depth[locs].reshape((-1, 1))

            zlocs = np.where(np.abs(Zb - Zbg) > self.args.net.bg_depth_diff)[0].tolist()
            num_grasp = len(zlocs)
            print(f'{num_grasp} candidates: depth different > {self.args.net.bg_depth_diff}')
        else:
            # if not hasattr(self, 'corner_depth_map'): self.get_workspace_corner_depth(rgbd)
            reference_depth_map = self.get_workspace_corner_depth(rgbd)
            # reference_depth_map = np.array(self.args.reference_depth_map)
            num_grasp, num_corner = len(locs[0]), len(reference_depth_map)
            corner_loc_map = ArrayUtils().repmat(reference_depth_map[:, :2], (1, 1, num_grasp))
            suc_loc_map = ArrayUtils().repmat(
                np.concatenate((locs[1].reshape(1, 1, -1), locs[0].reshape(1, 1, -1)), axis=1),
                (num_corner, 1, 1)).astype('float')
            dmap = corner_loc_map - suc_loc_map
            dmap = np.linalg.norm(dmap, axis=1)
            inv_dmap = np.max(dmap) - dmap
            dsum = ArrayUtils().repmat(np.sum(inv_dmap, axis=0).reshape((1, -1)), (num_corner, 1)) + 0.00001
            W = np.divide(inv_dmap, dsum)
            ZZ = np.multiply(W, ArrayUtils().repmat(reference_depth_map[:, -1].reshape((-1, 1)), (1, num_grasp)))
            ZZ = np.sum(ZZ, axis=0).reshape((-1, 1))

            depth_blur = cv2.blur(rgbd.depth, ksize=self.args.net.depth_blur_ksize)
            Zb = depth_blur[locs].reshape((-1, 1)).astype('float')
            zlocs = np.where(np.abs(Zb - ZZ) > self.args.net.bg_depth_diff)[0].tolist()
            num_grasp = len(zlocs)
            print(f'{num_grasp} candidates: depth different > {self.args.net.bg_depth_diff}')
        return zlocs

    def get_high_score_inds(self, Grasp):
        return np.where(Grasp[:, -1]>self.args.net.score_thresh)[0].flatten().tolist()

    def get_low_score_inds(self, Grasp):
        return np.where(Grasp[:, -1]<=self.args.net.score_thresh)[0].flatten().tolist()

    def get_exp_scores(self,rgbd, Grasp):
        spatial_w = ArrayUtils().truncate(1 - rgbd.workspace.center_bias(pts=Grasp[:, :2]), vmin=0, vmax=1)
        depth_w = ArrayUtils().truncate(1 - Grasp[:, 2] / self.args.sensor.depth_max, vmin=0, vmax=1)
        valid_depth = (Grasp[:, 2] > 100).astype('float')
        return (depth_w) * (spatial_w ** 0.3) * valid_depth * Grasp[:, -1]

    def select_best_ind(self, rgbd, Grasp):
        exp_scores = self.get_exp_scores(rgbd, Grasp)
        return int(np.argmax(exp_scores))

    def show_poses(self, rgbd, disp_mode='rgb', detected=None):
        if 'im' not in detected:out = rgbd.disp(mode=disp_mode)
        else:out = detected['im']
        best_poses, pose_types = [], []
        if 'suction' in detected:
            Suction = detected['suction']
            if Suction is not None:
                if self.args.flag.show_steps:
                    for suc in Suction[self.get_low_score_inds(Suction), :]:
                        xc, yc = suc[:2].astype('int')
                        cv2.drawMarker(out,(xc,yc), (0,0,255), cv2.MARKER_DIAMOND,self.args.disp.marker_size//2, 1)

                    for suc in Suction[self.get_high_score_inds(Suction), :]:
                        xc, yc = suc[:2].astype('int')
                        cv2.drawMarker(out,(xc,yc), (0,255,0), cv2.MARKER_DIAMOND,self.args.disp.marker_size//2, 1)

                best_ind = [self.select_best_ind(rgbd, Suction)]
                best_pose = Suction[best_ind,:]
                best_poses.append(best_pose)
                pose_types.append('suction')
                xc, yc = best_pose[:,:2].flatten().astype('int')
                cv2.drawMarker(out,(xc,yc), (255,0,0), cv2.MARKER_TILTED_CROSS,self.args.disp.marker_size, self.args.disp.marker_thick)
        if 'grip' in detected:
            Grip = detected['grip']
            if Grip is not None:
                if self.args.flag.show_steps:
                    for grip in Grip[self.get_low_score_inds(Grip), :]:
                        x0, y0, x1, y1 = grip[-5:-1].astype('int')
                        cv2.line(out, (x0, y0), (x1, y1), (0, 0, 255), 1)

                    for grip in Grip[self.get_high_score_inds(Grip), :]:
                        x0, y0, x1, y1 = grip[-5:-1].astype('int')
                        cv2.line(out, (x0, y0), (x1, y1), (0, 255, 0), 1)
                best_ind = [self.select_best_ind(rgbd, Grip)]
                best_pose = Grip[best_ind, :]
                best_poses.append(best_pose)
                pose_types.append('grip')
                x0, y0, x1, y1 = best_pose[:,-5:-1].flatten().astype('int')
                cv2.line(out, (x0, y0), (x1, y1), (255, 0, 0), self.args.disp.line_thick)

        if len(best_poses)!=0:
            ind = int(np.argmax(np.array([pp.flatten()[-1] for pp in best_poses])))
            best = {'type': pose_types[ind], 'pose': best_poses[ind]}
            detected.update({'best': best})
            if len(best_poses)>1:
                cv2.putText(out,best['type'],(50,50),cv2.FONT_HERSHEY_COMPLEX,
                            self.args.disp.text_scale, self.args.disp.text_color, self.args.disp.text_thick)

        if 'im' not in detected: detected.update({'im': out})
        else: detected['im'] = out
        return detected




































