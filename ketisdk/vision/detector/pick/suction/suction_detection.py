import numpy as np
from ketisdk.import_basic_utils import *

from torchvision.transforms import transforms
from ketisdk.vision.detector.classifier.roi_classifier import RoiCifarClassfier
import cv2
from ketisdk.vision.detector.pick.base_grasp_detection import BaseGraspDetector

class SuctionDetector(BaseGraspDetector):

    def find_candidates(self, rgbd):
        cleft, ctop, cright, cbottom = rgbd.workspace.bbox
        # cheight, cwidth = cbottom - ctop, cright - cleft

        # get rois
        pad_size_array = np.array(self.args.net.pad_sizes)
        px_max, py_max = np.amax(pad_size_array, axis=0).astype('int')
        # xc_shift, yc_shift = px_max // 2, py_max // 2

        Yc, Xc = np.mgrid[ctop:cbottom:self.args.net.stride, cleft:cright:self.args.net.stride]
        Yc, Xc = Yc.reshape((-1, 1)), Xc.reshape((-1, 1))

        all_boxes, all_width, all_height = [], [], []
        for p in self.args.net.pad_sizes:
            ww, hh = p
            Left, Top = np.maximum(cleft, Xc - ww // 2), np.maximum(ctop, Yc - hh // 2)
            Right, Bottom = np.minimum(Left + ww, cright), np.minimum(Top + hh, cbottom)
            all_width.append(Right - Left)
            all_height.append(Bottom - Top)
            all_boxes.append(np.concatenate((Left, Top, Right, Bottom), axis=1))
        all_width = np.concatenate(all_width, axis=1)
        all_height = np.concatenate(all_height, axis=1)
        valid_locs = np.where((np.min(all_width, axis=1) > 5) & (np.min(all_height, axis=1) > 5))[0].flatten().tolist()
        all_boxes = [bxs[valid_locs, :] for bxs in all_boxes]

        all_Yc = [np.mean(bxs[:, [1, 3]], axis=1).reshape((-1, 1)) for bxs in all_boxes]
        Yc = np.mean(np.concatenate(all_Yc, axis=1), axis=1).reshape((-1, 1))
        all_Xc = [np.mean(bxs[:, [0, 2]], axis=1).reshape((-1, 1)) for bxs in all_boxes]
        Xc = np.mean(np.concatenate(all_Xc, axis=1), axis=1).reshape((-1, 1))
        locs = (Yc.flatten().astype('int'), Xc.flatten().astype('int'))
        Zc = rgbd.depth[locs].reshape((-1, 1))
        Suction = np.concatenate((Xc, Yc, Zc), axis=1)

        print(f'{len(all_boxes[0])} candidates with {len(all_boxes[0]) * len(all_boxes)} boxes')
        return Suction, all_boxes

    def get_grasp_in_workspace(self, rgbd, Suction, all_boxes):
        # contain_all_boxes = [rgbd.workspace.contain_boxes(bxs) for bxs in all_boxes]
        # contain_boxes = True
        # for contains in contain_all_boxes: contain_boxes *= contains
        contain_boxes = rgbd.workspace.contain_points(Suction[:,:2].astype('int'))
        clocs = np.where(contain_boxes)[0].flatten().tolist()
        Suction = Suction[clocs, :]
        all_boxes = [bxs[clocs, :] for bxs in all_boxes]
        print(f'{len(all_boxes[0])} candidates in workspace')
        return Suction, all_boxes

    def remove_grasp_on_background(self, rgbd, Suction, all_boxes):
        # all_Yc = [np.mean(bxs[:, [1, 3]], axis=1).reshape((-1, 1)) for bxs in all_boxes]
        # Yc = np.mean(np.concatenate(all_Yc, axis=1), axis=1).reshape((-1, 1))
        # all_Xc = [np.mean(bxs[:, [0, 2]], axis=1).reshape((-1, 1)) for bxs in all_boxes]
        # Xc = np.mean(np.concatenate(all_Xc, axis=1), axis=1).reshape((-1, 1))
        # locs = (Yc.flatten().astype('int'), Xc.flatten().astype('int'))
        fg_locs = self.get_fg_depth_inds(rgbd, [Suction[:,1].astype('int'), Suction[:,0].astype('int')])
        Suction = Suction[fg_locs, :]
        all_boxes = [bxs[fg_locs, :] for bxs in all_boxes]
        return Suction, all_boxes

    def wrap_suction_from_boxes(self, rgbd, all_boxes):
        all_Yc = [np.mean(bxs[:, [1, 3]], axis=1).reshape((-1, 1)) for bxs in all_boxes]
        Yc = np.mean(np.concatenate(all_Yc, axis=1), axis=1).reshape((-1, 1))
        all_Xc = [np.mean(bxs[:, [0, 2]], axis=1).reshape((-1, 1)) for bxs in all_boxes]
        Xc = np.mean(np.concatenate(all_Xc, axis=1), axis=1).reshape((-1, 1))
        locs = (Yc.flatten().astype('int'), Xc.flatten().astype('int'))
        Zc = rgbd.depth[locs].reshape((-1, 1))
        Suction = np.concatenate((Xc, Yc, Zc), axis=1)
        return Suction

    def score_suction(self, array, all_boxes):
        # array = rgbd.array(get_rgb=self.args.get_rgb, get_depth=self.args.get_depth, depth2norm=self.args.depth2norm)
        array_tensor = self.transform(array)
        all_scores = [self.predict_tensor_rois([array_tensor, ], bxs)[:, self.gidx].reshape((-1, 1)) for bxs in
                      all_boxes]
        all_scores = np.concatenate(all_scores, axis=1)
        scores = np.mean(all_scores, axis=1).reshape((-1, 1))
        return scores

    def average_score_suction(self, rgbd, Suction):
        score_map = np.zeros((rgbd.height, rgbd.width), 'float')
        locs = [Suction[:, 1].astype('int'), Suction[:, 0].astype('int')]
        score_map[locs] = Suction[:,-1]

        kernel_org = np.array(self.args.net.nb_kernel)
        kernel_org = kernel_org/np.sum(kernel_org)

        kh_org, kw_org = kernel_org.shape
        kh , kw = self.args.net.stride*(kh_org-1)+1, self.args.net.stride*(kw_org-1)+1
        kernel = np.zeros((kh,kw), 'float')
        kernel[::self.args.net.stride, ::self.args.net.stride] = kernel_org

        avg_score_map = np.clip(cv2.filter2D(score_map,-1, kernel=kernel), 0, 1)
        # score_map_ = cv2.resize(score_map,None, fx=1/self.args.net.stride, fy=1/self.args.net.stride,
        #                         interpolation=cv2.INTER_CUBIC)
        # avg_score_map_ = np.clip(cv2.filter2D(score_map_,-1, kernel=kernel_org), 0, 1)
        # avg_score_map = cv2.resize(avg_score_map_,dsize=(rgbd.width, rgbd.height), interpolation=cv2.INTER_CUBIC)

        if self.args.net.do_suc_at_center:
            suc_map = (avg_score_map > self.args.net.score_thresh).astype('uint8')
            kernel = np.ones((2 * self.args.net.stride + 1, 2 * self.args.net.stride + 1), 'uint8')
            suc_map = cv2.morphologyEx(suc_map, cv2.MORPH_CLOSE, kernel)
            ret, labels = cv2.connectedComponents(suc_map)

            avg_score_map *= .2
            rr = self.args.net.stride
            for lb in range(1, ret):
                Y, X = np.where(labels == lb)
                # yc, xc = int(np.mean(Y)), int(np.mean(X))
                left, top, right, bottom = np.amin(X), np.amin(Y), np.amax(X), np.amax(Y)
                h,w = bottom-top, right-left
                if h<=self.args.net.stride or w<=self.args.net.stride: continue
                region_small = (right-left<=2*self.args.net.stride) or (bottom-top<=2*self.args.net.stride)
                # if right-left<=2*self.args.net.stride:continue
                # if bottom-top<=2*self.args.net.stride:continue
                yc, xc = (top+bottom)//2, (left+right)//2
                Y, X = np.meshgrid(range(yc - rr, yc + rr), range(xc - rr, xc + rr))

                mag = 5 if region_small else 7
                avg_score_map[(Y.flatten(), X.flatten())] *=mag

            avg_score_map = np.clip(avg_score_map, 0, 1)

        Suction[:,-1] = avg_score_map[locs]
        return Suction


    def sort_suction(self, Suction):
        sinds = np.argsort(Suction[:,-1].flatten())[::-1].flatten().tolist()  # sorting
        Suction = Suction[sinds, :]
        return Suction

    def detect_poses(self, rgbd, remove_bg=False):
        timer = Timer()
        self.gidx = self.args.net.classes.index('suction')

        Suction, all_boxes = self.find_candidates(rgbd)
        timer.pin_time('Find_candidates')

        Suction, all_boxes = self.get_grasp_in_workspace(rgbd, Suction, all_boxes)
        timer.pin_time('Remove_workspace')

        if remove_bg:
            Suction, all_boxes = self.remove_grasp_on_background(rgbd, Suction, all_boxes)
            if len(Suction)==0: return None
            timer.pin_time('Remove_background')


        array = rgbd.array(get_rgb=self.args.net.get_rgb, get_depth=self.args.net.get_depth, depth2norm=self.args.net.depth2norm)
        norm_map = array[:,:,-3:]/255
        # kernel = np.ones((5,5), 'float')/25
        # norm_map = cv2.filter2D(norm_map,-1,kernel=kernel)
        # norm_map = np.divide(norm_map, ArrayUtils().repmat(np.linalg.norm(norm_map, axis=2), (1, 1, 3)) + 0.00001)
        locs = (Suction[:,1].astype('int'),Suction[:,0].astype('int'))
        Vx, Vy, Vz = norm_map[:,:,0][locs].reshape((-1,1)), \
                     norm_map[:,:,1][locs].reshape((-1,1)), norm_map[:,:,2][locs].reshape((-1,1))
        Suction = np.concatenate((Suction, Vx, Vy, Vz), axis=1)
        timer.pin_time('get_normvec')

        scores = self.score_suction(array, all_boxes)
        Suction = np.concatenate((Suction,scores), axis=1)
        timer.pin_time('Scoring')

        Suction = self.average_score_suction(rgbd, Suction)
        timer.pin_time('Average_score')


        Suction = self.sort_suction(Suction)
        print(timer.pin_times_str())

        return Suction

    def detect_and_show_poses(self,rgbd, remove_bg=False ,disp_mode='rgb', detected=None):
        Suction = self.detect_poses(rgbd=rgbd, remove_bg=remove_bg)
        if detected is None: detected = {'suction': Suction}
        else:detected.update({'suction': Suction})
        detected = self.show_poses(rgbd, disp_mode=disp_mode, detected=detected)
        return detected

    def detect_and_show_step0(self, rgbd, disp_mode='rgb'):
        out = rgbd.disp(mode=disp_mode)
        Suction, all_boxes = self.find_candidates(rgbd)
        for bxs in all_boxes:
            for bx in bxs:
                left,top, right, bottom = bx.astype('int')
                cv2.rectangle(out,(left,top), (right, bottom), (0,255,0),1)
        return {'im':out}

    def detect_and_show_step1(self, rgbd, disp_mode='rgb'):
        out = rgbd.disp(mode=disp_mode)
        Suction, all_boxes  = self.find_candidates(rgbd)
        Suction, all_boxes  = self.get_grasp_in_workspace(rgbd, Suction, all_boxes)
        for bxs in all_boxes:
            for bx in bxs:
                left,top, right, bottom = bx.astype('int')
                cv2.rectangle(out,(left,top), (right, bottom), (0,255,0),1)
        return {'im':out}

    def detect_and_show_step2(self, rgbd, disp_mode='rgbd'):
        out = rgbd.disp(mode=disp_mode)
        Suction, all_boxes = self.find_candidates(rgbd)
        Suction, all_boxes = self.get_grasp_in_workspace(rgbd, Suction, all_boxes )
        Suction, all_boxes = self.remove_grasp_on_background(rgbd, Suction, all_boxes )
        for bxs in all_boxes:
            for bx in bxs:
                left,top, right, bottom = bx.astype('int')
                cv2.rectangle(out,(left,top), (right, bottom), (0,255,0),1)
        return {'im':out}


    # def get_exp_scores(self,rgbd, Suction):
    #     edge = rgbd.get_rgb_edges()
    #     px_max, py_max = np.amax(np.array(self.args.pad_sizes), axis=0).astype('int')
    #     # rx, ry = px_max//2, py_max//2
    #     rx, ry = self.args.count_edge_size[0]//2, self.args.count_edge_size[0]//2
    #     num_edge_map = []
    #     for suc in Suction:
    #         xc,yc = suc[:2].astype('int')
    #         num_edge_map.append(np.sum(edge[yc-ry:yc+ry+1, xc-rx:xc+rx+1]))
    #     edge_w = ArrayUtils().truncate(1 - np.array(num_edge_map) / self.args.num_edge_px_max, vmin=0, vmax=1)
    #     return (edge_w**0.2)*super().get_exp_scores(rgbd, Suction)

from ketisdk.vision.base.base_objects import DetGui
class SuctionGuiDetector(SuctionDetector, DetGui):
    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', detected=None):
        if method_ind == 0:
            ret = self.detect_and_show_poses(rgbd=rgbd, disp_mode=disp_mode, detected=detected)
        if method_ind==1:
            ret = self.detect_and_show_poses(rgbd=rgbd, remove_bg=True, disp_mode=disp_mode, detected=detected)
        if method_ind==2:
            ret = self.get_background_depth_map(rgbd=rgbd)
        if method_ind == 3:
            ret = self.detect_and_show_step0(rgbd=rgbd, disp_mode=disp_mode)
        if method_ind == 4:
            ret = self.detect_and_show_step1(rgbd=rgbd, disp_mode=disp_mode)
        if method_ind == 5:
            ret = self.detect_and_show_step2(rgbd=rgbd, disp_mode=disp_mode)
        return ret

def test_suction_detector(cfg_path):
    detector = SuctionDetector(cfg_path=cfg_path)
    aa = 1



































