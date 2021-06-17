from libs.dataset.polygon import *
from libs.dataset.cifar10 import Cifar10Maker, CombineCifar10Data, CifarDepth2Norm
from libs.dataset.polygon import OnMousePolygonDrawer, PolygonShower, PolygonAnalyzer
from libs.import_basic_utils import *
import json
import numpy as np
import math

class SuctionCifar10DMaker(Cifar10Maker):

    def process_this(self):
        self.mask_path = self.rgbd_path[0].replace('.png', '_mask.png')
        if not os.path.exists(self.mask_path):
            print('>>> %s does not exist >>> Skip' %self.mask_path)
            return False
        return super().process_this()

    def process_single(self):
        # read mask
        mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        mask_crop = ArrayUtils().rect_crop(mask, self.args.workspace.rect)

        rgbd_crop = self.rgbd.crop()




        # bound_mask = np.logical_and(50<mask_crop, mask_crop<200).astype('uint8')
        # bound_mask  =cv2.dilate(bound_mask, np.ones((3,3), 'uint8'))
        # suc_mask = np.logical_and(mask_crop>200,1-bound_mask).astype('uint8')
        # bg_mask = 1-(bound_mask+suc_mask)

        suc_mask_org = (mask_crop > 150).astype('uint8')
        suc_mask = cv2.erode(suc_mask_org,np.ones(self.args.suction_size, 'uint8'))
        bound_mask = cv2.dilate(suc_mask_org,np.ones(self.args.suction_size, 'uint8')) - suc_mask
        bg_mask = 1 - (bound_mask + suc_mask)


        locs_suc = np.where(suc_mask)
        locs_bound = np.where(bound_mask)
        locs_bg = np.where(bg_mask)

        num_suc_px = len(locs_suc[0])
        num_bound_px = len(locs_bound[0])
        num_bg_px = len(locs_bg[0])

        if self.args.show_steps:
            # cv2.imshow('mask_crop', mask_crop)
            # cv2.imshow('suc_mask', 255*suc_mask)
            # cv2.imshow('bg_mask', 255*bg_mask)
            # cv2.imshow('bound_mask', 255*bound_mask)
            rgbd_crop.show(args=self.args, mask=suc_mask)

        suc_mask = MaskUtils().sparse_mask(suc_mask, stride=round(math.sqrt(num_suc_px/self.args.num_roi)))
        bound_mask = MaskUtils().sparse_mask(bound_mask, stride=round(math.sqrt(num_bound_px / self.args.num_roi)))
        bg_mask = MaskUtils().sparse_mask(bg_mask, stride=round(math.sqrt(num_bg_px / self.args.num_roi)))

        locs_suc = np.where(suc_mask)
        locs_bound = np.where(bound_mask)
        locs_bg = np.where(bg_mask)

        num_suc_px = len(locs_suc[0])
        num_bound_px = len(locs_bound[0])
        num_bg_px = len(locs_bg[0])

        count = 0
        for pad in self.args.roi_pads:
            xp, yp = pad
            sleft, stop = xp//2, yp//2
            sright, sbottom = xp-sleft, yp-stop
            # for mmask, cls in zip([suc_mask, bound_mask, bg_mask], ['suction', 'bound', 'bg']):
            for mmask, cls in zip([suc_mask, bound_mask], ['suction', 'bound']):
                Y, X = np.where(mmask)
                self.cls_ind = self.args.classes.index(cls)

                for x, y in zip(X, Y):
                    rgbd_roi = rgbd_crop.crop(left=x-sleft, right=x+sright, top=y-stop, bottom=y+sbottom)
                    if self.args.show_steps: rgbd_crop.show(args=self.args, pt=(x,y), title='location')

                    # rgbd_crop.show(args=self.args, pt=(x,y), mask=suc_mask)
                    # rgbd_roi.show(args=self.args, title='roi')
                    # cv2.waitKey()
                    super().process_single(rgbd=rgbd_roi)
                    count+=1

        print('>>> %d samples ...' % count)



if __name__ =='__main__':
    cfg_path = 'configs/grasp_detection/suction_net.cfg'

    SuctionCifar10DMaker(cfg_path=cfg_path).run()
    # PolygonShower(cfg_path=cfg_path).run()
    # OnMouseDrawGrip(cfg_path=cfg_path).run()
    # PolygonAnalyzer(cfg_path=cfg_path).run()

    # CombineCifar10Data(cfg_path=cfg_path).run()
    # CifarDepth2Norm(cfg_path=cfg_path).run()




