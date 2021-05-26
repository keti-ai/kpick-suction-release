from libs.basic.basic_sequence import BasSeqImDetectCallback
from libs.basic.basic_imscanning_ray import EdgePairFinder
from libs.import_basic_utils import *
from libs.basic.basic_objects import BasDoLabel, BasJson
from grasp_detection.processing.grip_detection_v6 import GripDetector
import cv2
from time import time
import os



class LabelGripEdge(BasSeqImDetectCallback):
    def get_model(self):
        super().get_model()
        # self.model = EdgePairFinder(cfg_path=self.cfg_path)
        # self.detector = self.model.find_grip_candidates_from_edges
        self.model = GripDetector(cfg_path=self.cfg_path)
        self.detector = self.model.find_grip_pose_from_edges_v2
        self.labeler = BasDoLabel(cfg_path=self.cfg_path)


    def detect(self):
        self.Grips = self.detector(rgbd=self.rgbd)
        # self.grips = self.Grips.grips
        self.grips = self.Grips.get_top_higher_grips(score=0.2)
        # self.grips = self.detector(rgbd=self.rgbd)

    def process_single(self):
        super().process_single()
        t = time()

        self.detect()
        if self.grips is None:
            print('Grip pairing found no grip ...')
            return self.rgbd, None

        num_grip = len(self.grips)
        print('Found %d grip candidates in %0.3f s' % (num_grip, time() - t))

        self.rgbd.show(args=self.args, grips=GRIPS(grips=self.grips))
        print('ESC: next image, ANY KEY: continue')
        if cv2.waitKey() != ESC_KEY:
            self.labeled_grips = []
            for i in range(num_grip):
                ProcUtils().clscr()
                print('Grip %d out of %d ...' %(i,num_grip))
                self.rgbd.show(args=self.args, grips=GRIPS(grips=[self.grips[i]], disp_colors=[(0, 255, 0)]))
                print('ENTER: skip this, ANY KEY: continue')
                if cv2.waitKey() == ENTER_KEY: continue

                label_ret = self.labeler.get_key_label()


                if label_ret is not None:
                    self.labeled_grips.append(self.grips[i])
                    self.rgbd.show(args=self.args,
                                   grips=GRIPS(grips=[self.labeled_grips[-1]],
                                               disp_colors=[self.labeler.disp_colors[-1]]))

                print('ESC: next image, ANY KEY: continue')
                if cv2.waitKey() == ESC_KEY: break

            Grips = None
            if len(self.labeled_grips)>0: Grips = GRIPS(grips=self.labeled_grips, disp_colors=self.labeler.disp_colors)
            self.rgbd.show(args=self.args, grips=Grips)

            # save
            top, left, _, _ =  self.rgbd.roi
            for i in range(len(self.labeled_grips)): self.labeled_grips[i].shift(dx=-left, dy=-top)
            polys = [grip.pts for grip in self.labeled_grips]
            jsonname, _ = os.path.splitext(self.filename)
            BasJson().save_poly_json(jsonname + '.json', self.labeler.get_im_info(self.rgbd.rgb, self.filename),
                                     self.labeler.class_list, polys, [self.args.root_dir, self.args.ann_poly])
            ArrayUtils().save_array(self.rgbd.crop_rgb()[:,:,::-1], [self.args.root_dir, self.args.ann_rgb], self.filename)
            ArrayUtils().save_array(self.rgbd.crop_depth(), [self.args.root_dir, self.args.ann_depth], self.filename)

            # move file
            self.move_processed_rgbd()






if __name__=='__main__':
    LabelGripEdge(cfg_path='configs/grasp_detection/grip_v3.cfg').run()