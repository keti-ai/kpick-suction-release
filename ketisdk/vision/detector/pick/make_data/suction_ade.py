from libs.basic.basic_sequence import SeqAccumulator
import os
import cv2
import numpy as np
from libs.import_basic_utils import *
import json

class SuctionCifar10DMaker(SeqAccumulator):

    def process_this(self):
        self.mask_path = self.rgbd_path[0].replace('.png', '_mask.png')
        if not os.path.exists(self.mask_path):
            print('>>> %s does not exist >>> Skip' %self.mask_path)
            return False
        return super().process_this()
    def init_acc(self):
        super().init_acc()
        list_dir = os.path.join(self.args.root_dir, 'ade')
        if not os.path.exists(list_dir): os.makedirs(list_dir)
        self.train_list_file = open(os.path.join(list_dir,'training.odgt'), 'w+')
        self.test_list_file = open(os.path.join(list_dir, 'validation.odgt'), 'w+')

    def acc_sum(self):
        super().acc_sum()
        self.train_list_file.close()
        self.test_list_file.close()

    def process_single(self):
        # read mask
        mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        mask_crop = ArrayUtils().rect_crop(mask, self.args.workspace.rect)

        rgbd_crop = self.rgbd.crop()

        suc_mask_org = (mask_crop > 150).astype('uint8')
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.args.suction_size)
        suc_mask = cv2.erode(suc_mask_org,ker)

        # ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4*self.args.suction_size[0], 4*self.args.suction_size[1]))
        bound_mask = cv2.dilate(suc_mask_org, ker) - suc_mask

        # bg_mask = 1 - (bound_mask + suc_mask)

        mask_cmb = suc_mask + 2*bound_mask

        if self.args.show_steps:
            # cv2.imshow('mask_crop', mask_crop)
            # cv2.imshow('suc_mask', 255*suc_mask)
            # cv2.imshow('bg_mask', 255*bg_mask)
            # cv2.imshow('bound_mask', 255*bound_mask)
            cv2.imshow('mask_cmb', 50*mask_cmb)
            rgbd_crop.show(args=self.args, mask=bound_mask)
            if cv2.waitKey()==27: exit()


        if self.args.depth2norm:
            depth = ArrayUtils().get_mat_normal_map_U8(rgbd_crop.depth)
        else: depth = rgbd_crop.depth_U8()

        for j,color_order in enumerate(self.args.aug_color_orders):
            rgb = rgbd_crop.rgb[:,:,color_order]
            data = np.concatenate((rgb, depth), axis=2)
            height, width = rgb.shape[:2]


            # save images
            ann_filename = '%s_%d.png'%(self.filename, j)


            # update
            if self.to_train(self.acc['total']):
                num_train = self.acc['num_train']
                self.acc['mean'] = num_train / (num_train + 1) * self.acc['mean'] + \
                                   1 / (num_train + 1) * np.mean(data.astype('float') / 255, axis=(0, 1))
                self.acc['std'] = num_train / (num_train + 1) * self.acc['std'] + \
                                  1 / (num_train + 1) * np.std(data.astype('float') / 255, axis=(0, 1))
                self.acc['num_train'] += 1

                train_info = dict()
                fpath_img = os.path.join(self.args.root_dir, self.args.ann_rgb, 'training',ann_filename)
                train_info.update({'fpath_img':fpath_img})
                fpath_segm=os.path.join(self.args.root_dir, self.args.ann_mask, 'training', ann_filename)
                train_info.update({'fpath_segm': fpath_segm})
                fpath_depth = os.path.join(self.args.root_dir, self.args.ann_depth, 'training', ann_filename)
                # train_info.update({'fpath_depth': fpath_depth})
                train_info.update({'width': width, 'height':height})

                self.train_list_file.write(json.dumps(train_info) + '\n')

            else:
                self.acc['num_test'] += 1

                test_info = dict()
                fpath_img = os.path.join(self.args.root_dir, self.args.ann_rgb, 'validation', ann_filename)
                test_info.update({'fpath_img': fpath_img})
                fpath_segm = os.path.join(self.args.root_dir, self.args.ann_mask, 'validation', ann_filename)
                test_info.update({'fpath_segm': fpath_segm})
                fpath_depth = os.path.join(self.args.root_dir, self.args.ann_depth, 'validation', ann_filename)
                # test_info.update({'fpath_depth': fpath_depth})
                test_info.update({'width': width, 'height': height})

                self.test_list_file.write(json.dumps(test_info) + '\n')

            ArrayUtils().save_array_v3(array=rgb[:, :, ::-1], filepath=fpath_img)
            ArrayUtils().save_array_v3(array=depth, filepath=fpath_depth)
            ArrayUtils().save_array_v3(array=mask_cmb, filepath=fpath_segm)

            self.acc['total'] = self.acc['num_train'] + self.acc['num_test']



    def to_train(self, count):
        dur = self.args.train_val_div[0] + self.args.train_val_div[1]
        return (count % dur) < self.args.train_val_div[0]














if __name__ =='__main__':
    cfg_path = 'configs/segmentation/suction.cfg'

    SuctionCifar10DMaker(cfg_path=cfg_path).run()
    # PolygonShower(cfg_path=cfg_path).run()
    # OnMouseDrawGrip(cfg_path=cfg_path).run()
    # PolygonAnalyzer(cfg_path=cfg_path).run()

    # CombineCifar10Data(cfg_path=cfg_path).run()
    # CifarDepth2Norm(cfg_path=cfg_path).run()




