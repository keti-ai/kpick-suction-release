# from KAI.utils.proc_utils import *
# from KAI.utils.data_processing import *
# from KAI.utils.makedb_utilts import *
# from KAI.dataset.polygon import *
from glob import glob
from scipy.ndimage import rotate
# from KAI.basic.basic_objects import BasObj
import json
import os
import cv2

class GripDBer(BasObj):

    def load_params(self, args):
        super().load_params(args=args)
        aug_bg_dir = os.path.join(args.root_dir, args.aug_bg_dir)
        assert os.path.exists(aug_bg_dir)
        all_bg_path = glob(os.path.join(aug_bg_dir, '*'))
        self.all_bg = [cv2.resize(cv2.imread(p), args.out_size[::-1]) for p in all_bg_path]
        self.all_angle = list(np.arange(0,360, args.step_angle))
        self.num_aug = min(len(self.all_bg), len(self.all_angle))

        # load
        self.info_dir  = os.path.join(args.root_dir, args.info_dir)
        self.count_file = os.path.join(self.info_dir, self.args.count_file)
        self.rgb_mean_file = os.path.join(self.info_dir, self.args.rgb_mean_file)
        self.std_mean_file = os.path.join(self.info_dir, self.args.std_mean_file)
        self.rgb_aug_dir = os.path.join(self.args.root_dir, self.args.rgb_aug_dir)
        if not os.path.exists(self.rgb_aug_dir): os.makedirs(self.rgb_aug_dir)
        self.label_aug_dir = os.path.join(self.args.root_dir, self.args.label_aug_dir)
        if not os.path.exists(self.label_aug_dir): os.makedirs(self.label_aug_dir)

    def poly2maskAug(self, inputs, filename='unname', title='viewer'):
        json_path = os.path.join(self.args.root_dir, self.args.ann_dir, filename + '.json')
        if not os.path.exists(json_path):
            print('polygon file does not exist ...')
            return 1

        im_in = inputs['rgb']
        height_in, width_in = im_in.shape[:2]

        json_file = open(json_path)
        ann_dict = json.load(json_file)
        json_file.close()
        label_in = polys2masks(ann_dict['Polygons'], im_in.shape[:2])

        # random select background,  angle, flip
        select_bg = random.sample(self.all_bg, k=self.num_aug)
        select_angle = random.sample(self.all_angle, k=self.num_aug)

        loc_flip = random.sample(range(self.num_aug), k=int(self.num_aug / 2))
        select_flip = np.zeros(shape=(self.num_aug, 1), dtype=np.uint8)
        select_flip[loc_flip] = 1
        select_flip = tuple(select_flip)

        if not os.path.exists(self.info_dir): os.makedirs(self.info_dir)
        if not os.path.exists(self.count_file): count=0
        else: count = int(np.load(self.count_file))
        if not os.path.exists(self.rgb_mean_file): rgb_mean=0
        else: rgb_mean = np.load(self.rgb_mean_file)
        if not os.path.exists(self.std_mean_file): std_mean=0
        else: std_mean = np.load(self.std_mean_file)
        mask = np.ones((height_in, width_in))
        for ind in range(self.num_aug):
            bg = select_bg[ind]
            angle = select_angle[ind]

            # crop to interesting region
            im_crop = np.copy(im_in)
            label_crop = np.copy(label_in)
            mask_crop = np.copy(mask)

            # rotate
            im_rot = rotate(im_crop, angle=angle, reshape=True, order=3)
            label_rot = rotate(label_crop, angle=angle, reshape=True, order=0)
            mask_rot = rotate(mask_crop, angle=angle, reshape=True, order=0)

            # selective flip and change color
            if bool(select_flip[ind]):
                im_rot = np.flip(im_rot, axis=0)
                label_rot = np.flip(label_rot, axis=0)
                mask_rot = np.flip(mask_rot, axis=0)

            # fix scales
            height_rot, width_rot, _ = im_rot.shape

            height_rat = self.args.out_size[0] / height_rot
            width_rat = self.args.out_size[1] / width_rot

            if (height_rat < 1) or (width_rat < 1):
                if height_rat < width_rat:
                    im_rot = cv2.resize(im_rot, None, fx=height_rat, fy=height_rat, interpolation=cv2.INTER_CUBIC)
                    label_rot = cv2.resize(label_rot, None, fx=height_rat, fy=height_rat,
                                           interpolation=cv2.INTER_NEAREST)
                    mask_rot = cv2.resize(mask_rot, None, fx=height_rat, fy=height_rat, interpolation=cv2.INTER_CUBIC)

                if width_rat <= height_rat:
                    im_rot = cv2.resize(im_rot, None, fx=width_rat, fy=width_rat, interpolation=cv2.INTER_CUBIC)
                    label_rot = cv2.resize(label_rot, None, fx=width_rat, fy=width_rat, interpolation=cv2.INTER_NEAREST)
                    mask_rot = cv2.resize(mask_rot, None, fx=width_rat, fy=width_rat, interpolation=cv2.INTER_CUBIC)

            height_rot, width_rot, _ = im_rot.shape

            # paste to background
            im_out = np.copy(bg)
            label_out = np.zeros(self.args.out_size, dtype=np.uint16)

            label_out[0:height_rot, 0:width_rot] = np.multiply(label_out[0:height_rot, 0:width_rot], 1 - mask_rot) + \
                                                   np.multiply(label_rot, mask_rot)
            for ii in range(3):
                im_out[0:height_rot, 0:width_rot, ii] = np.multiply(im_out[0:height_rot, 0:width_rot, ii],
                                                                    1 - mask_rot) + \
                                                        np.multiply(im_rot[:, :, ii], mask_rot)
            # save

            cv2.imwrite(os.path.join(self.rgb_aug_dir, str(count + self.args.aug_start_from) + '.png'), im_out)
            cv2.imwrite(os.path.join(self.label_aug_dir, str(count) + '.png'), label_out)

            rgb_mean = count/(count + 1)* rgb_mean + 1/(count+1)*np.mean(im_out.astype('float')/255, axis=(0, 1))
            std_mean = count/(count + 1)* std_mean + 1/(count+1)*np.std(im_out.astype('float') /255, axis=(0, 1))
            count += 1

        np.save(self.count_file, count)
        np.save(self.rgb_mean_file, rgb_mean)
        np.save(self.std_mean_file, std_mean)
        return 0




