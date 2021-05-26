import numpy as np
import cv2
import random
from random import sample
from scipy.ndimage import rotate
import xml.etree.ElementTree as ET
from glob import glob
import os
from ketisdk.utils.proc_utils import ProcUtils
import json

class DataUtils():
    def thicken_binary(self, im, px=1):
        # Thicken binary image by px
        bn = 255*(im >127).astype('uint8')
        if px == 0: return bn
        kernel = np.ones((px+1, px+1), 'uint8')
        return cv2.dilate(bn, kernel)

    def shrink_binary(self, im, px=1):
        # shrink binary image by px
        bn = 255*(im >127).astype('uint8')
        if px==0: return bn
        kernel = np.ones((px+1, px+1), 'uint8')
        return cv2.erode(bn, kernel)

    def get_random_pose2D(self, im_size):
        h, w = im_size

        angle = random.randint(0, 360)
        x, y = random.randint(0, w), random.randint(0, h)

        return ((x,y), angle)

    def paste_roi_on_im(self, im, roi, roi_mask, paste_loc, angle, scale, reshape=True, order=3,
                        inst_mask=None, cls_idx=1, inst_idx=0, show_steps=False):
        im_shape = im.shape
        h, w = im_shape[:2]
        if inst_mask is None: inst_mask = np.zeros((h,w), 'uint16')
        if show_steps: cv2.imshow('im', im)
        if show_steps: cv2.imshow('roi', roi)

        hr, wr = roi.shape[:2]
        if hr>wr: hro, wro = scale, int(scale*wr/hr)
        else: hro, wro = int(scale*hr/wr), scale

        roi = cv2.resize(roi, (wro, hro), interpolation=cv2.INTER_CUBIC)
        roi_mask = cv2.resize(roi_mask, (wro, hro), interpolation=cv2.INTER_CUBIC)
        if show_steps: cv2.imshow('1. scale roi', roi)

        roi = rotate(roi, angle, reshape=reshape, order=order)
        roi_mask = rotate(roi_mask, angle, reshape=reshape, order=order)

        valid_loc = np.where(roi_mask>0.5)
        if show_steps: cv2.imshow('2. rotate roi', roi)

        hr, wr = roi.shape[:2]

        left, top, right, bottom = self.fix_roi_loc_on_image((h,w), (hr, wr), paste_loc)
        Y, X = valid_loc[0] + top, valid_loc[1]+left
        im_valid_loc = (Y,X)

        im[im_valid_loc] = roi[valid_loc]
        inst_mask[im_valid_loc] = 1000*cls_idx + inst_idx

        if show_steps:
            cv2.imshow('3. paste roi to im', im)

        if show_steps: cv2.waitKey()

        return (im, inst_mask)

    def fix_roi_loc_on_image(self, im_size, roi_size, roi_center):
        h, w = im_size
        hr, wr = roi_size
        xc, yc = roi_center

        dleft, dtop = hr//2, wr//2
        left, top= xc-dleft, yc-dtop
        right, bottom = left+wr, top+hr
        if left<0:  left, right = 0, wr
        if right>w: left, right = w-wr, w
        if top<0: top, bottom= 0, hr
        if bottom>h: top, bottom = h-hr, h

        return (left, top, right, bottom)

    def save_VOC_annotation(self, ann_path, im_info, classes, boxes):
        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'folder').text = 'CABLE 2019'
        ET.SubElement(annotation, 'filename').text = im_info['filename']
        source = ET.SubElement(annotation, 'source')
        ET.SubElement(source, 'database').text = 'KETI DB'
        ET.SubElement(source, 'annotation').text = 'KETI 2019'
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(im_info['width'])
        ET.SubElement(size, 'height').text = str(im_info['height'])
        ET.SubElement(size, 'depth').text = str(im_info['depth'])
        ET.SubElement(annotation, 'segmented').text = str(0)

        for cls, crop_rect in zip(classes, boxes):
            object = ET.SubElement(annotation, 'object')
            ET.SubElement(object, 'name').text = cls
            ET.SubElement(object, 'difficult').text = str(0)
            bndbox = ET.SubElement(object, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(crop_rect[0])
            ET.SubElement(bndbox, 'ymin').text = str(crop_rect[1])
            ET.SubElement(bndbox, 'xmax').text = str(crop_rect[2])
            ET.SubElement(bndbox, 'ymax').text = str(crop_rect[3])

        ann_obj = ET.ElementTree(annotation)
        save_dir, _ =os.path.split(ann_path)
        os.makedirs(save_dir, exist_ok=True)
        ann_obj.write(ann_path)

    def visualize_VOC_annotation(self, ann_path, im=None, im_path=None):
        assert im is not None or im_path is not None
        if im_path is not None: im = cv2.imread(im_path)
        ann = self.extract_VOC_annotation(ann_path=ann_path)
        boxes, classes = ann['boxes'], ann['classes']


        out = np.copy(im)
        for box, cls in zip(boxes, classes):
            xmin, ymin, xmax, ymax = box
            out = cv2.rectangle(out, (xmin, ymin), (xmax, ymax),  # show bbox
                          color=(0,255,0), thickness=2)
            out = cv2.putText(out, cls, (xmin, ymin-5),cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0))

        return out

    def extract_VOC_annotation(self, ann_path):
        xmlroot = ET.parse(ann_path).getroot()

        im_info = dict()
        im_info.update({'filename': xmlroot.findall('filename')[0].text})
        size = xmlroot.findall('size')[0]
        im_info.update({'width': int(size.findall('width')[0].text)})
        im_info.update({'height': int(size.findall('height')[0].text)})
        im_info.update({'depth': int(size.findall('depth')[0].text)})

        boxes, classes = [], []
        for elem in xmlroot.findall('object'):
            cls = elem.findall('name')[0].text

            bndbox = elem.findall('bndbox')[0]
            xmin = int(float(bndbox.findall('xmin')[0].text))
            xmax = int(float(bndbox.findall('xmax')[0].text))
            ymin = int(float(bndbox.findall('ymin')[0].text))
            ymax = int(float(bndbox.findall('ymax')[0].text))

            classes.append(cls)
            boxes.append((xmin, ymin, xmax, ymax))
        return {'im_info':im_info,'boxes':boxes, 'classes': classes}

    def combine_VOC_annotation(self, ann_path, ann2_path, out_path):
        ret = self.extract_VOC_annotation(ann_path)
        im_info, boxes, classes = ret['im_info'], ret['boxes'], ret['classes']
        ret2 = self.extract_VOC_annotation(ann2_path)
        im_info2, boxes2, classes2 = ret2['im_info'], ret2['boxes'], ret2['classes']

        boxes += boxes2
        classes += classes2
        self.save_VOC_annotation(out_path, im_info, classes, boxes)




    def fix_VOC_annotation(self, ann_path, new_ann_path, classes, old_classes, new_classes):
        ann = self.extract_VOC_annotation(ann_path)
        im_info, ann_boxes, ann_classes = ann['im_info'], ann['boxes'], ann['classes']

        out_boxes, out_classes = [], []
        for box, cls in zip(ann_boxes, ann_classes):
            new_cls = cls
            if cls in old_classes: new_cls = new_classes[old_classes.index(cls)]
            if new_cls not in classes: continue
            out_boxes.append(box)
            out_classes.append(new_cls)

        self.save_VOC_annotation(new_ann_path, im_info, out_classes, out_boxes)

    def get_db_mean_var(self, db_dir):

        ims = [cv2.imread(path) for path in glob(os.path.join(db_dir, '*'))]

        means = [np.mean(im, axis=(0,1)) for im in ims]
        stds = [np.std(im, axis=(0,1)) for im in ims]

        mean_array =  np.vstack(means)
        std_array = np.vstack(stds)

        mean = np.mean(mean_array, axis=0)
        std = np.mean(std_array, axis=0)

        if len(mean)>1:
            mean = mean[::-1]
            std = std[::-1]

        return mean, std


    def get_VOC_train_layout(self,voc_root):
        paths = [path for path in glob(os.path.join(voc_root, 'JPEGImages', '*'))]

        train_set_obj = open(os.path.join(voc_root, 'train.txt'), 'w+')
        test_set_obj = open(os.path.join(voc_root, 'test.txt'), 'w+')
        trainval_set_obj = open(os.path.join(voc_root, 'trainval.txt'), 'w+')
        val_set_obj = open(os.path.join(voc_root, 'val.txt'), 'w+')

        train_part, val_part, test_part = 3, 1,1

        total = train_part +val_part +test_part
        for j,path in enumerate(paths):
            _, filename = os.path.split(path)
            filename, _ = os.path.splitext(filename)
            kk = j % total
            if kk < train_part:
                train_set_obj.writelines(filename + '\n')
                trainval_set_obj.writelines(filename + '\n')
            if (train_part <= kk) and (kk < train_part + val_part):
                val_set_obj.writelines(filename + '\n')
                trainval_set_obj.writelines(filename + '\n')
            if kk >= train_part + val_part:
                test_set_obj.writelines(filename + '\n')

        train_set_obj.close()
        test_set_obj.close()
        trainval_set_obj.close()
        val_set_obj.close()

class JsonUtils():
    def read_json(self, json_path):
        """ read json file from path

        :param json_path: json file path
        :type json_path: str
        :return: dict of contents
        """
        assert ProcUtils().isexists(json_path)
        json_file = open(json_path)
        json_dict = json.load(json_file)
        json_file.close()
        return json_dict

    def save_poly_json(self, json_path, im_info, class_list, polygons):
        """ save polygon to json file

        :param filename: json filename
        :param im_info: include image height, width, number of channel, and image name
        :param class_list: list of classes
        :param polygons: list of polygon
        :param folds: serial of folders saving json file
        """

        # val json
        Ann_dict = dict()
        fold, _ = os.path.split(json_path)
        if not os.path.exists(fold): os.makedirs(fold)

        Ann_dict.update({'Info': im_info})
        Ann_dict.update({'Classes': class_list})
        Ann_dict.update({'Polygons': polygons})

        instance_json_obj = open(json_path, 'w+')
        instance_json_obj.write(json.dumps(Ann_dict))
        instance_json_obj.close()
        
























if __name__=='__main__':
    # mean, std = DataUtils().get_db_mean_var('/mnt/workspace/000_KETI_AIKit/data/ikea/synthetic_data/combine_20200903/JPEGImages')
    DataUtils().get_VOC_train_layout('/mnt/workspace/000_KETI_AIKit/data/ikea/synthetic_data/combine_20200904')


    aa = 1











