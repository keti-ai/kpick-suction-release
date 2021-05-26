from ..utils.data_utilts import JsonUtils
import os
import pycocotools.mask as cocoMaskUtils
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from datetime import datetime
import cv2
import numpy as np



class CocoUtils():
    def __init__(self, ann_path=None):
        self.ann_path = ann_path
        if ann_path is not None:
            ret = JsonUtils().read_json(ann_path)
            self.image_dict_list, self.annotation_dict_list, self.categories = ret['images'], ret['annotations'], ret['categories']
            self.image_inds = [im_dict['id'] for im_dict in self.image_dict_list]

    def segmToRLE(self, segm, h, w):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = cocoMaskUtils.frPyObjects(segm, h, w)
            rle = cocoMaskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = cocoMaskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm
        return rle

    def segmToMask(self, segm, h, w):
        rle = self.segmToRLE(segm, h, w)
        return cocoMaskUtils.decode(rle)

    def get_im_path(self, image_id, im_dir):
        im_dict = self.image_dict_list[self.image_inds.index(image_id)]
        return os.path.join(im_dir, im_dict['file_name'])

    def get_im_from_ind(self, image_id, im_dir):
        im_path = self.get_im_path(image_id, im_dir)
        print(im_path)
        return cv2.imread(im_path)[:,:,::-1]

    def get_instance_from_ind(self, ind, ann_path=None):
        if ann_path is not None:
            ret = JsonUtils().read_json(ann_path)
            annotation_dict_list = ret['annotations']
        else: annotation_dict_list= self.annotation_dict_list

        ann_dict = annotation_dict_list[ind]
        segm = ann_dict['segmentation']
        kpts = ann_dict['keypoints']
        bbox = ann_dict['bbox']
        image_id = ann_dict['image_id']
        category_id = ann_dict['category_id']

        return {'image_id': image_id,'segmentation':segm, 'bbox': bbox, 'keypoints': kpts, 'category_id':category_id}

    def make_ann_categories(self, classes, kpt_labels=None, kpt_skeletons=None):
        categories = []
        for j, cls in enumerate(classes):
            category = dict()
            category.update({ "supercategory": cls, "id": int(j+1), "name": cls})
            if kpt_labels is not None: category.update({'keypoints': kpt_labels[j]})
            if kpt_skeletons is not None: category.update({'skeleton': kpt_skeletons[j]})
            categories.append(category)
        return categories

    def make_ann_images(self, image_id, im_path, im_size, image_dict_list=[]):
        _, filename = os.path.split(im_path)
        image_dict_list.append({
            "id": int(image_id),
            "license": int(1),
            "coco_url": im_path,
            "flickr_url": "keti.re.kr",
            "width": int(im_size[0]),
            "height": int(im_size[1]),
            "file_name": filename,
            "date_captured": "unknown"
        })
        return image_dict_list

    def binary_mask_to_rle(self, binary_mask):
        from itertools import groupby
        shape = [int(s) for s in binary_mask.shape]
        rle = {'counts': [], 'size': shape}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(int(len(list(elements))))
        return rle

    def make_annotations(self, mask, bbox, ann_id, image_id, cls_id, keypoints=None, annotations=[]):
        h, w = mask.shape[:2]
        rles = cocoMaskUtils.encode(np.asfortranarray(mask))
        area = cocoMaskUtils.area(rles)
        segm = self.binary_mask_to_rle(mask)


        annotation = {
            'segmentation': segm,
            'iscrowd': int(0),
            'image_id': int(image_id),
            'category_id': int(cls_id),
            'id': int(ann_id+1),
            'bbox': [int(p) for p in bbox],
            'area': int(area)
        }
        if keypoints is not None:
            keypoints = [int(p) for p in keypoints]
            annotation.update({'keypoints': keypoints})
        annotations.append(annotation)
        return annotations

    def visualize_instance(self, rgb, instance, categories=None):

        if categories is None: categories=self.categories
        out = np.copy(rgb)
        h, w = rgb.shape[:2]


        # mask
        if 'segmentation' in instance:
            segm = instance['segmentation']
            mask = self.segmToMask(segm, h, w)
            locs = np.where(mask > 0)
            out[locs] = 0.7 * out[locs] + (0, 75, 0)

        # bbox
        if 'bbox' in instance:
            bbox = instance['bbox']
            left, top, w, h = np.array(bbox).astype('int')
            cv2.rectangle(out, (left, top), (left + w, top + h), (0, 255, 0), 2)

            category_id = instance['category_id']
            if category_id is not None:
                for cat in self.categories:
                    if cat['id'] == category_id:
                        cv2.putText(out, cat['name'], (left, top), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255), 2)

        if 'keypoints' in instance:
            kpts = instance['keypoints']
            cat = categories[0]
            kpt_labels = cat['keypoints']
            kpt_skeleton = cat['skeleton']

            # keypoint
            X, Y, V = kpts[::3], kpts[1::3], kpts[2::3]
            for x, y, v in zip(X, Y, V):
                if v == 0: continue
                cv2.drawMarker(out, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 5, 2)

            # skeleton
            for link in kpt_skeleton:
                i1, i2 = link[0] - 1, link[1] - 1
                if V[i1] == 0 or V[i2] == 0: continue
                x1, y1, x2, y2 = X[i1], Y[i1], X[i2], Y[i2]
                cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return out

    def save(self, ann_path, images=None, annotations=None, categories=None, ann_info=None, ann_license=None):
        from ketisdk.utils.proc_utils import ProcUtils
        import json
        if ann_info is None:

            ann_info = {"info": {
                "description": "KETI Dataset",
                "url": "keti.re.kr",
                "version": "1.0",
                "year": int(ProcUtils().get_current_time_str('%Y')),
                "contributor": "Trung Bui",
                "data_create": '{}/{}/{}'.format(ProcUtils().get_current_time_str('%Y'),
                                                 ProcUtils().get_current_time_str('%m'),
                                                 ProcUtils().get_current_time_str('%d'))
            }}

        if ann_license is None:
            ann_license = {"licenses": [
                {"url": "keti.re.kr",
                 "id": "1",
                 "name": "Atribution license"
                 }]}

        if images is None: images = self.image_dict_list
        if categories is None: categories = self.categories
        if annotations is None: annotations = self.annotation_dict_list

        ann_dict = dict()
        ann_dict.update(ann_info)
        ann_dict.update(ann_license)
        ann_dict.update({"images": images})
        ann_dict.update({"categories": categories})
        ann_dict.update({"annotations": annotations})

        save_dir, _ = os.path.split(ann_path)
        os.makedirs(save_dir, exist_ok=True)
        instance_json_obj = open(ann_path, 'w')
        instance_json_obj.write(json.dumps(ann_dict))
        instance_json_obj.close()

        print('{} {} saved'.format('+'*10, ann_path))

    def show_instances(self, im_dir, title='coco_viewer',im_size=(1080, 720)):

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, im_size[0], im_size[1])

        for j,instance in enumerate(self.annotation_dict_list):
            print('ann_ind: {}'.format(j))
            rgb = self.get_im_from_ind(image_id=instance['image_id'], im_dir=im_dir)
            out = self.visualize_instance(rgb, instance)
            cv2.imshow(title, out[:, :, ::-1])
            if cv2.waitKey( ) == 27: exit()

    def show_ims(self, im_dir, title='coco_viewer',im_size=(1080, 720)):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, im_size[0], im_size[1])
        num_im = len(self.image_dict_list)
        for j,im_dict in enumerate(self.image_dict_list):
            im_id = im_dict['id']
            im_path = os.path.join(im_dir, im_dict['file_name'])
            print(f'[{j}/{num_im}] {im_path}')
            out = np.copy(cv2.imread(im_path)[:,:,::-1])
            for ann_dict in self.annotation_dict_list:
                if im_id != ann_dict['image_id']: continue
                out = self.visualize_instance(out, ann_dict)
            cv2.imshow(title, out[:, :, ::-1])
            if cv2.waitKey() == 27: exit()

    def aug_single(self,im_path, im_id, bg_ims=None, angle_step=10, show_step=False):
        from scipy.ndimage import rotate
        from shutil import copyfile
        from ketisdk.utils.proc_utils import ProcUtils
        # makedir
        im_dir = os.path.split(im_path)[0]
        root_dir, dir_name = os.path.split(im_dir)
        save_dir = os.path.join(root_dir, f'{dir_name}_aug')
        os.makedirs(save_dir, exist_ok=True)

        # read image
        im = cv2.imread(im_path)[:, :, ::-1]
        im_height, im_width = im.shape[:2]
        if show_step: cv2.imshow('im', im[:,:,::-1])


        # get instances
        instances = [instance for instance in self.annotation_dict_list if im_id == instance['image_id']]

        for angle in range(0,360, angle_step):
            im_out_path = os.path.join(save_dir, ProcUtils().get_current_time_str() + '.png')

            # make image
            if angle==0:
                copyfile(im_path, im_out_path)
                im_rot = np.copy(im)
            else:
                im_rot = np.copy(rotate(im, angle=angle, reshape=False, order=3))
                cv2.imwrite(im_out_path,im_rot[:,:,::-1])

            self.make_ann_images(self.image_id,im_out_path,(im_width,im_height), self.out_images)

            # make annotation
            for instance in instances:
                mask, bbox = self.segmToMask(instance['segmentation'], h=im_height, w=im_width), instance['bbox']
                if angle !=0:
                    mask = rotate(mask,angle=angle, reshape=False, order=0)
                    Y,X = np.where(mask>0)
                    if len(Y)==0: continue
                    x,y = np.amin(X), np.amin(Y)
                    w, h = np.amax(X)-x, np.amax(Y)-y
                    bbox = [x,y,w,h]

                if show_step:
                    locs = np.where(mask>0)
                    im_rot[locs] = 0.7*im_rot[locs] +  (0,75,0)
                    cv2.rectangle(im_rot, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255,0,0), 2)
                    # cv2.rectangle(im_rot,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
                    cv2.imshow('im_rot', im_rot[:,:,::-1])
                    cv2.waitKey()

                self.make_annotations(mask, bbox, self.ann_id, self.image_id, instance['category_id'],annotations=self.out_annotations)
                self.ann_id +=1

            self.image_id+=1

    def augmentation(self,im_dir, ann_path, bg_dir=None, angle_step=10, show_step=False):
        if not os.path.exists(im_dir):
            print(f'{"+"*10} {im_dir} not exist')
            return

        # read background images
        if bg_dir is not None:
            from glob import glob
            bgs = [cv2.imread(path)[:,:,::-1] for path in glob(os.path.join(bg_dir, '*'))]

        self.out_annotations, self.out_images = [], []
        self.image_id, self.ann_id = 1,1
        num_im = len(self.image_dict_list)
        # query images
        for j,im_dict in enumerate(self.image_dict_list):
            im_id = im_dict['id']
            im_path = os.path.join(im_dir, im_dict['file_name'])
            print(f'[{j}/{num_im}] {im_path}')

            self.aug_single(im_path=im_path,im_id=im_id, angle_step=angle_step, show_step=show_step)

        # save annotation

        self.save(ann_path,images=self.out_images, annotations=self.out_annotations)

    def split_trainval(self, div=(2,1)):

        dur = div[0]+div[1]
        train_im_dict_list, val_im_dict_list = [], []
        train_ann_dict_list, val_ann_dict_list = [], []

        # split
        num_im = len(self.image_dict_list)
        for j, im_dict in enumerate(self.image_dict_list):
            if (j%100==0): print(f'[{j}/{num_im}] done')
            instances = [instance for instance in self.annotation_dict_list if im_dict['id'] == instance['image_id']]

            totrain = (j%dur<div[0])
            if totrain:
                train_im_dict_list.append(im_dict)
                train_ann_dict_list += instances
            else:
                val_im_dict_list.append(im_dict)
                val_ann_dict_list += instances


        #save
        self.save(ann_path=self.ann_path.replace('.json', '_train.json'),images=train_im_dict_list,
                  annotations=train_ann_dict_list, categories=self.categories)
        self.save(ann_path=self.ann_path.replace('.json', '_val.json'), images=val_im_dict_list,
                  annotations=val_ann_dict_list, categories=self.categories)























