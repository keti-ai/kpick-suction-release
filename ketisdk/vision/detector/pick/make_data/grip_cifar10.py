# from libs.dataset.polygon import *
# from libs.dataset.cifar10 import Cifar10Maker, CombineCifar10Data
# from libs.dataset.polygon import OnMousePolygonDrawer, PolygonShower, PolygonAnalyzer
# from libs.import_basic_utils import *
import json
import numpy as np
import math
from ketisdk.vision.base.base_objects import DetGuiObj
import os
from ketisdk.import_basic_utils import *
import cv2
import pickle

class GripCifar10DMaker(DetGuiObj):

    def load_ann(self,json_path):
        assert ProcUtils().isexists(json_path)
        json_file = open(json_path)
        ann_dict = json.load(json_file)
        json_file.close()
        return ann_dict

    def show_poly(self, im, classes, polygons):
        for cls, poly in zip(classes,polygons):
            cv2.line(im, tuple(poly[:2]), tuple(poly[-2:]), ProcUtils().get_color(self.args.classes.index(cls)),
                     self.args.line_thick)
        return im

    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', detected=None):

        # read polygon from json
        name, _ = os.path.splitext(filename)
        ann_dict = self.load_ann(os.path.join(self.args.root_dir, self.args.ann_poly,name+'.json'))
        classes = ann_dict['Classes']
        polygons = ann_dict['Polygons']
        polygons = np.array(polygons).reshape((-1,4))

        if self.args.show_steps:
            cv2.namedWindow('viewer', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('viewer', 1080, 720)

        if self.args.show_steps:
            out0 = self.show_poly(rgbd.disp(mode=disp_mode), classes, polygons)
            cv2.imshow('viewer', out0[:,:,::-1])
            cv2.waitKey()


        # pad
        left, top, right, bottom, diag = rgbd.get_diag_pad_params()
        center = (diag//2, diag//2)
        rgbd = rgbd.pad(top=top, left=left, bottom=bottom, right=right)
        polygons[:,[0,2]] += left
        polygons[:,[1,3]] += top
        if self.args.show_steps:
            out1 = self.show_poly(rgbd.disp(mode=disp_mode), classes, polygons)
            cv2.imshow('viewer', out1[:,:,::-1])
            cv2.waitKey()

        # rotate
        count=0
        for cls,poly in zip(classes, polygons):
            x1,y1,x2,y2 = poly
            # compute angle
            dx, dy = x1- x2, y1 - y2
            angle = math.atan(dy / (dx + 0.000001)) * 180 / math.pi
            rgbd_rot = rgbd.rotate(angle=angle)
            # new pt (y1=y2)
            x1, y1 = ProcUtils().rotateXY(x1, y1, -angle, org=center)
            x2, y2 = ProcUtils().rotateXY(x2, y2, -angle, org=center)
            if self.args.show_steps:
                out2 = rgbd_rot.disp(mode=disp_mode)
                cv2.line(out2, (x1,y1), (x2, y2), ProcUtils().get_color(self.args.classes.index(cls)),
                         self.args.line_thick)
                cv2.imshow('viewer', out2[:, :, ::-1])
                cv2.waitKey()

            # crop
            xmin, xmax = min(x1, x2), max(x1, x2)
            grip_hs, grip_w_margins = self.args.train_grip_hs, self.args.train_grip_w_margins
            for grip_h in grip_hs:
                for grip_w_margin in grip_w_margins:
                    count += 1
                    r = int(grip_h/2)
                    rgbd_crop=rgbd_rot.crop(left=xmin-grip_w_margin, top=y1-r, right=xmax+1+grip_w_margin, bottom=y1+r+1)

                    if self.args.show_steps:
                        rgbd_crop.show(title='rgbd_crop')
                        cv2.waitKey()
                    # self.cls_ind = self.args.classes.index(cls)
                    self.do_acc(rgbd_roi=rgbd_crop, cls_ind=self.args.classes.index(cls), filename=filename)
        print(f'>>> {count} samples ...')

    def to_train(self, count):
        dur = self.args.train_val_div[0] + self.args.train_val_div[1]
        return (count % dur) < self.args.train_val_div[0]

    def init_acc(self):
        self.num_classes = len(self.args.classes)
        self.acc = {'mean': 0.0, 'std': 0.0, 'num_train': 0, 'num_test': 0, 'total': 0}
        self.list_to_write = ['mean', 'std', 'num_train', 'num_test', 'total', 'cls_inds']
        self.acc.update({'cls_inds': np.zeros((self.num_classes,), np.uint32)})

        self.acc.update({'train_array': [], 'test_array': [],
                         'train_filenames': [], 'test_filenames': [],
                         'train_labels': [], 'test_labels': []})

    def do_acc(self, rgbd_roi, cls_ind=-1, filename='unnamed'):
        data = rgbd_roi.resize(self.args.input_shape[:2]).\
            array(get_rgb=self.args.get_rgb, get_depth=self.args.get_depth, depth2norm=self.args.depth2norm)
        data_shape = data.shape
        if len(data_shape) < 3: data = np.expand_dims(data, axis=2)
        h, w, num_ch = data.shape
        data_1D_org = [data[:, :, ch].reshape((1, h * w)) for ch in range(num_ch)]
        org_color_order = list(range(num_ch))
        color_orders = [org_color_order, ]

        if hasattr(self.args, 'aug_color_orders') and self.args.get_rgb:
            for color_order in self.args.aug_color_orders:
                color_orders.append(list(color_order) + org_color_order[3:])

        for color_order in color_orders:
            data_1D = [data_1D_org[ch] for ch in color_order]
            data_1D = np.hstack(data_1D)
            data_reorder = data[:, :, color_order]


            self.acc['cls_inds'][cls_ind] += 1
            # update
            if self.to_train(self.acc['total']):
                self.acc['train_array'].append(data_1D)
                self.acc['train_filenames'].append(filename)
                self.acc['train_labels'].append(cls_ind)

                num_train = self.acc['num_train']
                self.acc['mean'] = num_train / (num_train + 1) * self.acc['mean'] + \
                                   1 / (num_train + 1) * np.mean(data_reorder.astype('float') / 255, axis=(0, 1))
                self.acc['std'] = num_train / (num_train + 1) * self.acc['std'] + \
                                  1 / (num_train + 1) * np.std(data_reorder.astype('float') / 255, axis=(0, 1))
                self.acc['num_train'] += 1
            else:
                self.acc['test_array'].append(data_1D)
                self.acc['test_filenames'].append(filename)
                self.acc['test_labels'].append(cls_ind)

                self.acc['num_test'] += 1

            self.acc['total'] = self.acc['num_train'] + self.acc['num_test']

        if self.args.show_steps:
            if cv2.waitKey() == 27: exit()

    def finalize_acc(self):
        out_dir = os.path.join(self.args.root_dir, self.args.traindb_dir)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        # about
        about_db = open(os.path.join(self.args.root_dir, 'about_this_db'), 'w')
        for name in self.acc:
            if name not in self.list_to_write: continue
            about_db.write('%s:\t%s\n' % (str(name), str(self.acc[name])))
            np.save(os.path.join(out_dir, name + '.npy'), self.acc[name])
        about_db.close()


        out_dir = os.path.join(self.args.root_dir, self.args.traindb_dir)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        # train
        data = np.vstack(self.acc['train_array'])
        batch_label = 'training batch'
        data_dict = {'filenames': self.acc['train_filenames'],
                     'data': data,
                     'labels': self.acc['train_labels'],
                     'batch_label': batch_label}
        with open(os.path.join(out_dir, 'data_batch'), "wb") as f:
            pickle.dump(data_dict, f)
        # test
        data = np.vstack(self.acc['test_array'])
        batch_label = 'testing batch'
        data_dict = {'filenames': self.acc['test_filenames'],
                     'data': data,
                     'labels': self.acc['test_labels'],
                     'batch_label': batch_label}
        with open(os.path.join(out_dir, 'test_batch'), "wb") as f:
            pickle.dump(data_dict, f)
        # meta
        num_vis = self.args.input_shape[0] * self.args.input_shape[1] * self.args.input_shape[2]  # 3*32*32
        label_names = self.args.classes
        num_cases_per_batch = self.acc['total']
        meta_dict = {'num_vis': num_vis,
                     'label_names': label_names,
                     'num_cases_per_batch': num_cases_per_batch}
        with open(os.path.join(out_dir, 'batches.meta'), "wb") as f:
            pickle.dump(meta_dict, f)



def get_grip_CIFAR_gui(cfg_path):
    from ketisdk.gui.gui import GUI, GuiModule

    module = GuiModule(GripCifar10DMaker, name='Grip CIFAR', cfg_path=cfg_path)
    GUI(title='MAKE GRIP CIFAR DATA', modules=[module,])



if __name__ =='__main__':
    cfg_path = 'configs/pick/grip_net.cfg'
    get_grip_CIFAR_gui(cfg_path=cfg_path)






