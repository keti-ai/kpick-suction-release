
# from libs.dataset.polygon import *
# from libs.basic.basic_sequence import SeqAccumulator
# from libs.basic.basic_objects import BasObj
# from libs.import_basic_utils import *
import pickle
import numpy as np
import cv2
import os
from ketisdk.vision.base.base_objects import DetGuiObj

class Cifar10Maker(DetGuiObj):

    def acc_sum(self):
        super().acc_sum()
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
        num_vis = self.args.input_shape[0]*self.args.input_shape[1]*self.args.input_shape[2]  # 3*32*32
        label_names = self.args.classes
        num_cases_per_batch = self.acc['total']
        meta_dict = {'num_vis': num_vis,
                     'label_names': label_names,
                     'num_cases_per_batch': num_cases_per_batch}
        with open(os.path.join(out_dir, 'batches.meta'), "wb") as f:
            pickle.dump(meta_dict, f)


    def to_train(self, count):
        dur = self.args.train_val_div[0] + self.args.train_val_div[1]
        return (count % dur) < self.args.train_val_div[0]

    def init_acc(self):
        self.num_classes = len(self.args.classes)
        self.acc={'cls_inds': np.zeros((self.num_classes, ), np.uint32)}
        # self.list_to_write.append('cls_inds')

        self.acc.update({'train_array':[], 'test_array':[],
                'train_filenames':[], 'test_filenames':[],
                'train_labels':[], 'test_labels': []})


    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', detected=None):

        if data is None:
            if rgbd is None: rgbd=self.rgbd
            rgbd = rgbd.resize(self.args.input_shape[:2])
            if self.args.show_steps: rgbd.show(title='rgbd_resize',args=self.args)

            data = rgbd.array(get_rgb=self.args.get_rgb, get_depth=self.args.get_depth, depth2norm=self.args.depth2norm)
        if self.args.show_steps: cv2.imshow('depth', data[:,:,3:])
        data_shape = data.shape
        if len(data_shape)<3: data = np.expand_dims(data, axis=2)
        h, w, num_ch = data.shape
        data_1D_org = [data[:,:,ch].reshape((1, h * w)) for ch in range(num_ch)]

        org_color_order = list(range(num_ch))
        color_orders = [org_color_order,]

        if hasattr(self.args, 'aug_color_orders') and self.args.get_rgb:
            for color_order in self.args.aug_color_orders:
                color_orders.append(list(color_order) + org_color_order[3:])

        for color_order in color_orders:
            data_1D = [data_1D_org[ch] for ch in color_order]
            data_1D = np.hstack(data_1D)

            data_reorder = data[:,:, color_order]

            # for ch in range(data.shape[2]):
            #     ch_1D = data[:,:,ch].reshape((1, h * w))
            #     if data_1D is None: data_1D = ch_1D
            #     else: data_1D = np.concatenate((data_1D, ch_1D), axis=1)


            if  hasattr(self,'cls_ind'): cls_ind = self.cls_ind
            else: cls_ind = -1

            self.acc['cls_inds'][cls_ind] += 1

            # update
            if self.to_train(self.acc['total']):
                self.acc['train_array'].append(data_1D)
                self.acc['train_filenames'].append(self.filename)
                self.acc['train_labels'].append(cls_ind)

                num_train = self.acc['num_train']
                self.acc['mean'] = num_train/(num_train+1)*self.acc['mean'] +\
                              1/(num_train+1)*np.mean(data_reorder.astype('float')/255, axis=(0, 1))
                self.acc['std'] = num_train / (num_train + 1) * self.acc['std'] + \
                              1 / (num_train + 1) * np.std(data_reorder.astype('float')/255, axis=(0, 1))
                self.acc['num_train'] += 1
            else:
                self.acc['test_array'].append(data_1D)
                self.acc['test_filenames'].append(self.filename)
                self.acc['test_labels'].append(cls_ind)

                self.acc['num_test'] += 1

            self.acc['total'] = self.acc['num_train'] + self.acc['num_test']

        if self.args.show_steps:
            if cv2.waitKey() ==27: exit()





class CifarModifier(BasObj):
    def get_array(self, data_path):
        f= open(data_path, 'rb')
        entry = pickle.load(f)
        f.close()
        return entry

    def load_params(self, args):
        super().load_params(args=args)
        self.train_filenames, self.test_filenames=None, None
        self.train_array, self.test_array = None, None
        self.train_labels, self.test_labels = None, None
        self.total = None
    def save_db(self, out_dir=None):

        if out_dir is None: save_db_dir = os.path.join(self.args.combine_dir, 'cifar10')
        else: save_db_dir = os.path.join(out_dir, 'cifar10')

        if not os.path.exists(save_db_dir): os.makedirs(save_db_dir)

        # train
        batch_label = 'training batch'
        data_dict = {'filenames': self.train_filenames,
                     'data': self.train_array,
                     'labels': self.train_labels,
                     'batch_label': batch_label}
        with open(os.path.join(save_db_dir,'data_batch'), "wb") as f:
            pickle.dump(data_dict, f, protocol=4)
        # test
        batch_label = 'testing batch'
        data_dict = {'filenames': self.test_filenames,
                     'data': self.test_array,
                     'labels': self.test_labels,
                     'batch_label': batch_label}
        with open(os.path.join(save_db_dir, 'test_batch'), "wb") as f:
            pickle.dump(data_dict, f, protocol=4)
        # meta
        num_vis = self.args.input_shape[0] * self.args.input_shape[1] * self.args.input_shape[2]  # 3*32*32
        label_names = self.args.classes
        num_cases_per_batch = self.total
        meta_dict = {'num_vis': num_vis,
                     'label_names': label_names,
                     'num_cases_per_batch': num_cases_per_batch}
        with open(os.path.join(save_db_dir, 'batches.meta'), "wb") as f:
            pickle.dump(meta_dict, f)


        # about
        about_db = open(os.path.join(save_db_dir, 'about_this_db'), 'w')
        list_to_write = ['num_train', 'num_test', 'total', 'mean', 'std', 'cls_inds']
        for name in list_to_write:
            value = self.__getattribute__(name)
            about_db.write('%s:\t%s\n' % (str(name), str(value)))
            np.save(os.path.join(save_db_dir, name + '.npy'), value)
        about_db.close()


class CombineCifar10Data(CifarModifier):
    def run(self):
        self.num_train, self.num_test, self.total = 0, 0, 0
        self.mean, self.std = 0,0
        self.cls_inds = 0
        self.train_array, self.test_array = None, None
        self.train_filenames, self.test_filenames = [], []
        self.train_labels, self.test_labels = [], []
        for db_path in self.args.db_paths:
            num_train = int(np.load(os.path.join(db_path, 'cifar10','num_train.npy')))
            num_test = int(np.load(os.path.join(db_path, 'cifar10', 'num_test.npy')))
            total = int(np.load(os.path.join(db_path, 'cifar10', 'total.npy')))
            mean = np.load(os.path.join(db_path, 'cifar10', 'mean.npy'))
            std = np.load(os.path.join(db_path, 'cifar10', 'std.npy'))
            cls_inds = np.load(os.path.join(db_path, 'cifar10', 'cls_inds.npy'))

            self.num_train += num_train
            self.num_test += num_test
            self.total += total
            self.mean += mean*total
            self.std += std*total
            self.cls_inds += cls_inds

            train_entry = self.get_array(os.path.join(db_path, 'cifar10', 'data_batch'))
            test_entry = self.get_array(os.path.join(db_path, 'cifar10', 'test_batch'))

            if self.train_array is None: self.train_array = train_entry['data']
            else: self.train_array = np.concatenate((self.train_array, train_entry['data']), axis=0)
            if self.test_array is None: self.test_array = test_entry['data']
            else: self.test_array = np.concatenate((self.test_array, test_entry['data']), axis=0)

            self.train_filenames += train_entry['filenames']
            self.test_filenames += test_entry['filenames']

            self.train_labels += train_entry['labels']
            self.test_labels += test_entry['labels']

        self.mean /= self.total
        self.std /= self.total

        self.save_db()

class CifarDepth2Norm(CifarModifier):
    def run(self, db_path=None):
        if db_path is None: db_path  = self.args.combine_dir

        self.num_train = int(np.load(os.path.join(db_path, 'cifar10', 'num_train.npy')))
        self.num_test = int(np.load(os.path.join(db_path, 'cifar10', 'num_test.npy')))
        self.total = int(np.load(os.path.join(db_path, 'cifar10', 'total.npy')))
        mean = np.load(os.path.join(db_path, 'cifar10', 'mean.npy'))
        std = np.load(os.path.join(db_path, 'cifar10', 'std.npy'))
        self.cls_inds = np.load(os.path.join(db_path, 'cifar10', 'cls_inds.npy'))

        print('loading data...')
        train_entry = self.get_array(os.path.join(db_path, 'cifar10', 'data_batch'))
        test_entry = self.get_array(os.path.join(db_path, 'cifar10', 'test_batch'))

        train_array = train_entry['data']
        test_array = test_entry['data']

        self.train_filenames = train_entry['filenames']
        self.test_filenames = test_entry['filenames']

        self.train_labels = train_entry['labels']
        self.test_labels = test_entry['labels']

        print('computing ...')
        h,w,ch = self.args.input_shape

        depth_train_array = train_array[:,-h*w:]
        depth_test_array = test_array[:,-h*w:]

        norm_train_array = [self.depth2norm1D(depth_train_array[i, :].reshape((h,w)))
                             for i in range(depth_train_array.shape[0])]
        norm_train_array = np.vstack(norm_train_array)

        norm_test_array = [self.depth2norm1D(depth_test_array[i, :].reshape((h, w)))
                             for i in range(depth_test_array.shape[0])]
        norm_test_array = np.vstack(norm_test_array)

        self.train_array = np.concatenate((train_array[:,-h*w], norm_train_array), axis=1)
        self.test_array = np.concatenate((train_array[:,-h * w], norm_test_array), axis=1)

        self.mean = mean + tuple(np.mean(norm_train_array[:, i:(i+1)*h*w]/255.) for i in range(3))
        self.std = std + tuple(np.std(norm_test_array[:, i:(i + 1) * h * w]/255.) for i in range(3))

        print('saving ...')
        self.save_db(out_dir=os.path.join(self.args.combine_dir + '_norm_vector'))


    def depth2norm1D(self, depth):
        norm_map = ArrayUtils().get_mat_normal_map_U8(depth)

        if self.args.show_steps:
            cv2.imshow('norm_map', norm_map)
            cv2.waitKey()
        h,w = self.args.input_shape[:2]
        return np.hstack([norm_map[:,:,i].reshape(1,h*w) for i in range(3)])

















