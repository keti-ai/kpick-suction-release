import random
import numpy as np
from matplotlib import pyplot as plt
import configparser
from datetime import datetime
import imghdr
from matplotlib.path import Path
import json
import argparse

import cv2
import os
import math
from scipy import optimize
import copy
import platform
from copy import deepcopy
from shutil import copyfile

# import pyrealsense2 as rs
import shutil
import csv


color_base_list = [(0,0,255), (0,255,0), (255, 0, 0), (0,255,255), (255,255,0), (255,0,255)]
ncolor_base = len(color_base_list)

key_base_list = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
                 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
                 'z', 'x', 'c', 'v', 'b', 'n']

class BasDataObj():
    def fromdict(self, data_dict):
        for attr in data_dict:
            self.__setattr__(attr, data_dict[attr])

    def todict(self):
        data_dict = dict()
        for name in self.__dir__():
            value = self.__getattribute__(name)
            if callable(value) or '__' in name: continue
            data_dict.update({name: value})
        return data_dict


class ProcUtils():
    def read_csv(self, path):
        csvFile = open(path, 'r')
        reader = csv.reader(csvFile)
        ret = [row for row in reader]
        csvFile.close()
        return ret


    def open_dir(self, path):
        if not os.path.isdir(path):
            print('{} is not dir'.format(path))
            return
        os.system('nautilus {}'.format(path))

    def imshow(self, im, title='image', size=(720, 480)):
        try:
            cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, size[0], size[1])
        except: pass
        cv2.imshow(title, im)


    def is_exist(self, filepath):
        if filepath is None: return False
        if not os.path.exists(filepath):
            print('{} does not exist'.format(filepath))
            return False
        else: return True

    def rmdir(self, path):
        if os.path.exists(path): shutil.rmtree(path)

    def remove_confile_section(self,cfg_path, sections, out_path):
        parser = configparser.RawConfigParser()
        parser.read(cfg_path)
        for section in sections:
            parser.remove_section(section)
        with open(out_path, 'w') as configfile: parser.write(configfile)

    def select_confile_section(self,cfg_path, sections, out_path):
        parser = configparser.RawConfigParser()
        parser.read(cfg_path)
        for sec in parser.sections():
            if sec in sections: continue
            parser.remove_section(sec)
        with open(out_path, 'w') as configfile: parser.write(configfile)

    def replace_confile_value(self,cfg_path, section, option, new_value):
        parser = configparser.RawConfigParser()
        # parser.optionxform = str
        parser.read(cfg_path)
        parser.set(section=section, option=option, value=new_value)

        with open(cfg_path, 'w') as configfile: parser.write(configfile)
        # found = False
        # for sec in parser.sections():
        #     if sec != section: continue
        #     for opt in parser.options(sec):
        #         if opt != option: continue
        #         found = True
        #         value = parser.get(section, option)
        #         parser.set(section, )

    def read_txt_file(self, filepath):
        text = ''
        f = open(filepath, 'r')
        for line in f: text += line
        return text

    # def get_realsense_serials(self):
    #     realsense_ctx = rs.context()
    #     return [realsense_ctx.devices[i].get_info(rs.camera_info.serial_number) for i in range(len(realsense_ctx.devices))]



    def cfgfile2text(self, cfg_path, sections=None, excepts=None):
        parser = configparser.RawConfigParser()
        parser.optionxform = str
        parser.read(cfg_path)
        text = ''
        for section in parser.sections():
            if sections is not None:
                if section not in sections: continue
            if excepts is not None:
                if section  in excepts: continue

            for option in parser.options(section):
                value = parser.get(section, option)
                text += '%s = %s\n\n' %(option, value)

        return text

    def confile2singleValue(self, cfg_path, section, option):
        parser = configparser.RawConfigParser()
        parser.optionxform = str
        parser.read(cfg_path)
        value = None
        for sec in parser.sections():
            if sec != section: continue
            for opt in parser.options(sec):
                if opt != option: continue
                value = parser.get(section, option)
        return value

    def init_vid_writer(self, size, fold=None, name=None, fps=15, isColor=True):
        if name is None: name = self.get_current_time_str() + '.avi'
        if fold is None: fold = ''
        if not os.path.exists(fold): os.makedirs(fold)
        vid_path = os.path.join(fold, name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        return  cv2.VideoWriter(vid_path, fourcc=fourcc,fps=fps, frameSize=size, isColor=isColor)

    def get_color(self, n, dtype='uint8', scale=1):
        """ get n-th color. if n<=5, return color_base_list[n]. Otherwise, return a random combination of 2 base colors"""

        if n >= ncolor_base:  # random combine 2 colors
            twocolors = random.sample(color_base_list, k=2)
            alpha = random.uniform(0.1, 0.9)
            color = scale*alpha * np.array(twocolors[0]) + scale*(1 - alpha) * np.array(twocolors[1])
            color = tuple(color.astype(dtype))
        else:
            color = scale*np.array(color_base_list[n])
            color = tuple(color.astype(dtype))

        if dtype != 'uint8':
            vmin, vmax = np.iinfo(dtype).min, np.iinfo(dtype).max
            vscale = vmax / np.iinfo('uint8').max
            if vscale != 1: color = tuple(int(el * vscale) for el in color)


        return (int(color[0]), int(color[1]), int(color[2]))

    def get_color_list(self, n, dtype='uint8'):
        """
        :returns: list of n colors
        """
        return [self.get_color(i, dtype=dtype) for i in range(n)]


    def cls2colordict(self, classes):
        """
        :returns: dict of keys corresponding to classes. Each key connect with a class and a color
        """
        nclass = len(classes)
        key_dict = dict()
        for i in range(nclass):
            if i<9: key = str(i+1)
            else: key = key_base_list[i-9]
            key_dict.update({key: {'cls': classes[i], 'color': self.get_color(i)}})

        key_dict.update({'0': {'cls': None, 'color': (0,0,0)}})
        return key_dict

    def get_command_keys(self, commands):
        cmd_dict = dict()
        for i, cmd in enumerate(commands):
            if i<9: key = str(i+1)
            else: key = key_base_list[i-9]
            cmd_dict.update({key: cmd})
            print('%s: \t %s' % (key, cmd))
        return cmd_dict

    def get_key_dict(self, classes):
        """
        :returns: dict of keys corresponding to classes. Each key connect with a class and a color
        """
        nclass = len(classes)
        key_dict = dict()
        for i in range(nclass):
            if i < 9:
                key = str(i + 1)
            else:
                key = key_base_list[i - 9]
            key_dict.update({key: {'cls': classes[i], 'color': self.get_color(i)}})

        key_dict.update({'0': {'cls': None, 'color': (0, 0, 0)}})
        return key_dict

    def change_cmd_dict(self, cmd_dict, prev, lat):
        for kk in cmd_dict:
            if cmd_dict[kk] != prev: continue
            cmd_dict[kk] = lat
        return cmd_dict

    def rotate2D(self, pt, theta, org):
        """ rotate a point with a `theta` degree and origin `org`

        :param pt: input point (y,x)
        :type pt: tuple
        :param theta: angle
        :type theta: float
        :param org: rotate origin (y0, x0)
        :type org: tuple
        :returns: rotated point (y,x)
        """
        theta = theta * math.pi / 180
        cost = math.cos(theta)
        sint = math.sin(theta)

        y = pt[0] - org[0]
        x = pt[1] - org[1]

        return (int(org[0] + y * cost + x * sint), int(org[1] - y * sint + x * cost))

    def rotateXY_float(self, x, y, theta, org=(0, 0)):
        # pt = (y,x)
        theta = theta * math.pi / 180
        cost = math.cos(theta)
        sint = math.sin(theta)

        y -= org[1]
        x -= org[0]
        return (org[0] - y * sint + x * cost, org[1] + y * cost + x * sint)


    def rotateXY(self, x, y, theta, org=(0, 0)):
        xo, yo = self.rotateXY_float(x,y,theta, org)
        return (int(xo), int(yo))

    def rotateXY_array(self,X, Y, Theta, org=(0,0)):
        Theta_ = np.copy(Theta)*math.pi/180
        cosT, sinT  =np.cos(Theta_), np.sin(Theta_)
        X_, Y_ = X-org[0], Y-org[1]
        return (org[0] - np.multiply(Y_, sinT) + np.multiply(X_, cosT),
                org[1] + np.multiply(Y_, cosT) + np.multiply(X_, sinT))


    def bbox_iou(self, boxA, boxB):
        """ calculate  IoU of 2 boxes

        :param boxA: (left, top, right, bottom)
        :type boxA: tuple, list
        :param boxB: (left, top, right, bottom)
        :type boxB: tuple, list
        :returns iou:
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def bbox_max_iou(self, boxA, boxB):
        """ calculate max  IoU of 2 boxes

        :param boxA: (left, top, right, bottom)
        :type boxA: tuple, list
        :param boxB: (left, top, right, bottom)
        :type boxB: tuple, list
        :returns iou:
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(min(boxAArea, boxBArea))

        # return the intersection over union value
        return iou


    def box_relates(self, box1, box2):

        left1, top1, right1, bottom1 = box1
        left2, top2, right2, bottom2 = box2

        lefti, topi = max(left1, left2), max(top1, top2)
        righti, bottomi = min(right1, right2), min(bottom1, bottom2)

        xc1, yc1 = (right1 - left1)//2, (bottom1-top1)//2
        xc2, yc2 = (right2-left2)//2, (bottom2-top2)//2

        interArea = max(0, righti - lefti + 1) * max(0, bottomi - topi + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        area1 = (right1-left1 + 1) * (bottom1 - top1 + 1)
        area2 = (right2 - left2 + 1) * (bottom2 - top2 + 1)

        iou = interArea / float(area1 + area2 - interArea)

        if iou==0: dis = 0
        else: dis = np.linalg.norm((xc2-xc1, yc2-yc1))
        return  area1, area2, iou, dis

    def check_overlap_boxes_list(self, box0, boxes):
        """ check whether a box averlap with list of boxes

        :param box0: (left, top, right, bottom)
        :type box0: tuple, list
        :params boxes: list of boxes
        :returns is_overlap, and index of which closest to `box0`
        """
        close_to = -1
        is_overlap = False
        smallest_iou = 1
        for i in range(len(boxes)):
            box = boxes[i]
            if box[4] < 0.5: continue
            iou = self.bbox_iou(box0, box[0:4])
            if iou > 0:
                is_overlap = True
            if iou <= smallest_iou:
                smallest_iou = iou
                close_to = i
        return is_overlap, close_to

    def nms_boxes(self, boxes, scores):
        """ reduce the redundancy of boxes"""
        new_boxes = []

        for box, score in zip(boxes, scores):
            if score < 0.65: continue
            new_boxes.append(box)
            if len(new_boxes) == 0:
                new_boxes.append(box)
                continue
            ious = []
            for nbox in new_boxes:
                ious.append(self.bbox_iou(box[0:4], nbox[0:4]))
            ious_array = np.array(ious)
            max_iou = np.max(ious_array)
            if max_iou > 0.6:
                argmax_iou = int(np.argmax(ious_array))
                new_boxes[argmax_iou] = box
            else:
                new_boxes.append(box)

        new_boxes = np.vstack(new_boxes)
        return new_boxes

    def fix_size(self, im, size_out=(105, 105), fill=True, val=255):
        """ fix image to a fix size"""

        if size_out == (0, 0):
            return im

        h, w = im.shape[:2]
        h_o, w_o = size_out

        if len(im.shape) > 2:
            value = (val, val, val)
        else:
            value = val

        if (h, w) == (h_o, w_o):
            return im

        fx, fy = w_o / w, h_o / h

        if fx < fy:
            w_s, h_s = w_o, int(fx * h)  # h_s<h_o

            im_s = cv2.resize(im, dsize=(w_s, h_s), interpolation=cv2.INTER_CUBIC)
            if not fill: return im_s

            left, right = 0, 0
            top = int((h_o - h_s) / 2)
            bottom = h_o - (h_s + top)
            out = cv2.copyMakeBorder(im_s, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=value)


        elif fy < fx:
            w_s, h_s = int(fy * w), h_o  # w_s<w_o

            im_s = cv2.resize(im, dsize=(w_s, h_s), interpolation=cv2.INTER_CUBIC)
            if not fill: return im_s

            top, bottom = 0, 0
            left = int((w_o - w_s) / 2)
            right = w_o - (w_s + left)
            out = cv2.copyMakeBorder(im_s, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=value)
        else:
            out = cv2.resize(im, dsize=(w_o, h_o), interpolation=cv2.INTER_CUBIC)

        return out

    def resize_im(self, im, size, mode='fit_pad', pad_val=255):
        '''
        mode: 'fit_crop', 'fit_pad'
        '''
        h, w = im.shape[:2]
        scale, pad = 1, (0,0,0,0)
        if (w,h)==size: return {'im': im, 'scale': scale, 'pad': pad}

        wo, ho = size
        fx, fy = wo/w, ho/h

        if fx==fy: return {'im': cv2.resize(im, dsize=size, interpolation=cv2.INTER_CUBIC), 'scale': fx, 'pad': pad}

        color_im = (len(im.shape)>2)
        if color_im: pad_val = (pad_val, pad_val, pad_val)

        strech_x_more = (fx>fy)
        stretch_x_first = (not strech_x_more and mode=='fit_crop')  or (strech_x_more and mode=='fit_crop')
        if stretch_x_first: scale, w1, h1 = fx, wo, int(fx * h)
        else: scale, w1, h1 = fy, int(fy*w),ho

        dw, dh = wo-w1, ho-h1
        pleft, ptop = dw//2, dh//2
        pright, pbottom = dw-pleft, dh-ptop
        pad = (pleft, ptop, pright, pbottom)
        im_scale = cv2.resize(im, dsize=(w1, h1), interpolation=cv2.INTER_CUBIC)

        if mode=='fit_pad':
            out = cv2.copyMakeBorder(im_scale, ptop, pbottom,pleft,pright, borderType=cv2.BORDER_CONSTANT, value=pad_val)
        if mode=='fit_crop':
            out = im_scale[-ptop:h1+pbottom, -pleft: w1+pright, :]

        return {'im': out, 'scale': scale, 'pad': pad}


    def crop_im(self, im, crop_size):
        h,w = im.shape[:2]
        color_im = (len(im.shape) > 2)
        left, top, right, bottom = crop_size
        if color_im: im_crop = im[top: h-bottom, left:w-right, :]
        else: im_crop = im[top: h-bottom, left:w-right]
        return im_crop

    def unresize_im(self, im, size, crop=(0,0,0,0), pad_val=255):
        h, w = im.shape[:2]
        color_im = (len(im.shape) > 2)
        left, top, right, bottom = crop
        to_crop = (left>0) or (top>0) or (right>0) or (bottom>0)

        if to_crop: im_crop = self.crop_im(im, crop)
        else:
            if color_im: pad_val=(pad_val, pad_val, pad_val)
            im_crop = cv2.copyMakeBorder(im,-top, -bottom, -left, -right, cv2.BORDER_CONSTANT, value=pad_val)

        return cv2.resize(im_crop, dsize=size, interpolation=cv2.INTER_CUBIC)
        




    # def clscr(self):
    #     """ clear screen"""
    #     if platform.system() == 'Linux': os.system('clear')
    #     if platform.system() == 'Windows': os.system('cls')

    def clscr(self, actual_clear=False):
        """ clear screen"""
        if actual_clear:
            if platform.system() == 'Linux': os.system('clear')
            if platform.system() == 'Windows': os.system('cls')
        else: print(chr(27) + "[2J")

    def correct_cropsize(self, crop_size, im_size):
        height, width = im_size
        if crop_size is None: return (0, 0, height, width)
        top, left, bottom, right = crop_size
        top, left, bottom, right = max(0, top), max(0, left), min(bottom, height), min(right, width)
        if top >= bottom or left >= right:
            crop_size = (0, 0, height, width)
        else:
            crop_size = (top, left, bottom, right)
        return crop_size


    def get_current_time_str(self, format='%Y%m%d%H%M%S%f'):
        return datetime.now().strftime(format)

    def get_time_name(self, format='%Y%m%d%H%M%S%f'):
        return self.get_current_time_str(format=format)

    def isimpath(self, path):
        if path is None: return False
        if os.path.isdir(path): return  False
        return imghdr.what(path) is not None

    def isexists(self, path):
        if path is None: return False
        return os.path.exists(path)

    def get_dtype_name(self, anyObj):
        return type(anyObj).__name__

    def get_value_from_express(self, express, variable_dict=None):
        if variable_dict is not None:
            for key in variable_dict: locals()[key] = variable_dict[key]
        try:
            out = eval(express)
        except:
            out = express
        if callable(out): out=express       # avoid builtin function

        if isinstance(out, (tuple, list)):
            is_tuple = isinstance(out, tuple)
            out = [self.get_value_from_express(el) for el in out]
            if is_tuple: out = tuple(out)
        if out == 'None': out = None
        return out

    def pt2pt_distance(self, pt1, pt2):
        return np.linalg.norm(np.array(pt1)-np.array(pt2))


    def get_cfg_from_file(self, cfg_path):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()

        parser = configparser.RawConfigParser()
        parser.optionxform = str
        parser.read(cfg_path)

        arg_dict = dict()
        for section in parser.sections():
            for option in parser.options(section):
                value = parser.get(section, option)
                value = ProcUtils().get_value_from_express(value, arg_dict)
                arg_dict.update({option: value})

                args.__setattr__(option, value)

        return args

class Regression():
    def fitting_surface(self, X,Y, Z):
        """ approximate values of a,b,c where Z = aX + bY + c

        :param X, Y, Z:
        :type X, Y, Z: 1-D ndarray
        :returns a,b,c:
        """

        mean_X = np.mean(X)
        mean_Y = np.mean(Y)
        mean_Z = np.mean(Z)

        var_X = np.var(X)
        var_Y = np.var(Y)

        mean_XY = np.mean(np.multiply(X, Y))
        mean_YZ = np.mean(np.multiply(Y, Z))
        mean_ZX = np.mean(np.multiply(Z, X))

        cov_XY = mean_XY - mean_X * mean_Y
        cov_YZ = mean_YZ - mean_Y * mean_Z
        cov_ZX = mean_ZX - mean_Z * mean_X

        A = np.array([[var_X, cov_XY], [cov_XY, var_Y]], dtype=np.float)
        bb = np.array([[cov_ZX], [cov_YZ]], dtype=np.float)

        ab = np.dot(np.linalg.inv(A), bb)
        a = ab[0]
        b = ab[1]

        c = mean_Z - a * mean_X - b * mean_Y
        return a,b,c





class VisUtils():
    def show_surface(self,mats):
        num_mat = len(mats)
        fig = plt.figure(figsize=plt.figaspect(1/num_mat))
        for i in range(num_mat):
            mat = mats[i]
            h,w = mat.shape[:2]
            X,Y = np.meshgrid(np.arange(0,w), np.arange(0,h))
            # fig = plt.figure()
            ax = fig.add_subplot(1, num_mat, i+1, projection='3d')
            # surf = ax.plot_surface(Y, X, mat)
            # fig.colorbar(surf, shrink=1/num_mat, aspect=10)
            #
            ax = plt.gca(projection='3d')
            ax.plot_surface(Y, X, mat)
            ax.set_xlabel('Y')
            ax.set_ylabel('X')
            ax.set_zlabel('Z')
        plt.show()

    def draw_texts_blend(self, im, texts, text_rect=(0, 0, 200, 200), alpha=0.8, up2down=True, colors=(255, 0, 255), scale=1,
                         thick=2, space=30):
        im_size = im.shape
        is_color = len(im_size) == 3
        h, w = im_size[:2]
        top, left, bottom, right = ProcUtils().correct_cropsize((text_rect), (h, w))
        ht, wt = bottom - top, right - left
        text_im = 255 * np.ones((ht, wt, 3), np.uint8)
        if is_color:
            out = np.copy(im)
        else:
            out = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        if not isinstance(texts, (list, tuple)):
            num_text = 1
        else:
            num_text = len(texts)

        if not isinstance(colors[0], (list, tuple)): colors = [colors] * num_text

        if num_text == 1:
            text_im = cv2.putText(text_im, texts[0], (space, space), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                  fontScale=scale, color=colors[0], thickness=thick)
        else:
            if up2down:
                x, y = space, space
                for text, color in zip(texts, colors):
                    text_im = cv2.putText(text_im, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, scale, color, thick)
                    y += space
            else:
                x, y = space, bottom - space
                for text, color in zip(texts, colors):
                    text_im = cv2.putText(text_im, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, scale, color, thick)
                    y -= space
        out[top:bottom, left:right, :] = alpha * text_im + (1 - alpha) * out[top:bottom, left:right, :]
        return out.astype(np.uint8)

    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()




    # int_ = lambda x: int(round(x))


class PolyUtils():

    def polys2lefttops(self, polys):
        lefttops = []
        for poly in polys:
            poly_array = np.array(poly)
            left, top = np.amin(poly_array, axis=0)
            lefttops.append((left, top))
        return lefttops

    def poly2rect(self, poly):
        """ get rectangle bounding points of polygon
        :param poly: list of points
        :returns: rect [left, top, right, bottom]
        """
        poly_array = np.array(poly)

        left, top = np.amin(poly_array, axis=0)
        right, bottom = np.amax(poly_array, axis=0)

        return Rect(pt1=(left, top), pt2=(right, bottom))

    def poly2box(self, poly):
        """ get rectangle bounding points of polygon
        :param poly: list of points
        :returns: rect [left, top, right, bottom]
        """
        poly_array = np.array(poly)

        left, top = np.amin(poly_array, axis=0)
        right, bottom = np.amax(poly_array, axis=0)

        return (left, top, right, bottom)

    def box2poly(self, bbox):
        left, top, right, bottom = bbox
        return [(left, top), (right, top), (right, bottom), (left, bottom)]

    def rect2poly(self, rect):
        left, top = rect.lefttop
        right, bottom = rect.rightbottom


        poly = []
        poly.append((left, top))
        poly.append((right, top))
        poly.append((right, bottom))
        poly.append((left, bottom))

        return poly

    def poly2mask(self, poly, max_shape=None):
        poly_path = Path(poly)
        rect = self.poly2rect(poly)
        if max_shape is None: xmax, ymax = rect.rightbottom
        else: xmax, ymax = max_shape

        y,x = np.mgrid[:ymax, :xmax]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))  # coors.shape is (4000000,2)
        mask = poly_path.contains_points(coors).astype('uint8').reshape((ymax, xmax))

        return mask

    def mask2poly(self, mask):
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        X, Y = contour[:,:,0], contour[:,:,1]
        return [(int(x),int(y)) for x,y in zip(X,Y)]


    def poly2maskpts(self, poly, max_shape=None):
        mask = self.poly2mask(poly, max_shape)
        return MaskUtils().mask2pts(mask)

    def draw_poly(self, im, poly, args=None, color=(0,255,0), thick=2):
        """ draw a polyygon on given image

        :param im: given image
        :type im: ndarray
        :param poly: list of points
        :param color: display line-color
        :param thick: thickness of line
        :returns: drawn image
        """
        if len(poly)==0: return im
        if args is not None: color, thick = args.line_color, args.line_thick
        out = np.copy(im)
        color = (int(color[0]), int(color[1]), int(color[2]))

        if len(poly) == 1:
            return cv2.drawMarker(im, poly[0], color, cv2.MARKER_TILTED_CROSS, 10, 2)

        for i in range(len(poly) - 1):
            pt1, pt2 = tuple(poly[i]), tuple(poly[i + 1])
            cv2.line(out, pt1=pt1, pt2=pt2  , color=color, thickness=thick)
        cv2.line(out, tuple(poly[-1]), tuple(poly[0]), color, thick)
        return out

    def draw_polys(self, im, polys, args=None, colors=None, thick=None):
        """ draw  polygons on given image

        :param im: given image
        :type im: ndarray
        :returns:drawn image
        """
        out = np.copy(im)
        num_poly = len(polys)
        if args is not None: colors, thicks = [args.line_color]*num_poly, [args.line_thick]*num_poly
        if colors is None: colors = [(0,255,0)]*num_poly
        if thick is None: thicks = 2
        for i in range(num_poly):
            out = self.draw_poly(out, polys[i], color=colors[i], thick=thick)

        return out

    def get_cls_color_dict(self, classes):
        classes_unq = list(np.unique(classes))
        colors = ProcUtils().get_color_list(len(classes_unq))
        color_dict = dict()
        for cls, color in zip(classes_unq, colors):
            color_dict.update({cls: color})
        return color_dict


    def draw_polys_from_json(self, json_path, im=None, im_path=None, args=None, text_scale=1.5, text_thick=3,
                          rect=(0, 0, 100, 100), space=10, alpha=0.5, up2down=True):
        """
        :param json_path: json file path
        :param im: ndarray image
        :param im_path: image file path
        :param text_scale: opencv text_scale
        :param text_thick: opencv text_thick
        :param rect: rect location of text display
        :param space: space between text lines
        :param alpha: text transparency
        :param up2down: direct to display text lines
        :returns: ndarray
        """
        assert im is not None or im_path is not None
        if ProcUtils().isimpath(im_path): im = cv2.imread(im_path)

        ann_dict = JsonUtils().read_json(json_path)
        color_dict = self.get_cls_color_dict(classes=ann_dict['Classes'])
        colors = [color_dict[cls] for cls in ann_dict['Classes']]
        out = self.draw_polys(im, polys=ann_dict['Polygons'], colors=colors)




        if args is not None:
            text_scale, text_thick, rect, space, alpha, up2down = args.text_scale, args.text_thick, \
                                                                  args.text_rect, args.text_space, \
                                                                  args.text_alpha, args.text_alpha

        out = VisUtils().draw_texts_blend(out, texts=list(np.unique(ann_dict['Classes'])), colors=colors, scale=text_scale, thick=text_thick,
                               text_rect=rect, space=space, alpha=alpha, up2down=up2down)
        return out

class MaskUtils():
    """Binary mask Utils"""
    def mask2pts(self,mask):
        Y, X = np.where(mask)
        return [(x, y) for x, y in zip(X, Y)]

    def mask2boundlocs(self, mask, r=2):
        s = 2*r+1
        return np.where(mask - cv2.erode(mask, np.ones((s,s), 'uint8')))

    def sparse_mask(self, mask, sparse_mask=None, stride=2, margins=(0,0)):
        if stride <2: return mask

        if sparse_mask is None:
            h,w = mask.shape[:2]
            sparse_mask = np.zeros((h, w), 'uint8')
            mx, my = margins
            sparse_mask[my:h-my:stride, mx:w-mx:stride] = 1
        return np.logical_and(mask, sparse_mask).astype('uint8')

    def showCTmask(self, im, mask):
        mask_rgb = self.inst2rgb(mask)
        mask_loc = np.where(mask!=0)
        out = np.copy(im)
        out[mask_loc] = 0.7*out[mask_loc] + 0.3*mask_rgb[mask_loc]

        boxes, cls_ids, inst_ids = self.inst2boxes(mask)
        for box, cls_ind, inst_id in zip(boxes, cls_ids, inst_ids):
            left, top, right, bottom = box
            out = cv2.putText(out, 'cls:%d_id:%d'%(cls_ind, inst_id), (left, top-5),
                              cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 2)
            out = cv2.rectangle(out,(left, top),(right, bottom), (255,0,0), 2)

        return out


    def inst2color(self, cls_ind, inst_ind, step=0.02):
        return (step, cls_ind * step, inst_ind * step, 1.0)

    def rgb2inst(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        cls_ind = np.round(np.divide(g, r))
        inst_ind = np.round(np.divide(b, r))
        return (cls_ind * 1000 + inst_ind).astype('uint16')

    def inst2rgb(self, mask):
        cls = np.round(mask/1000).astype('int')
        ind = mask - 1000*cls

        num_cls = np.amax(cls)
        max_ind = np.amax(ind)

        step = np.floor(255/max(num_cls, max_ind))

        rgb = np.zeros(mask.shape+(3,))
        rgb[:,:,0] = step
        rgb[:,:,1] = cls*step
        rgb[:,:,2] = ind*step

        return rgb.astype(np.uint8)

    def extract_ints_mask(self, inst_mask, classes=None, small_size=100, denoise_kernel=(3,3)):
        # fg_mask = cv2.erode((inst_mask>0).astype('uint8'), np.ones(denoise_kernel, 'uint8'))
        # inst_mask = np.multiply(inst_mask, fg_mask.astype('uint16'))

        # cv2.imwrite('inst_mask.png', inst_mask)
        denoise_kernel = np.ones(denoise_kernel, 'uint8')

        vals = np.unique(inst_mask)
        label = {'boxes':[], 'classes': [], 'cls_idxs':[],'inst_idxs':[],'locs': []}
        for val in vals:
            if val==0: continue
            val_mask = cv2.morphologyEx((inst_mask==val).astype('uint8'), cv2.MORPH_OPEN, denoise_kernel)

            Y,X = np.where(val_mask>0)
            if len(Y)<small_size: continue
            bbox = (np.amin(X), np.amin(Y), np.amax(X),np.amax(Y))
            cls_idx = val//1000
            if cls_idx>len(classes): continue
            inst_idx = val - 1000*cls_idx
            if classes is None: cls = str(cls_idx)
            else: cls = classes[cls_idx-1]

            label['boxes'].append(bbox)
            label['classes'].append(cls)
            label['inst_idxs'].append(inst_idx)
            label['locs'].append((Y,X))
            label['cls_idxs'].append(cls_idx)
        return label




    def inst2boxes(self, mask):
        cls = np.round(mask / 1000).astype('int')
        ind = mask - 1000 * cls

        num_cls = np.amax(cls)
        max_ind = np.amax(ind)

        boxes, cls_ids, inst_ids = [], [], []
        for cl in range(1, num_cls+1):
            for i in range(max_ind+1):
                Y, X = np.where((cls==cl) & (ind==i))
                if len(Y)==0: continue
                left, top, right, bottom = np.amin(X), np.amin(Y), np.amax(X), np.amax(Y)
                boxes.append((left, top, right, bottom))
                cls_ids.append(cl)
                inst_ids.append(i)
        return (boxes, cls_ids, inst_ids)

class Rect():
    def __init__(self, pt1, pt2):
        """
        :param pt1: first point (x1,y1)
        :param pt2: second point (x2, y2)
        """
        x1, y1 = pt1
        x2, y2 = pt2
        self.left, self.top, self.right, self.bottom = min(x1, x2), min(y1,y2), max(x1,x2), max(y1, y2)
        self.lefttop = (self.left, self.top)
        self.rightbottom = (self.right, self.bottom)
        self.height = self.bottom-self.top + 1
        self.width = self.right - self.left +1
        self.maxsize = max(self.height, self.width)

        self.xc, self.yc = (self.left+self.right)//2, (self.top + self.bottom)//2

    def diag(self):
        return np.linalg.norm(np.array((self.height, self.width)))

    def center_bias(self, pts):
        dd_array = np.array(pts) - np.vstack([(self.xc, self.yc)]*len(pts))
        return np.linalg.norm(dd_array, axis=1)/(self.diag()/2)

        # return ProcUtils().pt2pt_distance(pt, (self.xc, self.yc)) / (self.diag()/2)
        #dx, dy = abs(pt[0]-self.xc), abs(pt[1]-self.yc)
        ##eturn max(dx, dy)/(self.maxsize/2)

    def copy(self):
        return Rect(pt1=self.lefttop, pt2=self.rightbottom)


class WorkSpace():
    """
    :param pts: list of points (tuple) (x,y)
    :param rect: list of 2 points [(x1, y1), (x2, y2)]
    :param max_shape: (xmax, ymax)
    """

    def __init__(self, pts=None, bbox=None, bound_margin=10):
        assert pts is not None or bbox is not None
        if pts is None: self.pts=None
        else: self.pts = np.array(pts)
        self.bbox = deepcopy(bbox)
        self.bound_margin = bound_margin

        self.infer()

    def infer(self):
        if self.pts is not None:
            self.bbox = PolyUtils().poly2box(self.pts)
        if self.bbox is not None and self.pts is None:
            self.pts = PolyUtils().box2poly(self.bbox)
        self.left, self.top, self.right, self.bottom = self.bbox
        self.height, self.width = self.bottom-self.top, self.right-self.left

    
    def set_bbox3d(self, pt_depths):
        pt_depths = np.array(pt_depths).reshape((-1, 1))
        norm = ArrayUtils().get_npoints_normal(np.concatenate((self.pts, pt_depths), axis=1))
        aa = 1

    def poly_path(self):
        return Path(self.pts)

    def get_mask(self):
        return PolyUtils().poly2mask(poly=self.pts)

    def get_mask_pts(self):
        mask = self.get_mask()
        return MaskUtils().mask2pts(mask=mask)

    def get_bound_mask(self):
        s = 2 * self.bound_margin + 1
        mask = self.get_mask()
        return mask - cv2.erode(mask, np.ones((s, s), np.uint8))

    def get_bound_margin(self):
        s = 2 * self.bound_margin + 1
        mask_margin = cv2.erode(self.get_mask(), np.ones((s, s), np.uint8))
        return mask_margin - cv2.erode(mask_margin, np.ones((5, 5), np.uint8))


    def get_bound_mask_pts(self):
        bound_mask = self.get_bound_mask()
        return MaskUtils().mask2pts(mask=bound_mask)

    def correct_ws(self, max_shape):
        xmax, ymax = max_shape
        # if self.rect is not None or self.pts is not None: return
        left, top, right, bottom = self.bbox
        if right <= xmax and bottom <= ymax: return
        self.pts = [(0, 0), (0, ymax), (xmax, ymax), (xmax, 0)]
        self.infer()

    def contain_points(self, pts):
        """
        :param pts: list of tuple (x,y)
        """
        return self.poly_path().contains_points(np.array(pts))

    def contain_boxes(self, boxes):
        """
        :param boxes: num_box x 4
        :type boxes: numpy.array
        :return: num_box x 1
        :rtype: numpy.array
        """
        contain_lefttop = self.poly_path().contains_points(boxes[:,[0,1]])
        contain_rightbottom = self.poly_path().contains_points(boxes[:,[2,3]])
        contain_leftbottom = self.poly_path().contains_points(boxes[:, [0,3]])
        contain_righttop = self.poly_path().contains_points(boxes[:, [2,1]])
        # contain_boxes =  np.logical_and(contain_lefttop, contain_rightbottom)
        return np.logical_and(np.logical_and(contain_lefttop, contain_rightbottom),
                              np.logical_and(contain_leftbottom, contain_righttop))

    def bound_pt(self, pt):
        pts_list = self.get_bound_mask_pts()
        return pt in pts_list

    def center_bias(self,pts):
        xc, yc = (self.left + self.right) // 2, (self.top + self.bottom) // 2
        diag = np.linalg.norm(np.array((self.height, self.width)))
        dd_array = np.array(pts).reshape((-1,2)) - np.vstack([(xc, yc)] * len(pts))
        return np.linalg.norm(dd_array, axis=1) / (diag / 2)

    def copy(self):
        return WorkSpace(pts=self.pts, bbox=self.bbox, bound_margin=self.bound_margin)

class CFG(BasDataObj):
    def __init__(self, cfg_path=None, separate=True, sections=None, excepts=None ):
        if cfg_path is not None:
            parser = configparser.RawConfigParser()
            parser.optionxform = str
            parser.read(cfg_path)
            arg_dict = dict()
            for section in parser.sections():
                if sections is not None:
                    if section not in sections: continue
                if excepts is not None:
                    if section in excepts: continue

                if separate: sec_config = CFG()
                for option in parser.options(section):
                    value = parser.get(section, option)
                    value = ProcUtils().get_value_from_express(value, arg_dict)#self.todict())
                    arg_dict.update({option: value})
                    if separate: sec_config.__setattr__(option,value)
                    else: self.__setattr__(option,value)
                if separate: self.__setattr__(section, sec_config)

    def keys(self):
        return [key for key in self.__dir__() if not callable(getattr(self,key)) and '__' not in key]

    def copy(self):
        # cfg = CFG()
        # [setattr(cfg, key, getattr(self, key)) for key in self.keys()]
        # return cfg
        return copy.deepcopy(self)

    def section_seperated(self):
        for name in self.keys():
            value = self.__getattribute__(name)
            if isinstance(value, CFG): return True
            else: return False

    def merge_with(self, args):
        for section_name in args.keys():
            section_value = args.__getattribute__(section_name)
            if not hasattr(self, section_name):
                self.__setattr__(section_name, section_value)
                continue

            section_ = self.__getattribute__(section_name)
            for option_name in section_value.keys():
                if hasattr(section_, option_name): continue
                section_.__setattr__(option_name, section_value.__getattribute__(option_name))


    def write(self, path='args.cfg'):
        save_dir, filename = os.path.split(path)
        filename, ext = os.path.splitext(filename)
        assert ext=='.cfg'
        if save_dir!='': os.makedirs(save_dir,exist_ok=True)
        f = open(path, 'w')
        f.write(self.to_string())
        f.close()

    def flatten(self):
        if not self.section_seperated(): return self
        else:
            args = CFG()
            for section_name in self.keys():
                section = self.__getattribute__(section_name)
                for option_name in section.keys():
                    value = section.__getattribute__(option_name)
                    args.__setattr__(option_name, value)
            return args

    def to_string(self, default_section='cfg'):
        text = ''
        if self.section_seperated():
            for section_name in self.keys():
                section = self.__getattribute__(section_name)
                text += '[{}]\n; {}\n'.format(section_name, '+'*50)
                for option_name in section.keys():
                    value = section.__getattribute__(option_name)
                    text += '{} = {}\n'.format(option_name, value)
                text+='\n'
        else:
            text += '[{}]\n; {}\n'.format(default_section, '+'*50)
            for option_name in self.keys():
                value = self.__getattribute__(option_name)
                text += '{} = {}\n'.format(option_name, value)

        return text

    def from_string(self,text, separate=False):
        path = 'args.cfg'
        f = open(path, 'w')
        f.write(text)
        f.close()
        args = CFG(cfg_path=path, separate=separate)
        os.remove(path)
        return args






from time import time
class Timer():
    def __init__(self):
        self.reset()

    def reset(self):
        self.times = [time(),]
        self.labels = []

    def len(self):
        return len(self.times)

    def pin_time(self, label='NaN'):
        self.add_time(label=label)
        return self.times[-1] - self.times[-2]

    def add_time(self, label='NaN'):
        self.times.append(time())
        self.labels.append(label)

    def run_time(self):
        if self.len()<2: self.add_time()
        return self.times[-1] - self.times[0]

    def fps(self):
        return 1/self.run_time()

    def run_time_str(self):
        runtime=self.run_time()
        if self.run_time() < 1: return'Total:%dms' % (1000*runtime)
        else:  return 'Total:%.2fs' %runtime

    def pin_times_str(self):
        if self.len()<2: return self.run_time_str()
        s = ''
        for i in range(self.len()-1):
            label  = self.labels[i]
            if label == 'NaN': continue
            dtime = self.times[i+1]-self.times[i]
            if dtime < 1: s += '%s:%dms-' %(self.labels[i], 1000*dtime)
            else: s += '%s:%.2fs-' % (self.labels[i], dtime)

        # if self.run_time() < 1:s+='Total:%dms' % (1000*self.run_time())
        # else:s+='Total:%.2fs' % self.run_time()
        s += self.run_time_str()

        return s

    def get_pinned_time(self, label):
        ind = self.labels.index(label)
        return self.times[ind+1]-self.times[ind]

class ArrayUtils():
    def copyfile(self, src_path, dst_path):
        if src_path==dst_path: return
        save_dir, _ = os.path.split(dst_path)
        os.makedirs(save_dir, exist_ok=True)
        copyfile(src_path, dst_path)


    def k_mean_cluster(self, anArray, n_cluster):
        from sklearn.cluster import  KMeans
        ret = KMeans(n_cluster, anArray)

        aa = 1

    def pt_in_box(self, pt, box):
        x, y = pt
        left, top, right, bottom = box
        x_in = (left<=x) and(x<=right)
        y_in = (top<=y) and (y<=bottom)
        return x_in and y_in

    def rot90(self,x,y, w,h, clockwise=True):
        if clockwise: return (h-y, x)
        else: return (y, w-x)

    def unrot90(self,x,y, w,h, clockwise=True):
        if clockwise: return (y, h-x)
        else: return (w - y, x)


    def crop_im(self, im, left=None, top=None, right=None, bottom=None):
        im_shape = im.shape
        h,w = im_shape[:2]
        if left is None: left=0
        if top is None: top=0
        if right is None: right=w
        if bottom is None: bottom=h
        if len(im_shape)>2: return im[top:bottom, left:right, :]
        else: return im[top:bottom, left:right]

    def resize(self, im, out_size, keep_ratio=True, pad_im=False, pad_value=0):
        if not keep_ratio: return cv2.resize(im, out_size, interpolation=cv2.INTER_CUBIC)
        h, w = im.shape[:2]
        wo, ho = out_size

        rat, rato = w/h, wo/ho
        if abs(rato-rat)<0.001: return cv2.resize(im, out_size, interpolation=cv2.INTER_CUBIC)

        wider = rato>rat
        if pad_im:
            if wider:       # pad x
                w1, h1 = int(rat*ho), ho
                left = (wo-w1)//2
                right = wo - (w1+left)
                out = cv2.resize(im, (w1, h1), interpolation=cv2.INTER_CUBIC)
                return  cv2.copyMakeBorder(out,0,0,left,right, cv2.BORDER_CONSTANT, value=pad_value)
            else:           # pad y
                w1, h1 = wo, int(wo/rat)
                top = (ho - h1) // 2
                bottom = ho - (h1 + top)
                out = cv2.resize(im, (w1, h1), interpolation=cv2.INTER_CUBIC)
                return cv2.copyMakeBorder(out, top, bottom,0, 0, cv2.BORDER_CONSTANT, value=pad_value)
        else:
            if wider:       # trim y
                w1, h1 = wo, int(wo/rat)
                top = (h1-ho) // 2
                bottom = top+ho
                out = cv2.resize(im, (w1, h1), interpolation=cv2.INTER_CUBIC)
                return self.crop_im(out, top=top, bottom=bottom)
            else:           # trim x
                w1, h1 = int(rat * ho), ho
                left = (w1 - wo) // 2
                right = left+wo
                out = cv2.resize(im, (w1, h1), interpolation=cv2.INTER_CUBIC)
                return self.crop_im(out, left=left, right=right)


    def fix_resize_scale(self, in_size, out_size):
        wi, hi = in_size
        wo, ho = out_size

        in_rat = wi/hi
        out_rat = wo/ho

        if out_rat>in_rat:
            h = ho
            w = int(ho*in_rat)
        else:
            w = wo
            h = int(wo/in_rat)

        return (w,h)





    def pt_in_im_range(self, pt, height, width, pad_size=(0,0)):
        xp, yp = pad_size
        w2, h2 = xp//2 +1, yp//2+1
        if pt[0]<w2 or (width-w2)<=pt[0]: return False
        if pt[1]<h2 or (height-h2)<=pt[1]: return False
        return True

    def append_array(self, anArray, container=None, axis=0):
        """ append `anArray` into `container`
        """
        if container is None:
            con_array = anArray
        else:
            con_array = np.concatenate((container, anArray), axis=axis)
        return con_array

    def concat_fixsize(self, im1, im2, data_type='uint8', axis=0, inter=cv2.INTER_CUBIC):
        """ concatenate 2 ndarray, if sizes are different, scale array2 equal to array1


        """
        im1, im2 = np.copy(im1), np.copy(im2)
        isColor1 = (len(im1.shape) == 3)
        isColor2 = (len(im2.shape) == 3)
        not_same_color = isColor1 ^ isColor2
        dtype1 = im1.dtype.name
        dtype2 = im2.dtype.name

        if dtype1 != dtype2:
            range0 = (0, 255)
            if dtype1 != data_type:
                range1 = (np.iinfo(dtype1).min, np.iinfo(dtype1).max)
                im1 = self.reval(im1, range1 + range0, data_type=data_type)
            if dtype2 != data_type:
                range2 = (np.iinfo(dtype2).min, np.iinfo(dtype2).max)
                im2 = self.reval(im2, range2 + range0, data_type=data_type)

        if not_same_color:
            if not isColor1: im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
            if not isColor2: im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)

        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]

        if axis == 0 and w1 != w2:
            h, w = int(1. * h2 / w2 * w1), w1
            im2 = cv2.resize(im2, (w, h), interpolation=inter)
        if axis == 1 and h1 != h2:
            h, w = h1, int(1. * w2 / h2 * h1)
            im2 = cv2.resize(im2, (w, h), interpolation=inter)
        return np.concatenate((im1, im2), axis=axis)

    def save_array(self, array, folds=None, filename=None, ext='.png'):
        if filename is None: filename = ProcUtils().get_current_time_str()
        filepath = filename + ext
        if folds is not None:
            folder = ''
            for fold in folds: folder = os.path.join(folder, fold)
            if not os.path.exists(folder): os.makedirs(folder)
            filepath = os.path.join(folder, filepath)
        cv2.imwrite(filepath, array)
        return filename

    def save_array_v2(self, array, fold=None, filename=None, ext='.png'):
        if filename is None: filename = ProcUtils().get_current_time_str()
        filepath = filename + ext
        if fold is not None:
            if not os.path.exists(fold): os.makedirs(fold)
        cv2.imwrite(filepath, array)
        return filename

    def save_array_v3(self, array, filepath=None, open_dir=False):
        if array is None: return
        if filepath is None: filepath = ProcUtils().get_time_name() + '.png'
        fold, _ = os.path.split(filepath)
        if not os.path.exists(fold): os.makedirs(fold)
        cv2.imwrite(filepath, array)
        print('image saved at {}'.format(filepath))
        if open_dir: ProcUtils().open_dir(fold)

    def get_mat_normal_map_U8(self, mat_in, nb_mean=False):
        return np.abs(255*self.get_mat_normal_map(mat_in=mat_in, nb_mean=nb_mean)).astype('uint8')

    def get_mat_normal_map(self, mat_in, nb_mean=False):
        mat = mat_in.astype(np.float)
        h, w = mat.shape[:2]
        X, Y = np.meshgrid(np.arange(0, w), np.arange(0, h))

        if nb_mean:
            M00 = self.matto3Dmat(mat, Y, X, 0, 0)
            Mm1m1 = self.matto3Dmat(mat, Y, X, -1, -1)
            M0m1 = self.matto3Dmat(mat, Y, X, 0, -1)
            M1m1 = self.matto3Dmat(mat, Y, X, 1, -1)
            M10 = self.matto3Dmat(mat, Y, X, 1, 0)
            M11 = self.matto3Dmat(mat, Y, X, 1, 1)
            M01 = self.matto3Dmat(mat, Y, X, 0, 1)
            Mm11 = self.matto3Dmat(mat, Y, X, -1, 1)
            Mm10 = self.matto3Dmat(mat, Y, X, -1, 0)

            v = np.zeros((h - 2, w - 2, 3, 8), np.float)
            v[:, :, :, 0] = self.get_3point_normals_mat(M00, Mm1m1, M0m1)
            v[:, :, :, 1] = self.get_3point_normals_mat(M00, M0m1, M1m1)
            v[:, :, :, 2] = self.get_3point_normals_mat(M00, M1m1, M10)
            v[:, :, :, 3] = self.get_3point_normals_mat(M00, M10, M11)
            v[:, :, :, 4] = self.get_3point_normals_mat(M00, M11, M01)
            v[:, :, :, 5] = self.get_3point_normals_mat(M00, M01, Mm11)
            v[:, :, :, 6] = self.get_3point_normals_mat(M00, Mm11, Mm10)
            v[:, :, :, 7] = self.get_3point_normals_mat(M00, Mm10, Mm1m1)
            v_mean = np.mean(v, axis=3)

            v_norm = np.linalg.norm(v_mean, axis=2)
            v_norm = self.repmat(v_norm, (1, 1, 3))
            v_norm = np.divide(v_mean, v_norm)


        else:
            Mm1m1 = self.matto3Dmat(mat, Y, X, -1, -1)
            M1m1 = self.matto3Dmat(mat, Y, X, 1, -1)
            M01 = self.matto3Dmat(mat, Y, X, 0, 1)
            v_norm = self.get_3point_normals_mat(M01, Mm1m1, M1m1)

        v = np.zeros((h, w, 3), np.float)
        v[1:-1, 1:-1, :] = np.copy(v_norm)

        v[0, 1:-1, :] = np.copy(v_norm[0, :, :])        # copy to boundary
        v[1:-1, 0, :] = np.copy(v_norm[:, 0, :])
        v[-1, 1:-1, :] = np.copy(v_norm[-1, :, :])
        v[1:-1, -1, :] = np.copy(v_norm[:, -1, :])
        v[(0,0)], v[(0,-1)], v[(-1,0)], v[(-1,-1)] = v_norm[(0,0)], v_norm[(0,-1)], v_norm[(-1,0)], v_norm[(-1,-1)]
        return v

    def get_mat_normals(self, mat, grad_weight=True):

        v = self.get_mat_normal_map(mat)
        v_norm = v[1:-1, 1:-1, :]

        # weighted mean
        if grad_weight:
            grad_x = self.get_gradient(mat, axis=1)
            grad_y = self.get_gradient(mat, axis=0)
            grad = np.abs(grad_x[1:-1, 1:-1]) + np.abs(grad_y[1:-1, 1:-1]) + 0.00001
            weight = grad / np.sum(grad)
            weight = self.repmat(weight, (1, 1, 3))
            v_mean = np.sum(np.multiply(v_norm, weight), axis=(0, 1))
        else:
            v_mean = np.mean(v_norm, axis=(0, 1))

        v_mean /= np.linalg.norm(v_mean)

        return v, v_mean

    def matto3Dmat(self, mat_in, Y_in, X_in, ry, rx):
        mat = np.copy(mat_in)
        X = np.copy(X_in)
        Y = np.copy(Y_in)
        h, w = mat.shape[:2]
        y_end = ry - 1
        if ry - 1 == 0: y_end = h
        x_end = rx - 1
        if x_end == 0: x_end = w

        mat = np.expand_dims(mat[ry + 1:y_end, rx + 1:x_end], axis=2)
        X = np.expand_dims(X[ry + 1:y_end, rx + 1:x_end], axis=2)
        Y = np.expand_dims(Y[ry + 1:y_end, rx + 1:x_end], axis=2)
        return np.concatenate((X, Y, mat), axis=2)

    def repmat(self, mat, ntimes):
        size = mat.shape
        mat_dim = len(size)
        out_dim = len(ntimes)
        assert mat_dim <= out_dim
        size += (1,) * (out_dim - mat_dim)
        out = np.copy(mat)
        for dim in range(out_dim):
            if dim >= mat_dim: out = np.expand_dims(out, axis=dim)
            ntime = ntimes[dim]
            out1 = np.copy(out)
            for i in range(ntime - 1): out1 = np.concatenate((out1, out), axis=dim)
            out = np.copy(out1)
        return out

    def get_3point_normals_mat(self, mat0, mat1, mat2):
        v1 = mat1 - mat0
        v2 = mat2 - mat0
        v = np.cross(v1, v2, axis=2)
        v_norm = np.linalg.norm(v, axis=2) +0.00001
        # v_norm = np.expand_dims(v_norm, axis=2)
        v_norm = self.repmat(v_norm, (1, 1, 3))
        v = np.divide(v, v_norm)
        return v

    def get_npoints_normal(self, pts):
        pts = np.array(pts)
        nel = len(pts)
        if nel<=3: return self.get_3points_normal(pts)

        ndim = pts.shape[1]
        center = np.mean(pts, axis=0)
        norms = []
        for i in range(nel-1):
            norms.append(self.get_3points_normal((center, pts[i,:], pts[i+1,:])))
        norms.append(self.get_3points_normal((center, pts[-1, :], pts[0, :])))
        norms = np.array(norms)
        norm = np.mean(norms, axis=0)
        return norm / np.linalg.norm(norm)

    def get_3points_normal(self, pts):
        pts = [np.array(pt).reshape((1,1,-1)) for pt in pts]
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        v = np.cross(v1, v2, axis=2).reshape(3,)
        v_norm = np.linalg.norm(v) + 0.00001
        return (v/v_norm).tolist()

    def truncate(self, input, vmin=None, vmax=None):
        out = np.copy(input)
        is_ndarray = isinstance(input, np.ndarray)
        if vmin is not None and vmax is None:
            if is_ndarray:
                out[np.where(input < vmin)] = vmin
            else:
                out = max(out, vmin)
        if vmin is None and vmax is not None:
            if is_ndarray:
                out[np.where(input > vmax)] = vmax
            else:
                out = min(out, vmax)
        if vmin is not None and vmax is not None:
            if vmin > vmax: return out
            if is_ndarray:
                out[np.where(input < vmin)] = vmin
                out[np.where(input > vmax)] = vmax
            else:
                out = max(out, vmin)
                out = min(out, vmax)
        return out

    def scale_val(self, value, scale_params, inverse=False):
        if not inverse:
            in_min, in_max, out_min, out_max = scale_params
        else:
            out_min, out_max, in_min, in_max = scale_params
        return (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

    def reval(self, mat, scale_params=(300., 1000., 0., 1.), invalid_locs=None, data_type=None):
        in_min, in_max, out_min, out_max = scale_params
        out = np.copy(mat)
        out[np.where(out < in_min)] = in_min
        out[np.where(out > in_max)] = in_max
        out = (out - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
        if invalid_locs is not None: out[invalid_locs] = 0
        if data_type is not None: out = out.astype(data_type)
        return out

    def convert_dtype(self, im, out_dtype):
        in_dtype = im.dtype.name
        if in_dtype == out_dtype: return im
        in_min, in_max = np.iinfo(in_dtype).min, np.iinfo(in_dtype).max
        out_min, out_max = np.iinfo(out_dtype).min, np.iinfo(out_dtype).max
        scale_params = (in_min, in_max, out_min, out_max)

        if len(im.shape) < 3: return self.reval(im, scale_params, data_type=out_dtype)

        ch = im.shape[-1]
        out = np.zeros(im.shape, out_dtype)
        for i in range(ch):
            out[:, :, i] = self.reval(im[:, :, i], scale_params, data_type=out_dtype)
        return out

    def concat_fixsize_list(self, ims, data_type=np.uint8, axes=0, inter=cv2.INTER_CUBIC):
        out = np.copy(ims[0])
        if not isinstance(axes, (list, tuple)):
            axes = [axes] * len(ims)
        for i in range(1, len(ims)):
            out = self.concat_fixsize(out, ims[i], data_type, axes[i], inter)
        return out

    def get_gradient(self, im_in, r=1, axis=0):
        im = im_in.astype('float')
        is_color = (len(im.shape) == 3)
        h, w = im.shape[:2]
        if is_color:
            im = np.copy(im[:, :, 0])
        else:
            im = np.copy(im)
        out = np.zeros((h, w), np.float)
        if axis == 0:
            out[r:-r, :] = im[0:-2 * r, :] - im[2 * r:, :]
        if axis == 1:
            out[:, r:-r] = im[:, 0:-2 * r] - im[:, 2 * r:]
        return out

    def locTo3dloc(self, locs):
        Y0, X0 = locs
        loc_len = len(locs[0])
        X = np.ndarray((3 * loc_len,), np.int)
        Y = np.ndarray((3 * loc_len,), np.int)
        Z = np.ndarray((3 * loc_len,), np.int)

        count = 0
        for y, x in zip(Y0, X0):
            for z in range(3):
                X[count] = x
                Y[count] = y
                Z[count] = z
                count += 1
        return (Y, X, Z)

    def line3Dregress(self, x, y, z, show=False):
        from mpl_toolkits.mplot3d import axes3d
        data = np.concatenate((x[:, np.newaxis],
                               y[:, np.newaxis],
                               z[:, np.newaxis]),
                              axis=1)
        datamean = data.mean(axis=0)
        uu, dd, vv = np.linalg.svd(data - datamean)

        x0, y0, z0 = vv[0]
        v = np.array((z0, z0 * y0 / x0, -x0 - y0 * y0 / x0))
        v /= np.linalg.norm(v)

        if show:
            linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]
            linepts += datamean

            linepts1 = v * np.mgrid[-7:7:2j][:, np.newaxis]
            linepts1 += datamean

            import matplotlib.pyplot as plt
            import mpl_toolkits.mplot3d as m3d
            ax = m3d.Axes3D(plt.figure())
            ax.scatter3D(*data.T)
            ax.plot3D(*linepts.T)
            ax.plot3D(*linepts1.T)
            plt.show()
        return v

    def sparsing_values(self, values, dy=10):
        ymin = np.amin(np.array(values))
        ymax = np.amax(np.array(values))
        if ymax - ymin > dy:
            strides = list(np.arange(ymin, ymax, dy))
            strides.append(ymax + 1)
            Y_reduce = []
            for j in range(len(strides) - 1):
                YY = [el for el in range(strides[j], strides[j + 1]) if el in values]
                if len(YY) == 0: continue
                med_value = int(np.median(np.array(YY)))
                if med_value not in YY:
                    Y_reduce.append(YY[0])
                else:
                    Y_reduce.append(med_value)
        else:
            med_value = int(np.median(np.array(values)))

            if med_value not in values:
                Y_reduce = [values[0], ]
            else:
                Y_reduce = [med_value, ]
        return Y_reduce

    def get_list_partition(self, alist, n=3):
        """
        partitioning a list of numbers into groups
        :param alist: list of numbers
        :param n: number of partitions
        :return: output: list of partitions
        """
        if n<2: return [alist]
        output = [None]*n
        aarray=np.array(alist)
        vmin, vmax = np.amin(aarray), np.amax(aarray)
        dv = (vmax-vmin)/n
        for avalue in alist:
            ind = int((avalue - vmin)/dv)
            if ind>(n-1): ind = n-1
            if output[ind] is None: output[ind] = [avalue]
            else: output[ind].append(avalue)
            # output[ind].append(avalue)
        return output

    def get_range_partition(self, start, end, n=3):

        if n <2: return [(start, end)]

        step = int(np.ceil((end-start)/n))
        a = np.arange(start, end, step)
        range_list = [(a[i], a[i+1]) for i in range(len(a)-1)]
        range_list.append((a[-1], end))
        return range_list




    def get_single_rgbd_partition(self, rgbd, partitions=(5, 5), loc=(0,0)):
        h, w = rgbd.size

        hp = math.ceil(h / partitions[0])
        wp = math.ceil(w / partitions[1])

        yp, xp = loc
        bottom = min(h, (yp + 1) * hp)
        right = min(w, (xp + 1) * wp)
        top, left = yp*hp, xp*wp

        return rgbd.crop(left=left, right=right, top=top, bottom=bottom), left, top

    def get_grid_locs_partition(self, pts, partitions=(5, 5), onlyBound=False):
        num_pt = len(pts)
        if num_pt < 10:
            return list(range(num_pt)), [num_pt]

        pts_array = np.array(pts)
        X, Y = pts_array[:, 0], pts_array[:, 1]
        ymin, ymax = np.amin(Y), np.amax(Y) + 1
        h = ymax - ymin
        hp = math.ceil(h / partitions[0])

        xmin, xmax = np.amin(X), np.amax(X) + 1
        w = xmax - xmin
        wp = math.ceil(w / partitions[1])



        concat_part_inds = []
        concat_part_lens = []

        for y_part in range(partitions[0]):
            top, bottom = ymin + y_part * hp, ymin + min(h, (y_part + 1) * hp)
            X_crop = X[np.where((top <= Y) & (Y < bottom))]

            if len(X_crop) ==0: continue

            xmin, xmax = np.amin(X_crop), np.amax(X_crop) + 1
            x_parts = list(np.arange(xmin, xmax, wp)) + [xmax]
            x_parts_len = len(x_parts)


            for i in range(x_parts_len-1):
                isBound = (y_part == 0) or (y_part == partitions[0] - 1) \
                          or (i==0) or (i==x_parts_len-2)
                if onlyBound and not isBound:
                    continue
                left, right= x_parts[i], x_parts[i+1]
                part_inds = []
                for i in range(num_pt):
                    x,y = X[i], Y[i]
                    if x < left or right <= x: continue
                    if y < top or bottom <= y: continue
                    part_inds.append(i)

                concat_part_inds += part_inds
                concat_part_lens.append(len(concat_part_inds))

        return concat_part_inds, concat_part_lens



    def meshgrid_locs (self, pts, partitions=(5, 5)):
        num_pt = len(pts)
        if num_pt < 2: return np.zeros((num_pt,), 'int')

        pts_array = np.array(pts)
        X, Y = pts_array[:, 0], pts_array[:, 1]
        ymin, ymax = np.amin(Y), np.amax(Y)
        xmin, xmax = np.amin(X), np.amax(X)

        h, w = ymax-ymin+1, xmax-xmin+1

        px, py = partitions
        wp, hp = int(np.ceil(w/px)), int(np.ceil(h/py))

        Xp, Yp = (X-xmin)//wp, (Y-ymin)//hp

        return (Xp, Yp)



    def reduce_connected_redundancy(self, anArray, value=1, mode='mean'):
        """
        :param anArray: 1D array
        :param value: decide 2 pixel connected while difference of their value larger than 'value'
        :return: reduced: redundancy reduced version of 'anArray'
        """
        num_el = len(anArray)
        a_sort = np.sort(anArray)
        a_diff = a_sort[1:] - a_sort[:-1]

        X = list(np.where(a_diff > value)[0])
        X.append(num_el)

        reduced = []
        for i in range(len(X)):
            rmax = X[i] + 1
            if i == 0:
                rmin = 0
            else:
                rmin = X[i - 1] + 1
            if mode == 'min':
                out_val = np.amin(a_sort[rmin:rmax])
            elif mode == 'max':
                out_val = np.amax(a_sort[rmin:rmax])
            else:
                out_val = int(np.mean(a_sort[rmin:rmax]))

            reduced.append(out_val)
        return np.array(reduced)

    def find_minmax_loc(self, im, find_min=True):
        if find_min: v = np.amin(im)
        else: v = np.amax(im)
        Y,X = np.where(im==v)
        return (Y[0], X[0])

    def find_partial_minmax_locs(self, im, partitions=(5,5), find_min=True):
        h,w = im.shape[:2]

        hp = math.ceil(h/partitions[0])
        wp = math.ceil(w/partitions[1])

        Y,X = [], []
        for yp in range(partitions[0]):
            ypmax = min(h, (yp+1)*hp)
            for xp in range(partitions[1]):
                xpmax = min(w, (xp + 1) * wp)
                imp = im[yp*hp:ypmax,xp*wp:xpmax]
                y0, x0 = self.find_minmax_loc(imp, find_min=find_min)
                Y.append(y0 + yp*hp)
                X.append(x0 + xp*wp)

        return (np.array(Y), np.array(X))

    def piecewise_linear(self, x, x0, y0, k1, k2):
        return np.piecewise(x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])

    def piecewise_linear_regress(self, X, Y):
        p,e = optimize.curve_fit(self.piecewise_linear, X, Y)
        return p,e

    def rect_crop(self, anArray, rect):
        if len(anArray.shape)>2:
            return np.copy(anArray[rect.top:rect.bottom, rect.left:rect.right, :])
        else:
            return np.copy(anArray[rect.top:rect.bottom, rect.left:rect.right])

    def crop_array_patch(self, anArray, center, pad_size):
        xc, yc = center
        xp, yp = pad_size
        xp2, yp2 = xp//2, yp//2

        array_shape = anArray.shape
        h,w = array_shape[:2]

        left, top = max(0, xc-xp2), max(0, yc-yp2)
        right, bottom = min(left+xp, w), min(top+yp, h)
        if len(array_shape)>2:
            return anArray[top:bottom, left:right, :]
        else:
            return anArray[top:bottom, left:right]








if __name__=='__main__':
    a = ProcUtils().read_csv('/mnt/workspace/000_data/ikea/out/detect.csv')




























