from ketisdk.utils.proc_utils import ProcUtils, WorkSpace, VisUtils
from ketisdk.base.base import BasObj
from ..utils.rgbd_utils_v2 import RGBD
from ..utils.array_utils import ArrayUtils
import os
from glob import glob
import cv2
import time
import json
import numpy as np


class BasIm():
    """ basic image manipulation

    :param args: configuration arguments
    :type args: :class:`.CFG`
    """

    def load_params(self, args):
        """ load image manipulation params
        """
        self.args = args
        if self.args is None: return 

        if not hasattr(self.args, 'depth_inpaint_rad'): self.args.__setattr__('depth_inpaint_rad', None)
        if not hasattr(self.args, 'depth_denoise_ksize'): self.args.__setattr__('depth_denoise_ksize', None)

        if not hasattr(self.args, 'crop_rect'): self.args.__setattr__('crop_rect', None)

        if not hasattr(self.args, 'crop_poly'): self.args.__setattr__('crop_poly', None)

        if self.args.crop_poly is not None or self.args.crop_rect is not None:
            workspace = WorkSpace(pts=self.args.crop_poly,
                                            bbox=self.args.crop_rect,
                                            bound_margin=self.args.bound_margin)
        else: workspace = None
        self.workspace=workspace


        if hasattr(self.args, 'bg_dir'):
            rgb_paths = glob(os.path.join(self.args.root_dir, self.args.bg_dir, 'rgb.*'))
            if len(rgb_paths) == 0:
                self.bg_rgb = None
            else:
                self.bg_rgb = cv2.imread(rgb_paths[0])

            depth_paths = glob(os.path.join(self.args.root_dir, self.args.bg_dir, 'depth.*'))
            if len(depth_paths) == 0:
                self.bg_depth = None
            else:
                self.bg_depth = cv2.imread(depth_paths[0], -1)
        else:
            self.bg_rgb = None
            self.bg_depth = None

        print('BasIm params loaded ...')

    def get_rgbd(self,rgb=None, depth=None, bg_rgb=None, bg_depth=None, depth_uint=1):
        """ get rgbd object

        :param rgb: RGB image
        :type rgb: 3-channel ndarray
        :param depth: depth image
        :type depth: single-channel ndarray
        :param bg_rgb: RGB image
        :type bg_rgb: 3-channel ndarray
        :param bg_depth: depth image
        :type bg_depth: single-channel ndarray
        :param depth_uint: real-world length per image pixel value
        :type depth_uint: float
        :returns: :class:`.RGBD`
        """
        assert rgb is not None or depth is not None
        if bg_rgb is None: bg_rgb= self.bg_rgb
        if bg_depth is None: bg_depth=self.bg_depth

        if not hasattr(self,'workspace'): self.workspace = None
        if not hasattr(self.args, 'depth_denoise_ksize'): self.args.__setattr__('depth_denoise_ksize', None)

        return RGBD(rgb=rgb, depth=depth, depth_unit=depth_uint,
                    rgb_bg=bg_rgb, depth_bg=bg_depth,
                    workspace=self.workspace,
                    depth_min=self.args.depth_min, depth_max=self.args.depth_max,
                    depth_denoise_ksize=self.args.depth_denoise_ksize)

    def save_rgbd(self, rgbd, filename=None, save_dir_name='cam_data'):
        """ save rgbd object to files

        :param rgbd: rgbd object
        :type rgbd: :class:`.RGBD`
        :param filename: image filename
        :type filename: str
        """
        subdir = ProcUtils().get_current_time_str('%m%d')
        if rgbd.hasRgb: filename = ArrayUtils().save_array(array=rgbd.bgr(), folds=[self.args.root_dir, save_dir_name, subdir,'image'], filename=filename)
        if rgbd.hasDepth:
            _ = ArrayUtils().save_array(array=rgbd.depth, folds=[self.args.root_dir, save_dir_name, subdir,'depth'], filename=filename)
            # _ = ArrayUtils().save_array(array=rgbd.depth_disp(args=self.args, show_jet=False),
            #                folds=[self.args.data_root, save_dir_name, subdir, 'depth_disp'],
            #                filename=filename)
            # _ = ArrayUtils().save_array(array=rgbd.depth_disp(args=self.args, show_jet=True),
            #                folds=[self.args.data_root, save_dir_name, subdir, 'depth_disp_jet'],
            #                filename=filename)
        print('image saved ...')
        time.sleep(0.03)

class BasImObj(BasObj, BasIm):
    """ [:class:`.BasObj`, :class:`.BasIm`] basic image manipulation object"""
    def load_params(self, args):
        BasObj.load_params(self, args=args)
        BasIm.load_params(self, args=args)

class BasDetect():
    """ basic detection process

    :param args: configuration arguments
    :type args: :class:`.CFG`
    """
    # def __init__(self,args=None):
    #     self.args=args

    def get_model(self):
        """ get model """
        self.model=None
        print('Model loaded ...')

    def detect(self, **kwargs):
        """ predict """
        print('Detecting ...')


class BasDetectObj(BasObj, BasDetect):
    """ [:class:`.BasObj`, :class:`.BasDetect`] basic detection obj

        :param args: configuration arguments
        :type args: :class:`.CFG`
        :param cfg_path: configuation file path
        :type cfg_path: .cfg file
        """
    def __init__(self, args=None, cfg_path=None):
        super().__init__(args=args, cfg_path=cfg_path)
        self.get_model()

    def reload_params(self):
        super().reload_params()
        if hasattr(self, 'model'):
            if hasattr(self.model,'reload_params'): self.model.reload_params()

class DetGui():
    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', detected=None):
        pass
    def finalize_acc(self):
        pass
    def init_acc(self):
        pass

class GuiObj(BasObj, DetGui):
    pass

class DetGuiObj(BasDetectObj, DetGui):
    pass



class BasJson():
    """ basic manipulation on json file

    :param args: configuration arguments
    :type args: :class:`.CFG`
    """
    def __init__(self,args=None):
        self.args=args

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

    def save_poly_json(self, filename, im_info, class_list, polygons, folds=['']):
        """ save polygon to json file

        :param filename: json filename
        :param im_info: include image height, width, number of channel, and image name
        :param class_list: list of classes
        :param polygons: list of polygon
        :param folds: serial of folders saving json file
        """
        json_path = BasFileManage().mkdir_and_mkfilepath(folds=folds, filename=filename)

        # val json
        Ann_dict = dict()

        Ann_dict.update({'Info': im_info})
        Ann_dict.update({'Classes': class_list})
        Ann_dict.update({'Polygons': polygons})

        instance_json_obj = open(json_path, 'w+')
        instance_json_obj.write(json.dumps(Ann_dict))
        instance_json_obj.close()


class BasPoly():
    """ basic manipulation on polygon

    :param args: configuration arguments
    :type args: :class:`.CFG`
    """

    def __init__(self,args=None):
        self.args=args

    def draw_poly(self, im, poly, color=None, thick=2):
        """ draw a polyygon on given image

        :param im: given image
        :type im: ndarray
        :param poly: list of points
        :param color: display line-color
        :param thick: thickness of line
        :returns: drawn image
        """
        if color is None:
            if hasattr(self.args,'line_color'): color = self.args.line_color
            else: color = (0,255,0)
        if hasattr(self.args, 'line_thick'): thick = self.args.line_thick

        out = np.copy(im)
        for i in range(len(poly) - 1):
            cv2.line(out, tuple(poly[i]), tuple(poly[i + 1]), color, thick)
        cv2.line(out, tuple(poly[-1]), tuple(poly[0]), color, thick)
        return out

    def draw_polys(self, im):
        """ draw  polygons on given image

        :param im: given image
        :type im: ndarray
        :returns:drawn image
        """
        out = np.copy(im)
        for cls, poly in zip(self.classes, self.polygons):
            out = self.draw_poly(out, poly, color=self.color_dict[cls])

        return out


    def get_info_from_json(self, json_path):
        """
        :param json_path: json file path
        """
        ann_dict = BasJson().read_json(json_path)

        info = ann_dict['Info']
        self.classes = ann_dict['Classes']
        self.polygons = ann_dict['Polygons']

        self.classes_unq = list(np.unique(self.classes))
        self.colors = ProcUtils().get_color_list(len(self.classes_unq))
        self.color_dict = dict()
        for cls, color in zip(self.classes_unq, self.colors):
            self.color_dict.update({cls: color})

    def draw_polys_from_json(self, json_path, im=None, im_path=None, text_scale=1.5, text_thick=3,
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

        self.get_info_from_json(json_path)
        out = self.draw_polys(im)



        if hasattr(self, 'args'):
            text_scale, text_thick, rect, space, alpha, up2down = self.args.text_scale, self.args.text_thick, \
                                                                  self.args.text_rect, self.args.text_space, \
                                                                  self.args.text_alpha, self.args.text_alpha

        out = VisUtils().draw_texts_blend(out, texts=self.classes_unq, colors=self.colors, scale=text_scale, thick=text_thick,
                               text_rect=rect, space=space, alpha=alpha, up2down=up2down)

        return out

    def draw_polys_on_rgbd(self,  json_path, rgbd):
        self.get_info_from_json(json_path)
        rgb, depth = None, None
        if rgbd.hasRgb: rgb = self.draw_polys(rgbd.rgb)
        if rgbd.hasDepth: depth = self.draw_polys(rgbd.depth)
        args=None
        if hasattr(self, 'args'): args = self.args
        return RGBD(rgb=rgb, depth=depth).disp(args=args)


    def get_rect_from_poly(self, poly):
        """ get rectangle bounding points of polygon
        :param poly: list of points
        :returns: rect [left, top, right, bottom]
        """
        poly_array = np.array(poly)

        left, top = np.amin(poly_array, axis=0)
        right, bottom = np.amax(poly_array, axis=0)

        return (left, top, right, bottom)

class BasSeqIm(BasIm):
    """ [:class:`.BasIm`] basic manipulation on image sequence
    """
    def get_im_list(self):
        """ get image path lists """
        self.use_rgb = hasattr(self.args,'im_suffixes')
        self.use_depth = hasattr(self.args,'depth_suffixes')
        self.paths = []
        if self.use_rgb and not self.use_depth:
            if isinstance(self.args.im_suffixes, str): self.args.im_suffixes = [self.args.im_suffixes]
            for im_suffix in self.args.im_suffixes:
                for path in glob(os.path.join(self.args.root_dir, im_suffix), recursive=True):
                    path = path.replace('\\', '/')
                    self.paths.append([path])
        if not self.use_rgb and self.use_depth:
            if isinstance(self.args.depth_suffixes, str): self.args.depth_suffixes = [self.args.depth_suffixes]
            for depth_suffix in self.args.depth_suffixes:
                for path in glob(os.path.join(self.args.root_dir, depth_suffix), recursive=True):
                    path = path.replace('\\', '/')
                    self.paths.append([path])
        if self.use_rgb and self.use_depth:
            if isinstance(self.args.im_suffixes, str): self.args.im_suffixes = [self.args.im_suffixes]
            if isinstance(self.args.depth_suffixes, str): self.args.depth_suffixes = [self.args.depth_suffixes]
            for im_suffix, depth_suffix in zip(self.args.im_suffixes, self.args.depth_suffixes):
                all_im_path = glob(os.path.join(self.args.root_dir, im_suffix), recursive=True)
                for im_path in all_im_path:
                    im_path = im_path.replace('\\', '/')
                    depth_path = im_path
                    for im_suf, dp_suf in zip(im_suffix.split('*'), depth_suffix.split('*')):
                        depth_path = depth_path.replace(im_suf, dp_suf)
                    if not os.path.exists(depth_path): continue
                    self.paths.append([im_path, depth_path])

        self.num_im = len(self.paths)
        self.has_bg = hasattr(self.args, 'bg_dir')

    def get_rgbd_from_path(self, paths):
        """
        :param paths: list of rgbd path. paths[0]: image path, paths[-1]: depth path
        """
        if self.use_rgb: rgb = cv2.imread(paths[0])[:,:,::-1]
        else: rgb=None
        if self.use_depth: depth = cv2.imread(paths[-1], -1)
        else: depth=None

        bg_rgb = None
        bg_depth = None
        if self.has_bg:
            rgb_paths = glob(os.path.join(self.args.root_dir, self.args.bg_dir, 'rgb.*'))
            if len(rgb_paths)>0: bg_rgb = cv2.imread(rgb_paths[0])
            depth_paths = glob(os.path.join(self.args.root_dir, self.args.bg_dir, 'depth.*'))
            if len(depth_paths) > 0: bg_depth=cv2.imread(depth_paths[0], -1)

        self.rgbd = super().get_rgbd(rgb=rgb, depth=depth, bg_rgb=bg_rgb, bg_depth=bg_depth)

class BasSeqImObj(BasObj, BasSeqIm):
    """ [:class:`.BasObj`, :class:`.BasSeqIm`] basic image sequence manipulation object
    """
    def load_params(self, args):
        BasObj.load_params(self, args=args)
        BasSeqIm.load_params(self, args=args)
        self.get_im_list()

        if self.args.wait_key:
            # display windows
            cv2.namedWindow('viewer', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('viewer', args.window_size[1], args.window_size[0])

    def paths_exist(self):
        """ check if paths exist. If not, choose EXIT or run with reloaded configs"""
        if self.num_im > 0: return True

        print('No images in %s' % self.args.root_dir)
        if not self.args.wait_key: exit()

        cv2.imshow('viewer', 150 * np.ones(shape=(400, 600), dtype=np.uint8))
        print('ESC: to exit,\tANY_KEY: to run with different configs ...')
        if cv2.waitKey() == 27: exit()
        self.reload_params()
        return False


    def process_this(self):
        in_dir, self.filename = os.path.split(self.rgbd_path[0])
        self.filename, self.fileext = os.path.splitext(self.filename)
        test_some = hasattr(self.args, 'images')
        test_ignore = hasattr(self.args, 'ignores')
        if test_some:
            if self.filename not in self.args.images:
                print('>>> This image not in test list >>> Skip')
                ProcUtils().clscr()
                return False
        if test_ignore:
            if self.filename in self.args.ignores:
                print('>>> This image in ignore list >>> Skip')
                ProcUtils().clscr()
                return False
        return True

    def process_single(self, **kwargs):
        print('processing single ... ')

    def run(self):
        """ iteratively examine whole images of sequence"""
        while True:
            if not self.paths_exist(): continue
            for i in range(self.num_im):
                self.ind=i
                self.rgbd_path = self.paths[i]
                print('Image %d out of %d' % (self.ind, self.num_im))
                print(self.rgbd_path[0])
                if not self.process_this(): continue
                self.get_rgbd_from_path(self.rgbd_path)
                self.process_single()
                # self.move_processed_rgbd()
            break

    def move_processed_rgbd(self):
        """move processed file to temp folder"""
        # if not hasattr(self.args, 'move_file'): return
        # if not self.args.move_file: return
        for path in self.rgbd_path:
            in_dir, filename = os.path.split(path)
            out_dir = in_dir + '_processed'
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            os.rename(path, os.path.join(out_dir, filename))

class BasFileManage():
    """ basic file manager """
    def move_files_dir2dir(self, in_dir, out_dir, file_format='*'):
        """ move all file with `format` from `in_dir` to `out_dir`

        :param in_dir: input directory path
        :type in_dir: str
        :param out_dir: output directory path
        :type out_dir: str
        :param file_format: format of which is going to be move
        :type file_format: str
        """
        assert os.path.exists(in_dir)
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        file_paths = glob(os.path.join(in_dir, file_format), recursive=True)

        for path in file_paths:
            print('moving %s ...' %path)
            os.rename(path, path.replace(in_dir, out_dir))

    # def add_subdir_to_path(self, path):
    #     indir, filename = os.path.split(path)
    #     indir, subdir = os.path.split(indir)
    #     return os.path.join(indir, '%s_%s' %(subdir, filename))

    def move_file(self, path, root_dir, out_dir):
        """ move a file with given `path` from  `root_dir` to `out_dir`.
        Add subdirs to `out_path` if `root_dir` differs with folder containing file.

        :param path: input file path
        :type path: str
        :param root_dir: input root directory path
        :type root_dir: str
        :param out_dir: output directory path
        :type out_dir: str
        """
        assert os.path.exists(path)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        indir, filename = os.path.split(path)
        subdirs = indir.replace(root_dir, '')

        out_path = os.path.join(out_dir, '%s_%s' %(subdirs.replace('/','_'), filename))
        os.rename(path, out_path)

    # def get_subdir_name(self, path, root_dir):
    #     indir, filename = os.path.split(path)
    #     return indir.replace(root_dir, '')

    def mkdir_and_mkfilepath(self, filename, folds):
        out_dir = ''
        for fold in folds: out_dir = os.path.join(out_dir, fold)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        return os.path.join(out_dir, filename)


class BasSeqImDetectObj(BasSeqImObj, BasDetectObj):
    """[:class:`.BasSeqImObj`, :class:`.BasDetectObj`] basic detection on image sequence """
    def __init__(self, args=None, cfg_path=None):
        BasSeqImObj.__init__(self, args=args, cfg_path=cfg_path)
        BasDetectObj.__init__(self, args=args, cfg_path=cfg_path)

    def reload_params(self):
        BasSeqImObj.reload_params(self)
        BasDetectObj.reload_params(self)

if __name__ =='__main__':
    pass
