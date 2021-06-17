import cv2
import numpy as np
from ketisdk.utils.proc_utils import ProcUtils, BasDataObj, Rect, WorkSpace, ArrayUtils
from scipy.ndimage import rotate


class RGBD(BasDataObj):
    """ RGBD class

    - can be initialized by given ndarray or image file path
    - crop and inpaint if given params
    - subtract background if given background rgb or depth
    - can treat different depth specs

    :param rgb: rgb color image
    :type rgb: 3-channel ndarray, uint8
    :param depth: depth image
    :type depth: 1-channel ndarray, uint16
    :param rgb_path: rgb image file path
    :type rgb_path: str
    :param depth_path: depth image file path
    :type depth_path: str
    :param crop_size: for set workspace(top, left, bottom, right)
    :type crop_size: int tuple
    :param depth_inpaint_rad: radius params used in depth inpainting
    :type depth_inpaint_rad: int
    :param rgb_bg: rgb of background (if existed)
    :type rgb_bg: 3-channel ndarray, uint8
    :param depth_bg: depth of background (if existed)
    :type depth_bg: 1-channel ndarray, uint16
    :param depth_unit: real-world length per depth value. Ex, depth_unit=1 means 1mm per depth value
    :type depth_unit: float
    """
    def __init__(self, rgb=None, depth=None, rgb_path=None, depth_path=None, workspace=None, depth_inpaint_rad=None,
                 rgb_bg=None, depth_bg=None, depth_unit=1, depth_min=300, depth_max=1200, depth_denoise_ksize=None, extra=None):
        self.depth_inpaint_rad = depth_inpaint_rad
        self.depth_denoise_ksize=depth_denoise_ksize
        self.hasRgb=False
        self.hasDepth=False
        self.width=None
        self.height=None
        self.depth=depth
        self.rgb=rgb
        self.depth_bg=depth_bg
        self.rgb_bg=rgb_bg
        self.depth_unit = depth_unit
        if workspace is None: self.workspace=None
        else:self.workspace=workspace.copy()

        self.depth_min = depth_min
        self.depth_max = depth_max

        self.set_rgb(rgb=rgb, rgb_path=rgb_path)
        self.set_depth(depth=depth, depth_path=depth_path)
        self.set_bg_rgb(rgb=rgb_bg)
        self.set_bg_depth(depth=depth_bg)
        self.has_bg=False

        if extra is not None:
            for key in extra: self.__setattr__(key, extra[key])


    def bgr(self):
        return np.copy(self.rgb[:,:,::-1])

    def copy_rgb(self):
        return np.copy(self.rgb)

    def copy_depth(self):
        return np.copy(self.depth)

    def gray(self):
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)

    def crop_gray(self):
        gray=self.gray()
        return gray[self.workspace.top:self.workspace.bottom, self.workspace.left:self.workspace.right]

    def set_workspace(self, pts=None):
        # if self.workspace is None: self.workspace=WorkSpace(pts=pts,
        #                                                     bbox=(0,0,self.width,self.height))
        # else: self.workspace.correct_ws(max_shape=(self.width, self.height))
        self.workspace = WorkSpace(pts=pts, bbox=(0, 0, self.width, self.height))
        self.workspace.correct_ws(max_shape=(self.width, self.height))

    def change_workspace(self, workspace=None):
        self.workspace = workspace
        self.workspace.correct_ws(max_shape=(self.width, self.height))

    def set_rgb(self, rgb=None, rgb_path=None):
        """set rgb with given ndarray or rgb file path
        """
        if rgb is not None: self.rgb=np.copy(rgb)
        else: self.rgb=None
        self.hasRgb = rgb is not None or ProcUtils().isimpath(rgb_path)
        if not self.hasRgb: return 0
        if ProcUtils().isimpath(rgb_path):
            self.rgb = cv2.imread(rgb_path)[::-1]
        self.height, self.width = self.rgb.shape[:2]
        if self.workspace is not None: self.workspace.correct_ws(max_shape=(self.width,self.height))
        else: self.set_workspace()
        # self.roi = ProcUtils().correct_cropsize(self.crop_size, self.size)
        self.fix_depth_size()

    def set_depth(self, depth=None, depth_path=None):
        """set depth with given ndarray or depth file path"""
        if depth is not None: self.depth=np.copy(depth)
        else: self.depth=None
        self.hasDepth = depth is not None or ProcUtils().isimpath(depth_path)
        if not self.hasDepth: return 0
        if ProcUtils().isimpath(depth_path):
            self.depth = cv2.imread(depth_path, -1)
        if self.hasRgb: self.height, self.width = self.rgb.shape[:2]
        else: self.height, self.width = self.depth.shape[:2]
        if self.workspace is not None: self.workspace.correct_ws(max_shape=(self.width, self.height))
        else: self.set_workspace()
        self.fix_depth_size()
        if self.depth_inpaint_rad is not None:
            depth_crop = self.crop_depth()
            left, top, right, bottom = self.workspace.bbox
            self.depth[top:bottom, left:right] = self.inpaint_depth(self.depth_inpaint_rad,depth_crop)

        if self.depth_denoise_ksize is not None:
            depth_crop_filtered = cv2.medianBlur(self.crop_depth(), ksize=self.depth_denoise_ksize)
            left, top, right, bottom = self.workspace.bbox
            self.depth[top:bottom, left:right] = depth_crop_filtered

        # self.invalid_depth_locs = np.where((self.depth < self.depth_min) | (self.depth_max < self.depth))
    def depth_U8(self):
        return ArrayUtils().reval(self.depth, scale_params=(self.depth_min, self.depth_max, 5, 250),
                                  data_type='uint8')

    def array(self, get_rgb=True, get_depth=True, depth2norm=False):
        """ return raw data of rgbd in format uint8"""
        use_rgb = self.hasRgb and get_rgb
        use_depth = self.hasDepth and get_depth
        if use_rgb and not use_depth: return self.rgb
        if use_depth:
            if not use_rgb and not depth2norm: return ArrayUtils().repmat(self.depth_U8, (1,1,3))
            if not use_rgb and depth2norm: return ArrayUtils().get_mat_normal_map_U8(self.depth)
            if use_rgb and not depth2norm: return np.concatenate((self.rgb, np.expand_dims(self.depth_U8(), axis=2)), axis=2)
            if use_rgb and depth2norm: return np.concatenate((self.rgb, ArrayUtils().get_mat_normal_map_U8(self.depth)), axis=2)

    def resize(self, size):
        """ return a resize of rgbd

        :param size: output size (out_height, out_width)
        :type size: int tuple
        """
        if size == (self.height, self.width): return self
        rgbd = RGBD(depth_min=self.depth_min, depth_max=self.depth_max)
        if self.hasRgb:
            rgbd.set_rgb(rgb=cv2.resize(self.rgb, size[::-1], interpolation=cv2.INTER_CUBIC))
        if self.hasDepth:
            rgbd.set_depth(depth=cv2.resize(self.depth, size[::-1], interpolation=cv2.INTER_CUBIC))
        if hasattr(self, 'rgb_bg'): rgbd.set_bg_rgb(rgb=self.rgb_bg)
        if hasattr(self, 'depth_bg'): rgbd.set_bg_depth(depth=self.depth_bg)
        return rgbd

    def get_diag_pad_params(self):
        """get diagonal padding params"""
        # length = 2 * int(np.linalg.norm((self.height, self.width)) / 2) + 1
        length = int(np.linalg.norm((self.height, self.width)))
        top, left = int((length - self.height) / 2), int((length - self.width) / 2)
        bottom, right = length - self.height - top, length - self.width - left

        return (left, top, right, bottom, length)

    def pad(self, left=None, top=None, right=None, bottom=None, rgb_val=(0,0,0), depth_val=0):
        """rgbd padding"""
        if left is None or top is None or right is None or bottom is None:
            left, top, right, bottom, length= self.get_diag_pad_params()
        rgbd = RGBD(depth_min=self.depth_min, depth_max=self.depth_max)
        if self.hasRgb:
            rgbd.set_rgb(rgb=cv2.copyMakeBorder(self.rgb,top,bottom,left, right, cv2.BORDER_CONSTANT, value=rgb_val))
        if self.hasDepth:
            rgbd.set_depth(depth=cv2.copyMakeBorder(self.depth,top,bottom,left, right, cv2.BORDER_CONSTANT, value=depth_val))
        return rgbd

    def rotate(self, angle, rgb_val=0.0, depth_val=0.0):
        """rgbd rotate

        :param  angle: rotate angle
        :type angle: float, degree
        """
        rot_90 = ((angle//90)==(angle/90))
        if angle != 0:
            rgbd = RGBD(depth_min=self.depth_min, depth_max=self.depth_max)
            rotation_matrix = cv2.getRotationMatrix2D((self.width / 2, self.height / 2), angle, 1)
            if self.hasRgb:
                # rgbd.set_rgb(rgb=rotate(self.rgb, angle=angle, reshape=False, order=3, cval=rgb_val))
                rgbd.set_rgb(rgb=cv2.warpAffine(self.rgb, rotation_matrix, (self.width, self.height)))
            if self.hasDepth:
                # rgbd.set_depth(depth=rotate(self.depth, angle=angle, reshape=False, order=3, cval=depth_val))
                rgbd.set_depth(depth=cv2.warpAffine(self.depth, rotation_matrix, (self.width, self.height)))
            return rgbd
        return self

    def crop(self, left=None, top=None, right=None, bottom=None):
        """ crop rgbd
        """
        if left is None or right is None or top is None or bottom is None:
            if self.workspace is None: return self
            left, top, right, bottom = self.workspace.bbox
        else:
            left, right = self.cx(left), self.cx(right)
            top, bottom = self.cy(top), self.cy(bottom)
        rgbd = RGBD(depth_min=self.depth_min, depth_max=self.depth_max)
        if self.hasRgb:
            rgbd.set_rgb(rgb=self.rgb[top:bottom, left:right])
        if self.hasDepth:
            rgbd.set_depth(depth=self.depth[top:bottom, left:right])
        return rgbd


    def crop_patch(self, center, pad_size):
        x, y = center
        wp, hp = pad_size
        w2, h2 = wp//2, hp//2
        left, top = max(0,x-w2), max(0, y-h2)
        right, bottom = min(left+wp, self.width), min(top+hp, self.height)
        return self.crop(left=left, right=right, top=top, bottom=bottom)

    def crop_patch_array(self, center, pad_size, dsize=None, get_rgb=True, get_depth=True, depth2norm=False):
        x, y = center
        wp, hp = pad_size
        w2, h2 = wp//2, hp//2
        left, top = max(0,x-w2), max(0, y-h2)
        right, bottom = min(left+wp, self.width), min(top+hp, self.height)
        return self.crop(left=left, right=right, top=top, bottom=bottom).resize(size=dsize).\
            array(get_rgb=get_rgb,get_depth=get_depth, depth2norm=depth2norm)


    def get_nb_arrays(self, center, pad_size, dsize=None, get_rgb=True, get_depth=True, r=3, depth2norm=False):

        xc, yc = center
        wp, hp = pad_size

        if not self.pt_in_im_range(center, pad_size=(wp+r, hp+r)): return None


        use_rgb = self.hasRgb and get_rgb
        use_depth = self.hasDepth and get_depth

        if dsize is None: dsize = pad_size



        ws, hs = dsize
        fx, fy = 1.*ws/wp, 1.*hs/hp

        if use_rgb:
            rgb_s = cv2.resize(self.rgb,None, fx=fx, fy=fy,interpolation=cv2.INTER_CUBIC)
            height, width = rgb_s.shape[:2]
        if use_depth:
            if not depth2norm: depth_s = ArrayUtils().repmat(cv2.resize(self.depth_U8,None, fx=fx, fy=fy,
                                                                        interpolation=cv2.INTER_CUBIC), (1,1,3))
            else: depth_s = ArrayUtils().get_mat_normal_map_U8(self.depth)
            height, width = depth_s.shape[:2]



        xc, yc = int(fx*xc), int(fy*yc)

        wp, hp = dsize
        w2, h2 = wp // 2, hp // 2

        nb_range = range(-r, r+1)
        arrays = []
        for dy in nb_range:
            for dx in nb_range:
                left, top = max(0, xc +dx - w2), max(0, yc+dy - h2)
                right, bottom = min(left + wp, width), min(top + hp, height)
                if use_rgb and not use_depth:
                    array = rgb_s[top:bottom, left:right, :]
                if use_depth:
                    array = depth_s[top:bottom, left:right, :]

                if use_rgb and use_depth:
                    if not depth2norm: array = np.concatenate((rgb_s[top:bottom, left:right, :],
                                            np.expand_dims(depth_s[top:bottom, left:right, 0], axis=2)), axis=2)
                    else: array = np.concatenate((rgb_s[top:bottom, left:right, :],
                                                  depth_s[top:bottom, left:right, :]), axis=2)
                arrays.append(array)
        return arrays


    def fix_depth_size(self):
        """ fix depth size equal to image size"""
        if self.hasRgb and self.hasDepth:
            if self.depth.shape[:2] != (self.height, self.width):
                self.depth = cv2.resize(self.depth, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

    def set_bg_rgb(self, rgb=None, rgb_path=None):
        """ set rgb of background"""
        if rgb is not None or ProcUtils().isimpath(rgb_path):
            self.has_bg=True
            if ProcUtils().isimpath(rgb_path): rgb = cv2.imread(rgb_path)[:,:,::-1]
            self.rgb_bg = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

    def set_bg_depth(self, depth=None, depth_path=None):
        """ set depth of background"""
        if depth is not None or ProcUtils().isimpath(depth_path):
            self.has_bg = True
            if ProcUtils().isimpath(depth_path): depth = cv2.imread(depth_path, -1)
            self.depth_bg = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_CUBIC)


    def get_fg_mask(self, bg_depth_diff_thres = 20):        # 20 mm
        """set foreground mask"""
        diff = self.depth.astype(np.float)-self.depth_bg.astype(np.float)
        out = 255*(np.abs(diff)>=bg_depth_diff_thres).astype(np.uint8)
        return out

    def crop_rgb(self, left=None, top=None, right=None, bottom=None, pad_val=None):
        """crop rgb"""
        if self.rgb is None: return None
        if left is None or right is None or top is None or bottom is None:
            left, top, right, bottom = self.workspace.bbox
        else:
            left, right = self.cx(left), self.cx(right)
            top, bottom = self.cx(top), self.cx(bottom)

        out = np.copy(self.rgb[top:bottom, left:right, :])
        if pad_val is not None:
            ones_array = np.ones((self.height, self.width,1), 'uint8')
            out_pad = np.concatenate((pad_val[0]*ones_array, pad_val[1]*ones_array, pad_val[2]*ones_array), axis=2)
            out_pad[top:bottom, left:right, :] = out
            out = out_pad

        return out


    def crop_depth(self, left=None, top=None, right=None, bottom=None, pad_val=None):
        """crop depth"""
        if self.depth is None: return None
        if left is None or right is None or top is None or bottom is None:
            left, top, right, bottom = self.workspace.bbox
        else:
            left, right = self.cx(left), self.cx(right)
            top, bottom = self.cx(top), self.cx(bottom)

        out = np.copy(self.depth[top:bottom, left:right])
        if pad_val is not None:
            out_pad = pad_val*np.ones((self.height, self.width), 'uint16')
            out_pad[top:bottom, left:right] = out
            out = out_pad
        return out

    def depth_color(self):
        """convert 1-channel depth to 3-channel depth"""
        return cv2.cvtColor(self.depth, cv2.COLOR_GRAY2RGB)

    def invalid_depth_mask(self, depth=None):
        """ return mask of invalid depth"""
        if depth is None: depth = self.depth
        return 255*(depth==0).astype('uint8')

    def draw_workspace(self, im):
        if self.workspace is None: return im
        for i in range(len(self.workspace.pts) - 1):
            cv2.line(im, tuple(self.workspace.pts[i]), tuple(self.workspace.pts[i+1]), (0,255,0), 2)
        cv2.line(im, tuple(self.workspace.pts[-1]), tuple(self.workspace.pts[0]), (0,255,0), 2)

        # bound_margin_locs = np.where(self.workspace.get_bound_margin())
        # im[bound_margin_locs] = (255,0,0)

        return im

    def disp(self, mode='rgb'):
        if self.hasRgb and not self.hasDepth:
            out = np.copy(self.rgb)
        if not self.hasRgb and self.hasDepth:
            out = ArrayUtils().repmat(self.depth_U8(), (1, 1, 3))
        if self.hasRgb and self.hasDepth:
            if mode=='rgb': out=np.copy(self.rgb)
            if mode=='depth': out = ArrayUtils().repmat(self.depth_U8(), (1, 1, 3))
            if mode=='depth_jet': out = cv2.applyColorMap(255 - self.depth_U8(), cv2.COLORMAP_JET)
            if mode=='depth_norm': out = ArrayUtils().get_mat_normal_map_U8(self.depth)

        out = self.draw_workspace(out)
        return out


    def show(self, title='viewer', mode='rgb'):
        """show rgbd"""
        cv2.imshow(title, self.disp(mode=mode)[:,:,::-1])

    def inpaint_depth(self, rad, depth=None):
        """inpaint depth"""
        if depth is None: depth = self.depth
        return cv2.inpaint(depth, self.invalid_depth_mask(depth=depth), rad, cv2.INPAINT_TELEA)

    def cx(self, x):   # fix x in image location range
        """ return value in range (0,width)"""
        return min(max(0, x), self.width)

    def cy(self, y):# fix y in image location range
        """ return value in range (0,height)"""
        return min(max(0, y), self.height)

    def fromdict(self, data_dict, only_copy=False):
        """ copy data from dict"""
        if only_copy: super().fromdict(data_dict=data_dict)
        else:
            assert 'rgb' in data_dict or 'depth' in data_dict
            if 'rgb' in data_dict: self.set_rgb(rgb=data_dict['rgb'])
            if 'depth' in data_dict: self.set_depth(depth=data_dict['depth'])
            if 'bg_rgb' in data_dict: self.set_bg_rgb(rgb=data_dict['bg_rgb'])
            if 'bg_depth' in data_dict: self.set_bg_depth(depth=data_dict(['bg_depth']))

    def get_crop_depth_edges(self, depth_scale_params):
        return cv2.Canny(ArrayUtils().reval(self.crop_depth(), scale_params=depth_scale_params, data_type='uint8'), 100, 200)

    def get_rgb_edges(self):
        edge = np.zeros((self.height, self.width), 'uint8')
        for ch in range(3):
            edge += cv2.Canny(self.rgb[:,:,ch], 100, 200)
        return (edge>0).astype('uint8')


    def val(self, pt):
        loc = pt[::-1]
        val = ()
        if self.hasRgb: val += tuple(v for v in self.rgb[loc])
        if self.hasDepth: val += (self.depth[loc], )
        return val

    def val_str(self, pt):
        val = self.val(pt=pt)
        text = ''
        for v in val: text += '{}.'.format(v)
        return text[:-1]

    def pt_in_im_range(self, pt, pad_size=(0,0)):
        xp, yp = pad_size
        w2, h2 = xp//2 +1, yp//2+1
        if pt[0]<w2 or (self.width-w2)<=pt[0]: return False
        if pt[1]<h2 or (self.height-h2)<=pt[1]: return False
        return True

    def depth_hist(self, normed=True):
        histSize = self.depth_max - self.depth_min + 1
        depth_crop = self.crop_depth()
        hist, bins = np.histogram(depth_crop.ravel(), histSize, [self.depth_min, self.depth_max], normed=normed)
        # hist = cv2.calcHist([depth_crop], [histSize], None, [histSize], [self.depth_min, self.depth_max])
        return hist

    def workspace_grid(self, partitions=(5,5)):
        h, w = self.workspace.height, self.workspace.width

        px, py = partitions
        wp, hp = int(np.ceil(w / px)), int(np.ceil(h / py))

        X = list(self.workspace.left + wp*np.arange(0,px)) +  [self.workspace.right,]
        Y = list(self.workspace.top + hp*np.arange(0,py)) +  [self.workspace.bottom,]

        ver_line = [[(x,self.workspace.top), (x,self.workspace.bottom)] for x in X]
        hor_line = [[(self.workspace.left, y), (self.workspace.right, y)] for y in Y]

        return (ver_line, hor_line)

    def invalid_depth_rate(self):
        return (np.mean((self.depth==0).astype('float32')))

    # def im2camLoc(self, pts, focal, center=None):
    #     if not self.hasDepth: return None
    #     fx, fy = focal
    #     if center is None: center = (self.width/2, self.height/2)
    #     xc, yc = center
    #     pts = np.array(pts).reshape((-1,2))
    #     pts[:,0] -= xc
    #     pts
    #
    #
    #     x, y = loc
    #     x, y = x - xc, y - yc
    #     Z = self.depth[tuple(loc)]
    #     X, Y = x/(fx*Z+0.0000001), y/(fy*Z+0.0000001)
    #     return (X,Y,Z)









