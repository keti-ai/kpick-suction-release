import cv2
from ketisdk.utils.proc_utils import ProcUtils
import os
import math
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

class ArrayUtils():
    def crop_oriented_rect_polar(self, im, center, angle, rx, ry):
        xc, yc = center
        # top, left, right, bottom = xc-rx, yc-ry, xc+rx, yc+ry
        dx,dy = ProcUtils().rotateXY_float(-rx, ry, angle)
        pt0 = (int(xc+dx), int(yc+dy))
        dx,dy = ProcUtils().rotateXY_float(-rx, -ry, angle)
        pt1 = (int(xc+dx), int(yc+dy))
        dx,dy = ProcUtils().rotateXY_float(rx, -ry, angle)
        pt2 = (int(xc+dx), int(yc+dy))
        dx,dy = ProcUtils().rotateXY_float(rx, ry, angle)
        pt3 = (int(xc+dx), int(yc+dy))

        return self.crop_oriented_rect(im, (pt0, pt1, pt2, pt3))




    def crop_oriented_rect(self, im, oriented_box):
        # points for test.jpg
        pt0, pt1, pt2, pt3 = oriented_box
        cnt = np.array([[list(pt0)], [list(pt1)], [list(pt2)], [list(pt3)]])
        rect = cv2.minAreaRect(cnt)

        # the order of the box points: bottom left, top left, top right,
        # bottom right
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(im, M, (width, height))
        return warped




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

    def save_array_v3(self, array, filepath=None):
        if filepath is None: filepath = ProcUtils().get_time_name() + '.png'
        fold, _ = os.path.split(filepath)
        if not os.path.exists(fold): os.makedirs(fold)
        cv2.imwrite(filepath, array)


    def get_mat_normals(self, mat_in, grad_weight=True):
        mat = mat_in.astype(np.float)
        h, w = mat.shape[:2]
        X, Y = np.meshgrid(np.arange(0, w), np.arange(0, h))

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
        v[:, :, :, 0] = self.get_3point_normals(M00, Mm1m1, M0m1)
        v[:, :, :, 1] = self.get_3point_normals(M00, M0m1, M1m1)
        v[:, :, :, 2] = self.get_3point_normals(M00, M1m1, M10)
        v[:, :, :, 3] = self.get_3point_normals(M00, M10, M11)
        v[:, :, :, 4] = self.get_3point_normals(M00, M11, M01)
        v[:, :, :, 5] = self.get_3point_normals(M00, M01, Mm11)
        v[:, :, :, 6] = self.get_3point_normals(M00, Mm11, Mm10)
        v[:, :, :, 7] = self.get_3point_normals(M00, Mm10, Mm1m1)
        v_mean = np.mean(v, axis=3)

        v_norm = np.linalg.norm(v_mean, axis=2)
        v_norm = self.repmat(v_norm, (1, 1, 3))
        v_norm = np.divide(v_mean, v_norm)
        v = np.zeros((h, w, 3), np.float)
        v[1:-1, 1:-1, :] = v_norm

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

    def get_3point_normals(self, mat0, mat1, mat2):
        v = np.cross(mat1 - mat0, mat2 - mat0)
        v_norm = np.linalg.norm(v, axis=2)
        # v_norm = np.expand_dims(v_norm, axis=2)
        v_norm = self.repmat(v_norm, (1, 1, 3))
        return np.divide(v, v_norm)

    def truncate(self, input, vmax=None, vmin=None):
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

        step = round((end-start)/n)
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


    # def set_value_to_array(self,anArray, locs, value):
    #
    #
    #     Y, X = locs
    #     num_loc = len(Y)
    #     num_cpus = os.cpu_count()
    #     ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    #     range_list = self.get_range_partition(0, num_loc, num_cpus)
    #     outArray = np.copy(anArray)
    #
    #     # anArray_id = ray.put(anArray)
    #     outArray_id = ray.put(outArray)
    #     X_id, Y_id = ray.put(X), ray.put(Y)
    #
    #     for arange in range_list:
    #         ray.get(set_value_to_array_ray.remote(outArray_id, X_id, Y_id, value, arange[0], arange[1]))
    #
    #     return outArray

#
# @ray.remote
# def set_value_to_array_ray(outArray, X, Y, value, start, end):
#     outArray.flags.writeable = True
#     locs = (Y[start:end], X[start:end])
#     outArray[locs] = value




if __name__ == '__main__':
    hist = np.arange(200, 0, -10)

    hmin, hmax = np.amin(hist), np.amax(hist)



    aa = 1

    # # now display the array X as an Axes in a new figure
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111, frameon=False)
    # ax2.imshow(X)
    # plt.show()





