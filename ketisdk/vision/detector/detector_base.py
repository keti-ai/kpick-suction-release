import numpy as np
from time import time
import cv2
from ttcv.import_basic_utils import *

class BaseDetector():
    def get_model(self, **kwargs):
        self.model = Nones

    def run(self, im, thresh=None, classes=None):
        pass


    def predict_array(self, anArray, need_rot=False, test_shape=None, lefttop=(0,0), classes=None,
                      thresh=None, sort=None):
        h,w = anArray.shape[:2]
        if need_rot: im_rot = cv2.rotate(anArray, cv2.ROTATE_90_CLOCKWISE)           # 90 rot
        else: im_rot = anArray

        need_reshape = (test_shape is not None)
        if need_reshape:
            h1, w1 = im_rot.shape[:2]
            im_scale = cv2.resize(im_rot, test_shape, interpolation=cv2.INTER_CUBIC)
            h2, w2 = im_scale.shape[:2]
            fx, fy = w1/w2, h1/h2       # restore ratios
        else: im_scale = im_rot

        ret = self.run(im_scale, thresh=thresh, classes=classes)

        out = dict()
        scores = ret['scores']
        if sort is not None:
            if sort=='des': inds = np.argsort(np.array(scores.flatten()))[::-1]
            if sort == 'inc': inds = np.argsort(np.array(scores.flatten()))
            inds = inds.tolist()
            out.update({'scores': scores[inds, :]})
            out.update({'classes': [ret['classes'][j] for j in inds]})
        else:
            inds = list(range(len(scores)))
            out.update({'scores': scores})
            out.update({'classes': ret['classes']})

        if 'boxes' in ret:
            boxes = ret['boxes'][inds, :]
            if need_reshape: boxes[:, [0, 2]], boxes[:, [1, 3]] = boxes[:, [0, 2]] * fx, boxes[:, [1, 3]] * fy
            if need_rot:
                boxes = boxes[:, [1, 2, 3, 0]]
                boxes[:, [1, 3]] = h - boxes[:, [1, 3]]
            if lefttop != (0, 0): boxes[[0, 2], :], boxes[[1, 3], :] = boxes[[0, 2], :] + lefttop[0], boxes[[1, 3], :] + \
                                                                       lefttop[1]
            out.update({'boxes': boxes})

        if 'masks' in ret:
            masks = ret['masks']
            if need_reshape: pass
            if need_rot: pass
            if lefttop != (0, 0): pass
            out.update({'masks': masks})

        if 'keypoints' in ret:
            keypoints = ret['keypoints'][inds, :,:].reshape((1, -1, 3))
            if need_reshape: keypoints[:, :, 0], keypoints[:, :, 1] = keypoints[:, :, 0]*fx, keypoints[:, :, 1]*fy
            if need_rot:
                keypoints = keypoints[:,:,[1,0,2]]
                keypoints[:,:,1] = h - keypoints[:,:,1]
            if lefttop != (0,0): keypoints[:,:,0], keypoints[:,:,1] = keypoints[:,:,0]+lefttop[0], keypoints[:,:,1]+lefttop[1]
            out.update({'keypoints': keypoints})

        return out

    def show_predict(self, im, predict, kpt_skeleton=None, colors=None, line_thick=2, text_scale=1, text_thick=2, marker_size=5):
        out = np.copy(im)
        boxes = predict['boxes'] if 'boxes' in predict else None
        scores = predict['scores'] if 'scores' in predict else None
        classes = predict['classes'] if 'classes' in predict else None
        masks = predict['masks'] if 'masks' in predict else None
        keypoints = predict['keypoints'] if 'keypoints' in predict else None

        if boxes is not None:
            out = self.show_boxes(out,boxes, labels=classes,scores=scores, colors=colors, line_thick=line_thick,
                                  text_scale=text_scale, text_thick=text_thick)
        if masks is not None:
            out = self.show_mask(out, masks=masks, colors=colors)
        if keypoints is not None:
            out = self.show_keypoints(out, keypoints, skeleton=kpt_skeleton, marker_size=marker_size, text_scale=text_scale)
        return out

    def show_keypoint_inst(self, im, kpts, scores=None, skeleton=None, thresh=0.5, marker_size=5, text_scale=1):
        out = np.copy(im)
        kpts = kpts.astype('int')
        for j,pt in enumerate(kpts):
            x,y = pt
            if scores is not None:
                if scores[j]<thresh: continue
            cv2.drawMarker(out, (x,y), (0,0,255), cv2.MARKER_TILTED_CROSS, marker_size)
            cv2.putText(out, 'k%d'%j, (x,y), cv2.FONT_HERSHEY_COMPLEX, text_scale, (0,0,255))

        if skeleton is not None:
            for link in skeleton:
                i1, i2 = link[0] - 1, link[1] - 1
                if scores is not None:
                    if scores[i1] < thresh or scores[i2]<thresh: continue
                x1, y1 = kpts[i1, :]
                x2, y2 = kpts[i2, :]
                cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return out

    def show_keypoints(self, im, keypoints, skeleton=None, thresh=0.5, marker_size=5, text_scale=1):
        out = np.copy(im)
        for kpts in keypoints:
            out = self.show_keypoint_inst(out, kpts=kpts[:,:2], scores=kpts[:,-1], skeleton=skeleton,
                                          thresh=thresh, marker_size=marker_size, text_scale=text_scale)
        return out


    def show_boxes(self, im, boxes, labels=None, scores=None, colors=None, line_thick=2, text_scale=1, text_thick=2):
        out = np.copy(im)
        for j, box in enumerate(boxes):
            if colors is None: color = (0,255,0)
            else: color = colors[j]

            left, top, right, bottom = box[:4]
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            out = cv2.rectangle(out,(left,top),(right,bottom), color, line_thick)

            txt = ''
            if labels is not None: txt = str(labels[j])
            if scores is not None:
                score= round(float(scores[j]),ndigits=3)
                if len(txt)==0: txt = str(score)
                else:  txt = f'{txt}:{score}'

            if len(txt)>0:
                out = cv2.putText(out, txt, (left, top), cv2.FONT_HERSHEY_COMPLEX,
                              text_scale, color, text_thick)
        return out

    def show_mask(self, im, masks, colors=None):
        out = np.copy(im)
        for j,mask in enumerate(masks):
            locs = np.where(mask>0)
            if colors is None: color = ProcUtils().get_color(j)
            else: color=colors[j]
            out[locs] = 0.7*out[locs] + tuple(0.3*np.array(color))
        return out









