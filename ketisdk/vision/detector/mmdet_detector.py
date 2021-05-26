from ttcv.import_basic_utils import *
import cv2
import numpy as np
from mmdet.apis import  init_detector, inference_detector
from mmdet.core.mask import utils
import pycocotools.mask as maskUtils

class BaseMmdet():
    def get_model(self, cfg_file, ckpt, device='cuda:0'):
        self.model = init_detector(cfg_file, ckpt, device=device)

    def run(self, im, thresh=None, classes=None):
        pred = inference_detector(self.model, im)
        boxes_list, masks_list = pred
        out = dict()
        num_cls = len(boxes_list)
        if classes is None: classes = list(range(num_cls))
        boxes_, scores_, classes_, masks_ = [], [], [], []
        for j, boxes in enumerate(boxes_list):
            if thresh is not None: locs = np.where(boxes[:,-1]>thresh)
            bbs = boxes if thresh is None else boxes[locs]
            ms = masks_list[j] if thresh is None else [masks_list[j][i] for i in locs[0]]

            boxes_.append(bbs[:,:4])
            scores_.append(bbs[:,-1])
            classes_+= [classes[j],]*len(bbs[:,-1])
            masks_ += [maskUtils.decode(m).astype(np.uint8) for m in ms]
        boxes_ = np.vstack(boxes_)
        scores_ = np.vstack(scores_).reshape((-1, 1))
        out.update({'boxes': boxes_.astype('int')})
        out.update({'scores': scores_})
        out.update({'classes': classes_})
        out.update({'masks': masks_})

        return out

    def run_extract(self, predictions):
        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores.numpy() if predictions.has("scores") else None
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints.numpy() if predictions.has("pred_keypoints") else None
    
from .detector_base import BaseDetector
class MmdetDetector(BaseDetector, BaseMmdet):
    def get_model(self, cfg_file, ckpt):
        BaseMmdet.get_model(self, cfg_file=cfg_file, ckpt=ckpt)

    def run(self, im, thresh=None, classes=None):
        return BaseMmdet.run(self,im,thresh=thresh, classes=classes)

from ttcv.basic.basic_objects import DetGuiObj
class MmdetDetectorGuiObj(MmdetDetector, DetGuiObj):
    def get_model(self):
        super().get_model(cfg_file=self.args.cfg_file, ckpt=self.args.ckpt)

    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb'):
        if method_ind==0:
            out = self.show_predict(rgbd.rgb,self.predict_array(rgbd.rgb, thresh=0.5))
            ret = {'im':out}
        return ret

def test_base_mmdet(cfg_file, ckpt):
    detector = BaseMmdet()
    detector.get_model(cfg_file=cfg_file, ckpt=ckpt)

def test_mmdet_gui(cfg_file, ckpt, run_realsense=False):
    # detection module
    args = CFG()
    setattr(args, 'cfg_file', cfg_file)
    setattr(args, 'ckpt', ckpt)
    from ketisdk.gui.gui import GUI,GuiModule
    module = GuiModule(MmdetDetectorGuiObj, type='mmdetector', category='detector', args=args)

    # realsense module
    realsense_modules = []
    if run_realsense:
        from ketisdk.vision.sensor.realsense_sensor import get_realsense_modules
        realsense_modules += get_realsense_modules()

    #
    GUI(title='Mmdetetion GUI', modules=[module,]+realsense_modules)


    
    







