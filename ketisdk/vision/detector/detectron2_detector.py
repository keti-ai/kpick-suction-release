from ttcv.import_basic_utils import *
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import numpy as np

class BaseDetectron():
    def get_model(self, cfg_file, score_thresh=0.5):
        cfg = get_cfg()
        cfg.merge_from_file(cfg_file)
        # cfg.DATASETS.TEST = self.args.test_dataset
        # cfg.DATALOADER.NUM_WORKERS = self.args.num_workers
        # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.args.batch_size_per_image
        # cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.args.num_classes  # only has one class (ballon)
        # cfg.MODEL.ANCHOR_GENERATOR.SIZES = self.args.anchor_sizes
        # cfg.MODEL.WEIGHTS = self.args.checkpoint
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.args.score_thresh_test

        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = score_thresh
        cfg.freeze()

        self.model = DefaultPredictor(cfg)

    def run(self, im, thresh=None, classes=None):
        pred = self.model(im)
        out = dict()
        if 'instances' in pred:
            inst = pred["instances"].to("cpu")
            if inst.has("pred_boxes"): out.update({'boxes': inst.pred_boxes.tensor.numpy()})
            if inst.has("scores"): out.update({'scores': inst.scores.numpy().reshape((-1,1))})
            if inst.has("pred_classes"): out.update({'classes': inst.pred_classes.numpy()})
            if inst.has("pred_keypoints"): out.update({'keypoints': inst.pred_keypoints.numpy()})
            if inst.has("pred_masks"): out.update({'masks': inst.pred_masks.numpy()})
        if 'sem_seg' in pred:
            sem_seg = pred["sem_seg"].argmax(dim=0).to("cpu").numpy()
            out.update({'sem_seg': sem_seg})
        return out

    def run_extract(self, predictions):
        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores.numpy() if predictions.has("scores") else None
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints.numpy() if predictions.has("pred_keypoints") else None
    
from .detector_base import BaseDetector
class DetectronDetector(BaseDetector, BaseDetectron):
    def get_model(self, cfg_file, score_thresh=0.5):
        BaseDetectron.get_model(self, cfg_file=cfg_file, score_thresh=score_thresh)

    def run(self, im, thresh=None, classes=None):
        return BaseDetectron.run(self,im,thresh=thresh, classes=classes)

from ttcv.basic.basic_objects import DetGuiObj
class DetectronDetectorGuiObj(DetectronDetector, DetGuiObj):
    def get_model(self):
        super().get_model(cfg_file=self.args.cfg_file, score_thresh=self.args.score_thresh)

    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb'):
        if method_ind==0:
            ret = self.predict_array(rgbd.bgr())
            out = self.show_predict(rgbd.rgb,ret, self.args.kpt_skeleton)
        return out

def test_base_detectron(cfg_file, score_thresh=0.5):
    detector = BaseDetectron()
    detector.get_model(cfg_file=cfg_file, score_thresh=score_thresh)

def test_detectron_gui(cfg_file, score_thresh=0.5, kpt_skeleton=None, run_realsense=False):
    # detection module
    args = CFG()
    setattr(args, 'cfg_file', cfg_file)
    setattr(args, 'score_thresh', score_thresh)
    setattr(args, 'kpt_skeleton', kpt_skeleton)
    from ttcv.basic.basic_gui import BasGUI,GuiModule
    module = GuiModule(DetectronDetectorGuiObj, type='detectron detector', category='detector', args=args)

    # realsense module
    realsense_modules = []
    if run_realsense:
        from ketisdk.vision.sensor.realsense_sensor import get_realsense_modules
        realsense_modules += get_realsense_modules()

    #
    BasGUI(title='Detectron GUI', modules=[module,]+realsense_modules)


    
    







