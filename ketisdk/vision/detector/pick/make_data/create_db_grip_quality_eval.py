from libs.utils.data_processing import *
from libs.policy.policy import *
from libs.policy.sequence_show_predict import SeqPredictor
from libs.policy.policy import GenDetector
from libs.dataset.polygon import *
from libs.dataset.grip import GripDBer


def detect_sequence_local(cfg_path):
    drawer = OnmousePolygon(cfg_path=cfg_path)
    SeqPredictor(cfg_path=cfg_path, detector=GenDetector(cfg_path=cfg_path, detector=drawer.get_polygon,
                                                     show_ret_func=show_polygon).predict_show, detector_reload=drawer.load_params).run()
    # drawer = DrawPolyon(cfg_path=cfg_path)
    # SequencePredictor(cfg_path, detector=GenDetector(cfg_path=cfg_path, detector=drawer.get_ann,
    #                                                  show_ret_func=show_polygon).predict_show, detector_reload=drawer.load_params).run()

    # drawer = GripDBer(cfg_path=cfg_path)
    # SeqPredictor(cfg_path=cfg_path, detector=GenDetector(cfg_path=cfg_path, detector=drawer.poly2maskAug).predict_show,
    #                   detector_reload=drawer.load_params).run()
if __name__ == '__main__':
    detect_sequence_local('configs/grasp_detection/grip_evaluator.cfg')


