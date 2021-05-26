from libs.detector.cifar_classfication.classification_v2 import BasCifarClassfier
import torch

if __name__=='__main__':
    cfg_path ='configs/grasp_detection/suction_net.cfg'
    evaluator = BasCifarClassfier(cfg_path=cfg_path)
    evaluator.trainval()
