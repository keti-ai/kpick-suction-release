![license](https://img.shields.io/badge/license-MIT-green) ![PyTorch-1.4.0](https://img.shields.io/badge/PyTorch-1.4.0-blue)
# Keti SDK
## System Requirements:
```sh
- Ubuntu 16.04 or 18.04
- CUDA >=10.0, CUDNN>=7
```
```sh
- Pytorch >=1.4.0
```
## Install
```sh
sudo apt install python3.6-dev
sudo apt install python3.6-tk
cd $ROOT
pip install -e .
```
## Checkpoint
``` sh
mkdir data/model/pick/suction_evaluator/resnet20_32x32x6_20200616_RB_CH_cmb_0519_0522_0526_0604_0605_norm_vec
```
```sh
https://drive.google.com/file/d/1gq5uJPu5E8rjrBqht_gnGSWr76_aiYcN/view?usp=sharing
```
** To change the checkpoint path, refer to checkpoint_dir option in suction_net.cfg file

## HOW TO USE
```sh
def demo_suction_gui(cfg_path):
    from ketisdk.vision.detector.pick.suction.suction_detection import SuctionGuiDetector
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.gui.gui import GUI, GuiModule

    detect_module = GuiModule(SuctionGuiDetector, type='suction_detector', name='Suction Detector',
                              category='detector', cfg_path=cfg_path, num_method=6)

    GUI(title='Grip Detection GUI',
        modules=[detect_module,] + get_realsense_modules(),
        )
```
```sh
def demo_suction_without_gui():
    from ketisdk.vision.detector.pick.suction.suction_detection import SuctionDetector
    import cv2
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

    # path
    cfg_path = 'configs/pick/suction_net.cfg'
    rgb_path = 'test_data/rgb.png'
    depth_path = 'test_data/depth.png'

    # set polygon for detection space
    ws_pts = [(177, 696), (76, 19), (1160, 8), (1067, 687), (906, 690), (776, 566), (727, 564), (724, 604), (766, 693)]

    # Initialize network
    detector = SuctionDetector(cfg_path=cfg_path)
    print('Network initialized...')

    # change configurations if needed
    detector.args.net.test_batch = 1024
    detector.args.net.pad_sizes = [(30, 30), (50, 50)]
    detector.args.net.stride = 10
    detector.args.net.score_thresh = 0.7
    detector.args.net.do_suc_at_center = True

    # Set viewer
    cv2.namedWindow('viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('viewer', 1080, 720)

    # Load image
    rgb = cv2.imread(rgb_path)[:, :, ::-1]
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    rgbd = RGBD(rgb=rgb, depth=depth)

    # Set workspace
    rgbd.set_workspace(pts=ws_pts)

    # Run
    ret = detector.detect_and_show_poses(rgbd=rgbd)

    # Show
    cv2.imshow('viewer', ret['im'][:,:,::-1])
    if cv2.waitKey()==27: exit()
```



















