
def demo_suction_gui(cfg_path):
    from ketisdk.vision.detector.pick.suction.suction_detection import SuctionGuiDetector
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.gui.gui import GUI, GuiModule

    detect_module = GuiModule(SuctionGuiDetector, type='suction_detector', name='Suction Detector',
                              category='detector', cfg_path=cfg_path, num_method=6)

    GUI(title='Grip Detection GUI',
        modules=[detect_module,] + get_realsense_modules(),
        )

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

if __name__=='__main__':
    # cfg_path = 'configs/pick/suction_net.cfg'
    # demo_suction_gui(cfg_path=cfg_path)
    demo_suction_without_gui()

