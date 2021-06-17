
def demo_suction_gui(cfg_path):
    from ketisdk.vision.detector.pick.suction.suction_detection import SuctionGuiDetector
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.gui.gui import GUI, GuiModule

    detect_module = GuiModule(SuctionGuiDetector, type='suction_detector', name='Suction Detector',
                              category='detector', cfg_path=cfg_path, num_method=6)

    GUI(title='Grip Detection GUI',
        modules=[detect_module,] + get_realsense_modules(),
        )

def demo_suction_without_gui(cfg_path, rgb_path, depth_path, ws_pts=None):
    from ketisdk.vision.detector.pick.suction.suction_detection import SuctionDetector
    import cv2
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

    # Set viewer
    cv2.namedWindow('viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('viewer', 1080, 720)

    # Load image
    rgb = cv2.imread(rgb_path)[:, :, ::-1]
    depth = cv2.imread(depth_path)[:, :, 0]
    rgbd = RGBD(rgb=rgb, depth=depth)

    # Set workspace
    # workspace_pts = [(550, 100), (840, 100), (840, 600), (550, 600)]
    if ws_pts is not None: rgbd.set_workspace(pts=ws_pts)

    # Initialize network
    detector = SuctionDetector(cfg_path=cfg_path)

    # Run
    ret = detector.detect_and_show_poses(rgbd=rgbd)

    # Show
    cv2.imshow('viewer', ret['im'][:,:,::-1])
    if cv2.waitKey()==27: exit()

if __name__=='__main__':
    cfg_path = 'configs/pick/suction_net.cfg'
    demo_suction_gui(cfg_path=cfg_path)
