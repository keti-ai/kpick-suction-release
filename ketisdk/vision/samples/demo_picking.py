
def demo_grip_3finger():
    from kpick.processing.grip_3fingers import ThreeFingerGraspGui
    from ketisdk.gui.gui import GUI, GuiModule
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.sensor.openni_sensor import get_openni_sensor_modules
    detect_module = GuiModule(ThreeFingerGraspGui, name='3finger gripper',
                              cfg_path='configs/grip_3finger.cfg')
    GUI(title='3 Finger Grasp Detection', modules=[detect_module]+get_realsense_modules()+get_openni_sensor_modules())

def demo_grip_gui():
    from kpick.processing.grip_detection_v8 import GripGuiDetector
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.sensor.openni_sensor import get_openni_sensor_modules
    from ketisdk.gui.gui import GUI, GuiModule

    cfg_path = 'configs/pick/grip_net.cfg'
    detect_module = GuiModule(GripGuiDetector, type='grip_detector', name='Grip Detector',
                              category='detector', cfg_path=cfg_path, num_method=2)

    GUI(title='Grip Detection GUI',
           modules=[detect_module] + get_realsense_modules()+get_openni_sensor_modules(),
           )

def demo_suction_gui():
    from ketisdk.vision.detector.pick.suction_detection import SuctionGuiDetector
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.sensor.openni_sensor import get_openni_sensor_modules
    from ketisdk.gui.gui import GUI, GuiModule
    from ketisdk.vision.detector.pick.grip_detection import GripGuiDetector


    cfg_path = 'configs/pick/suction_net.cfg'
    detect_module = GuiModule(SuctionGuiDetector, type='suction_detector', name='Suction Detector',
                              category='detector', cfg_path=cfg_path, num_method=6)

    cfg_path = 'configs/pick/grip_net.cfg'
    detect_module1 = GuiModule(GripGuiDetector, type='grip_detector', name='Grip Detector',
                              category='detector', cfg_path=cfg_path, num_method=6)

    GUI(title='Grip Detection GUI',
        modules=[detect_module,detect_module1] + get_realsense_modules() + get_openni_sensor_modules(),
        )

def demo_grip_without_gui(cfg_path, rgb_path, depth_path):
    from ketisdk.vision.detector.pick.grip_detection import GripDetector
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
    workspace_pts = [(550, 100), (840, 100), (840, 600), (550, 600)]
    rgbd.set_workspace(pts=workspace_pts)

    # Initialize network
    detector = GripDetector(cfg_path=cfg_path)

    # Run
    ret = detector.detect_and_show_poses(rgbd=rgbd)

    # Show
    cv2.imshow('viewer', ret['im'][:,:,::-1])
    if cv2.waitKey()==27: exit()

if __name__=='__main__':
    # demo_grip_3finger()
    # demo_grip_gui()
    demo_suction_gui()
    # demo_grip_without_gui(cfg_path='configs/pick/grip_net.cfg',
    #                       rgb_path='/mnt/workspace/000_data/picking/test_images/210415/image/20210415152544262140.png',
    #                       depth_path='/mnt/workspace/000_data/picking/test_images/210415/depth/20210415152544262140.png',
    #                       )
    # from kpick.processing.suction_detection_v3 import test_suction_detector
    # test_suction_detector(cfg_path='configs/pick/suction_net.cfg')