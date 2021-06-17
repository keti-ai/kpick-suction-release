import os, sys
sys.path.append(os.getcwd())

import pyrealsense2 as rs
# from ..sensor.sensor import Sensor
import numpy as np
from ketisdk.gui.gui import GuiModule, GUI
from ..sensor.sensor import Sensor
from ketisdk.vision.utils.rgbd_utils_v2 import RGBD


class RSSensor(Sensor):
    def get_device_sn(self):
        realsense_ctx = rs.context()
        for i in range(len(realsense_ctx.devices)):
            detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)

            print(detected_camera)
    def start(self, device_serial=None, size=(1280, 720)):

        if device_serial is not None:
            self.start_(device_serial=device_serial)
        else:
            realsense_ctx = rs.context()
            for i in range(len(realsense_ctx.devices)):
                detected_serial = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
                try:
                    self.start_(device_serial=detected_serial, size=size)
                    break
                except:
                    pass


    def start_(self, device_serial, size=(1280, 720)):
        # >>>>>>>>> Configs for REALSENSE CAMERA
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device_serial)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, 15)

        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(config)

        intrinsic_params = self.get_intrinsic_params(profile)
        print('intrinsic params:')
        print(intrinsic_params)

        print('sensor initialized ... ')

    def stop(self):
        self.pipeline.stop()
        print('sensor terminated ... ')

    def get_data(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()  # bgr
        rgb = np.asanyarray(color_frame.get_data())[:, :, ::-1]

        depth_frame = aligned_frames.get_depth_frame()
        depth = np.asanyarray(depth_frame.get_data())

        return rgb, depth

    def get_rgbd(self, workspace=None, depth_min=300, depth_max=1200, rot_90=False):
        rgb, depth = self.get_data()
        if rot_90: rgb, depth = np.rot90(rgb), np.rot90(depth)
        if rgb is None and depth is None: return None
        rgbd =  RGBD(rgb=rgb, depth=depth,workspace=workspace,depth_min=depth_min, depth_max=depth_max)
        return rgbd

    def get_intrinsic_params(self, profile):
        return  profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()



def get_valid_realsenses():
    realsense_ctx = rs.context()
    valid_serials = []
    for i in range(len(realsense_ctx.devices)):
        detected_serial = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
        try:
            config = rs.config()
            config.enable_device(detected_serial)
            valid_serials.append(detected_serial)
        except:
            pass

    print('{} {} realsense detected: {}'.format('+'*10, len(valid_serials), valid_serials))
    return valid_serials


def get_realsense_modules():
    RS_serials = get_valid_realsenses()
    RS_modules = []
    for j, serial in enumerate(RS_serials):
        RS_modules.append(GuiModule(RSSensor, type='realsense', name='Realsense%d' % j, short_name='RS%d'%j, category='vision_sensor', serial=serial))
    return RS_modules

def run_realsense_gui():
    GUI(title='Realsense', modules=get_realsense_modules())


if __name__ == '__main__':
    # cfg_path = 'configs/sensor/realsense.cfg'
    # # cfg_path = 'configs/grasp_detection/grasp_v2.cfg'
    # RSSensor(cfg_path=cfg_path).run()
    cfg_path = 'configs/sensor/realsense.cfg'

    GUI(modules=get_realsense_modules())

