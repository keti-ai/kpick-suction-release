import os, sys
sys.path.append(os.getcwd())
import numpy as np
from ketisdk.gui.gui import GuiModule, GUI
from ..sensor.sensor import Sensor
from ketisdk.vision.utils.rgbd_utils_v2 import RGBD
from openni import openni2
from openni import _openni2

class OpenniSensor(Sensor):
    def start(self, device_serial=None, size=(1280, 720)):
        openni2.initialize()
        self.dev = openni2.Device.open_any()

        if self.dev.has_sensor(openni2.SENSOR_COLOR):
            self.stream = self.dev.create_stream(openni2.SENSOR_COLOR)
            sensor_info = self.stream.get_sensor_info()

            for videoMode in sensor_info.videoModes:
                if videoMode.pixelFormat == _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888:
                    self.stream.set_video_mode(videoMode)
                    break
            self.stream.start()

        if self.dev.has_sensor(openni2.SENSOR_DEPTH):
            self.stream1 = self.dev.create_stream(openni2.SENSOR_DEPTH)
            sensor_info1 = self.stream1.get_sensor_info()
            self.stream1.set_video_mode(sensor_info1.videoModes[len(sensor_info1.videoModes) - 1])
            self.stream1.start()

    def stop(self):
        if hasattr(self,'stream'): self.stream.stop()
        if hasattr(self,'stream1'):self.stream1.stop()
        self.dev.close()
        print('sensor terminated ... ')

    def get_data(self):
        rgb, depth=None, None
        if hasattr(self,'stream'):
            frame = self.stream.read_frame()
            rgb = np.array(frame.get_buffer_as_triplet()).reshape([frame.height, frame.width, 3])
        if hasattr(self,'stream1'):
            frame = self.stream1.read_frame()
            depth = np.array(frame.get_buffer_as_uint16()).reshape([frame.height, frame.width])
        return rgb, depth

    def get_rgbd(self, workspace=None, depth_min=300, depth_max=1200, rot_90=False):
        rgb, depth = self.get_data()
        # if rot_90: rgb, depth = np.rot90(rgb), np.rot90(depth)
        if rgb is None and depth is None: return None
        return RGBD(rgb=rgb, depth=depth,workspace=workspace,depth_min=depth_min, depth_max=depth_max)

def get_openni_sensor_modules():
    return [GuiModule(OpenniSensor, type='openni_sensor', name='Openni', short_name='OPI',
                      category='vision_sensor', serial=None),]



if __name__ == '__main__':
    pass
