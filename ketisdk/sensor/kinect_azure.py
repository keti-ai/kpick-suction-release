from libs.utils.proc_utils import *
import open3d as o3d
from libs.import_basic_utils import *
from libs.sensor.sensor import Sensor
import cv2
from glob import glob

class KASensor(Sensor):
    def start(self):            #
        # camera cofigs
        self.flag_exit = False
        self.align_depth_to_color = self.args.align_depth_to_color
        if not hasattr(self.args, 'ka_config'): config = o3d.io.AzureKinectSensorConfig()
        else: config = o3d.io.read_azure_kinect_sensor_config(self.args.ka_config)

        self.sensor = o3d.io.AzureKinectSensor(config)
        if not self.sensor.connect(self.args.device):
            raise RuntimeError('Failed to connect to sensor')

        glfw_key_escape = 256
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(glfw_key_escape, self.stop)
        self.vis.create_window('image', self.args.window_size[1], self.args.window_size[0])

        print('='*50+'Sensor initialized ...')

    def stop(self):        #
        self.flag_exit = True
        print('=' * 50 + 'Sensor  terminated...')
        return False

    def get_rgbd(self, **kwargs):
        rgbd = self.sensor.capture_frame(self.align_depth_to_color)
        if rgbd is None: return None
        rgb = cv2.cvtColor(np.asarray(rgbd.color), cv2.COLOR_RGB2BGR)
        depth = np.asarray(rgbd.depth)

        return super().get_rgbd(rgb=rgb, depth=depth)

if __name__ == '__main__':
    cfg_path = 'configs/sensor/kinect_azure.cfg'
    realsense = KASensor(cfg_path).run()