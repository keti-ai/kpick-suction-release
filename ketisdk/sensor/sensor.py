
class Sensor():
    def start(self, device_serial=None, size=(1280, 720)):
        pass
    
    def stop(self):
        pass

    def get_rgbd(self, workspace=None, depth_min=300, depth_max=1200):
        pass


if __name__ == '__main__':
    cfg_path = 'configs/DDI/grip_kinect_azure_hcr.cfg'
    realsense = Sensor(cfg_path).run()
