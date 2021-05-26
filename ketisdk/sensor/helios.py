

import os
import numpy as np
import cv2

from arena_api.system import system
from KAI.utils.data_processing import RGBD, get_grip_from_rgbd
from KAI.sensor.sensor import Sensor
import time
from KAI.utils.proc_utils import get_dtype_name


class HLOSensor(Sensor):
    def start(self):
        # Get connected devices ---------------------------------------------------
        devices = system.create_device()
        print(f'Created {len(devices)} device(s)')
        try:
            self.device = devices[0]
        except:
            print('No device found!')
            exit()
        print(f'Device used in the example:\n\t{self.device}')

        # Get nodes ---------------------------------------------------------------
        nodes = self.device.nodemap.get_node(['Width', 'Height', 'PixelFormat'])
        self.height, self.width = nodes['Height'].value, nodes['Width'].value

        print('Depth image size: %d x %d' % (self.height, self.width))

        nodes['PixelFormat'].value = self.args.pixel_format
        print('-------------------------------------------Starting stream')
        self.device.start_stream(1)
        print('---------------------->>>>>>>>>> Sensor initialized ....')
        self.get_sensor_specs()

    def get_sensor_specs(self):
        image = self.device.get_buffer()  # optional args
        for i in ['8', '16']:
            if i not in self.args.pixel_format: continue
            self.byte_per_ch = int(int(i)/8)
        self.bits_per_pixel = image.bits_per_pixel

        self.byte_per_pixel = int(image.bits_per_pixel / 8)
        self.num_ch = int(image.buffer_size / (self.height * self.width * self.byte_per_ch))

        self.buffer_size = image.buffer_size
        self.dtype = 'uint' + str(self.byte_per_ch*8)
        self.device.requeue_buffer(image)

    def stop(self):
        # clean ups ---------------------------------------------------------------
        # with no arguments the function will destroy all of the
        # created devices call is optional. If this function is not called here,
        # it will be called automatically when the system module is unloading.
        system.destroy_device()
        print('---------------------->>>>>>>>>> Sensor terminated ....')

    def get_rgbd(self,**kwargs):
        image = self.device.get_buffer()
        data = np.asarray(image.data, dtype=np.uint8)
        im= np.zeros((self.height, self.width, self.num_ch), 'float')
        for ch in range(self.num_ch):
            for byte_id in range(self.byte_per_ch):
                layer = data[(np.arange(ch*self.byte_per_ch+byte_id, self.buffer_size, self.byte_per_pixel),)].reshape((self.height, self.width))
                im[:,:,ch] = im[:,:,ch] + layer * (256 ** byte_id)

        rgb, depth = None, None
        if self.num_ch == 1: # gray or depth
            if 'Mono' in self.args.pixel_format:rgb = im[:,:,0]
            else: depth=im[:,:,0]
        if self.num_ch == 3: # X,Y,depth
            depth = im[:,:,-1]
        if self.num_ch == 4: # X,Y, depth, gray
            rgb = im[:,:,-1]
            depth = im[:,:,-2]

        if rgb is not None: rgb = cv2.cvtColor(rgb.astype(self.dtype), cv2.COLOR_GRAY2BGR)
        if depth is not None: depth = depth.astype(self.dtype)

        self.device.requeue_buffer(image)
        return super().get_rgbd(rgb= rgb, depth=depth, depth_uint=self.args.depth_unit)

    def process_click_locs(self, rgbd, im):
        loc0, loc1 = self.click_locs[0], self.click_locs[1]
        # get_grip_from_rgbd(rgbd, (loc0, loc1), 9, args=self.args)
        rgbd.show(args=self.args, grips={'plates': loc0 + loc1}, title='show')
        return im



if __name__ == '__main__':
    cfg_path = 'configs/sensor/helios.cfg'
    realsense = HLOSensor(cfg_path=cfg_path).run()












