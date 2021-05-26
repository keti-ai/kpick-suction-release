import cv2
from libs.basic.basic_objects import BasImObj, BasVidWriter, RGBDWriter
from libs.basic.basic_onmouse import BasOnMouse
from libs.basic.basic_tcp_connect import ClientThread
import time
from libs.import_basic_utils import *
import copy
import open3d as o3d

class Sensor(BasImObj, BasOnMouse, ClientThread):
    def load_params(self, args):
        BasImObj.load_params(self,args=args)
        BasOnMouse.load_params(self, args=args)
        ClientThread.load_params(self, args=args)
        self.cmd_dict = ProcUtils().get_command_keys(self.args.command)
        self.cmd_dict.update({'0': 'do_nothing'})
        self.show_mouse_click = False
        self.flag_exit = False


    def vis_callback(self):
        cv2.setMouseCallback('viewer', self.on_mouse, 0)
        rgbd_disp = self.rgbd.disp(args=self.args)

        if self.rgbd.pt_in_im_range(self.mouse_loc) and self.mouse_loc != (0,0):
            dsp = '%d.%d.%d.%d' %self.rgbd.val(pt=self.mouse_loc)
            rgbd_disp = cv2.putText(rgbd_disp, dsp, self.mouse_loc, cv2.FONT_HERSHEY_COMPLEX, self.args.text_scale, self.args.text_color)
        if self.show_mouse_click:rgbd_disp = PolyUtils().draw_poly(rgbd_disp, self.click_locs)


        cv2.imshow('viewer', rgbd_disp)
        key = cv2.waitKey(self.args.mspf)
        if key == -1: key = ord(str(0))
        if key == 27:
            self.release_all()
            exit()
        return key

    def run(self):
        self.start()

        try:
            need_load_config = True
            while True:
                for kk in self.cmd_dict: print('%s: \t %s' %(kk, self.cmd_dict[kk]))

                # -------------------------->>>>>>>>>>> Retrieve RGB and depth images
                if not hasattr(self, 'get_rgbd'): exit()
                self.rgbd = self.get_rgbd()
                if self.rgbd is None: continue
                key = self.vis_callback()

                # -------------------------->>>>>>>>>>> Get command
                if chr(key) not in self.cmd_dict.keys(): continue
                command = self.cmd_dict[chr(key)]

                # -------------------------->>>>>>>>>>> Reload configs if command
                if command == 'reload_configs' or need_load_config:
                    self.reload_params()
                    need_load_config = False
                    time.sleep(0.1)
                if command == 'save_image':
                    self.save_rgbd(rgbd=self.rgbd)

                if command == 'transfer_image':
                    self.send(aDict=self.rgbd.todict())

                # video
                if command == 'start_save_video':
                    self.start_save_video()
                if hasattr(self, 'rgbdWriter'):
                    if self.write_vid: self.rgbdWriter.write(rgbd=self.rgbd)
                if command == 'stop_save_video': self.stop_save_video()

                if command == 'on_mouse':
                    self.init_locs()
                    self.run_on_mouse(rgbd=self.rgbd)

                if command == 'detect_single':
                    self.detect_single()


                if command == 'mouse_select_workspace':
                    self.init_locs()
                    self.show_mouse_click = True


                if command == 'set_workspace':
                    if len(self.click_locs)<3: continue
                    self.args.workspace = WorkSpace(pts=copy.deepcopy(self.click_locs), bound_margin=self.args.bound_margin)
                    self.show_mouse_click = False
                    command = None



                # if command == 'set_workspace':


                # -------------------------->>>>>>>>>>> Do command
                ProcUtils().clscr()
                self.do_command(command=command)

        except Exception as e:
            print('unexpected stop ...')
            self.release_all()

    def release_all(self):
        self.stop()
        if hasattr(self, 'rgbdWriter'): self.rgbdWriter.release()

    def start_save_video(self):
        self.rgbdWriter = RGBDWriter(args=self.args, hasRgb=self.rgbd.hasRgb, hasDepth=self.rgbd.hasDepth,
                                     frame_size=(self.rgbd.width,self.rgbd.height))
        for kk in self.cmd_dict:
            if self.cmd_dict[kk] != 'start_save_video': continue
            self.cmd_dict[kk] = 'stop_save_video'
        self.write_vid = True

    def stop_save_video(self):
        self.rgbdWriter.release()
        for kk in self.cmd_dict:
            if self.cmd_dict[kk] != 'stop_save_video': continue
            self.cmd_dict[kk] = 'start_save_video'
        self.write_vid = False

    def detect_single(self):
        print('detecting single frame ...')

    def start(self):
        print('initiating sensor ...')

    def stop(self):
        print('terminating sensor ...')

    def do_command(self, command):
        print('doing command ...')








if __name__ == '__main__':
    cfg_path = 'configs/DDI/grip_kinect_azure_hcr.cfg'
    realsense = Sensor(cfg_path).run()
