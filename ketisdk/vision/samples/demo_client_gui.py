from ketisdk.base.base_tcp import ClientThread
import os,sys
# sys.path.append(os.getcwd())

from ketisdk.vision.base.base_objects import DetGuiObj
from ketisdk.vision.sensor.realsense_sensor import get_realsense_modules
from ketisdk.gui.gui import GUI, GuiModule
from ketisdk.utils.proc_utils import CFG

def demo_client_gui(host='localhost',port=8888):
    class ClientGui(DetGuiObj, ClientThread):
        def __init__(self, args=None, cfg_path=None):
            DetGuiObj.__init__(self, args=args, cfg_path=cfg_path)
            ClientThread.__init__(self, host=self.args.host, port=self.args.port)

        def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb'):
            return  self.send_and_get_return({'rgb': rgbd.rgb, 'depth': rgbd.depth,
                       'bbox': rgbd.workspace.bbox, 'ws_pts': rgbd.workspace.pts, 'disp_mode': disp_mode})['im']


    args = CFG()
    args.host, args.port = host, port
    client_module = GuiModule(ClientGui, type='client_gui', name='Client GUI', category='detector',
                              run_thread=True, args=args)
    GUI(title='Client Gui', modules=[client_module,]+get_realsense_modules())

if __name__=='__main__':
    demo_client_gui()
