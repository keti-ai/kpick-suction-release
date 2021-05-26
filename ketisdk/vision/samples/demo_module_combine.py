
from  ketisdk.vision.base.base_objects import DetGuiObj
import cv2
from scipy.ndimage import rotate
from ketisdk.gui.gui import GUI, GuiModule, GuiProcess
from ketisdk.vision.sensor.realsense_sensor import get_realsense_modules

def demo_module_combine():
    class ImRot(DetGuiObj):
        def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb'):
            if method_ind==0:
                # out = rotate(rgbd.rgb,angle=45)
                rotation_matrix = cv2.getRotationMatrix2D((rgbd.width / 2, rgbd.height / 2), 45, 1)
                out = cv2.warpAffine(rgbd.rgb, rotation_matrix, (rgbd.width, rgbd.height))
                # out = rotate(rgbd.rgb, angle=45, reshape=False, order=3)
                ret = {'im': out}
            return ret

    class ImScale(DetGuiObj):
        def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb'):
            if method_ind==0:
                out = cv2.resize(rgbd.rgb, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
                ret = {'im': out}
            return ret

    class ImGray(DetGuiObj):
        def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb'):
            if method_ind==0:
                out = cv2.cvtColor(rgbd.rgb, cv2.COLOR_RGB2GRAY)
                ret = {'im': out}
            return ret

    # Define modules
    rot_module = GuiModule(ImRot, name='Im Rotator', short_name='Rot')
    scale_module = GuiModule(ImScale, name='Im Scale', short_name='Scl')
    gray_module = GuiModule(ImGray, name='Im Grayer', short_name='Gry')

    # # define input sources
    # scale_module.input_sources=[rot_module.short_name, gray_module.short_name]
    # gray_module.input_sources=[scale_module.short_name, rot_module.short_name]

    # define processes
    # process0 = GuiProcess(gui_modules=[rot_module, scale_module],name='rot->scale', ask_next=False)
    # process1 = GuiProcess(gui_modules=[rot_module, scale_module, gray_module],name='rot->scale->gray')

    GUI(title='Image Process Moudle Combine', modules=[rot_module,scale_module, gray_module]+get_realsense_modules(),
        # processes=[process0,process1]
        )

if __name__=='__main__':
    demo_module_combine()
