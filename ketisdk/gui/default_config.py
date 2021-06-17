from ..utils.proc_utils import CFG

def default_args():
    args_ = CFG()
    args_.flag  =CFG()
    args_.flag.save_detect = True
    args_.flag.show_detect = False
    args_.flag.do_predict = True
    args_.flag.wait_key = True
    args_.flag.show_steps = False
    args_.flag.train_mode = False


    args_.sensor  =CFG()
    args_.sensor.depth_scale_params = [400, 600, 100, 60000]
    args_.sensor.bound_margin=20
    args_.sensor.use_rgb=True
    args_.sensor.use_depth=True
    args_.sensor.depth_min = args_.sensor.depth_scale_params[0]
    args_.sensor.depth_max = args_.sensor.depth_scale_params[1]
    args_.sensor.crop_poly = None
    args_.sensor.crop_rect=None
    args_.sensor.ws_depth = 300
    args_.sensor.depth_size = (480, 848)
    args_.sensor.rgb_size = (720, 1280)
    # args_.sensor.mspf = 5
    args_.sensor.vid_fps = 15.0

    args_.path  =CFG()
    args_.path.root_dir=''
    args_.path.im_suffixes='image/*'
    args_.path.depth_suffixes='depth/*'

    args_.disp  =CFG()
    args_.disp.window_size = (800, 1200)
    args_.disp.show_depth_jet=True
    args_.disp.show_rgb_main = True
    args_.disp.text_info_org = (50, 50)
    args_.disp.text_scale = 0.8
    args_.disp.text_thick = 2
    args_.disp.text_color=(0,0,255)
    args_.disp.text_rect =(0,0,100,100)
    args_.disp.text_alpha=0.5
    args_.disp.up2down=True
    args_.disp.text_space=30
    args_.disp.line_thick = 2
    args_.disp.line_color = (0,255,0)
    args_.disp.marker_size = 10
    args_.disp.marker_thick=2
    args_.disp.marker_color =( 0,255,0)
    args_.disp.bg_color=(0,100,0)
    args_.disp.bg_depth_diff_thres=5
    args_.disp.roi_disp_size = (150, 150)
    args_.disp.pad_size = (50, 50)

    return args_
