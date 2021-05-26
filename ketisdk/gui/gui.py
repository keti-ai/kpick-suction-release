import os, sys

sys.path.append(os.getcwd())

import tkinter as tk
from tkinter import filedialog as fd
from tkinter import ttk
from tkinter import messagebox as msgbox
from PIL import Image, ImageTk

import cv2
from ..import_basic_utils import *
import numpy as np
import time

from threading import Thread
from glob import glob
from functools import partial
from ..utils.proc_utils import CFG
from .default_config import default_args

commands = ['show_annotation', 'switch_imshow', 'show_roi', '90_rot', 'crop_roi', 'measure']


class GUI():
    def __init__(self, title='unnamed', gui_size=(1200, 800), im_disp_scale=(1000, 600),
                 default_cfg_path=None, modules=[], test_dirs=[], run_defaults=None, processes=[]):
        self.test_dirs = test_dirs
        self.gui_size = gui_size
        self.modules = modules
        self.processes = processes
        self.im_disp_scale = im_disp_scale
        self.default_cfg_path = default_cfg_path
        if self.default_cfg_path is None: self.default_cfg_path = 'configs/default.cfg'
        if not ProcUtils().is_exist(self.default_cfg_path):
            self.args = default_args()
        else:
            self.args = CFG(cfg_path=self.default_cfg_path, separate=True)
            self.args.merge_with(default_args())

        # self.args = self.args_.flatten()
        self.key_commands = ProcUtils().get_command_keys(commands)
        self.init_params()

        self.window = tk.Tk()
        self.window.geometry('{}x{}'.format(gui_size[0], gui_size[1]))
        self.window.title(title)

        self.source_selected = tk.StringVar()
        self.show_steps_var = tk.BooleanVar()
        self.flag_run_all_var = tk.BooleanVar()
        self.flag_run_acc_var = tk.BooleanVar()
        self.flag_hold_var = tk.BooleanVar()
        self.flag_loop_var = tk.BooleanVar()
        # self.str_num_run = tk.StringVar()
        self.str_range_min = tk.StringVar()
        self.str_range_max = tk.StringVar()

        self.ann_format_var = tk.StringVar()
        self.ann_format_var.set('CT')
        self.flag_show_mask_var = tk.BooleanVar()
        self.flag_show_mask_var.set(True)
        self.flag_show_box_var = tk.BooleanVar()
        self.flag_show_add_im_var = tk.BooleanVar()
        self.flag_show_add_im_var.set(True)
        self.flag_show_add_sym_var = tk.BooleanVar()
        self.flag_show_add_key_var = tk.BooleanVar()
        self.flag_show_add_graph_var = tk.BooleanVar()

        #
        self.fr_task_cmd = tk.Frame(master=self.window)
        # self.fr_task_cmd.pack(fill=tk.BOTH, expand=False, side=tk.LEFT)
        self.fr_config = tk.Frame(master=self.window, borderwidth=5, relief=tk.RAISED)
        # self.fr_config.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.viewer_parent = ttk.Notebook(master=self.fr_config)

        self.layout_module()

        if run_defaults is not None:
            dir_ind = run_defaults['dir_ind'] if 'dir_ind' in run_defaults else None
            if dir_ind is not None:
                root_dir, im_dir = os.path.split(self.test_dirs[dir_ind])
                self.open_dir_(root_dir, os.path.join(im_dir, '*'))
            run_behaves = run_defaults['run_behaves'] if 'run_behaves' in run_defaults else []
            if 'all' in run_behaves: self.flag_run_all_var.set(True)
            if 'acc' in run_behaves: self.flag_run_acc_var.set(True)
            if 'show_steps' in run_behaves: self.show_steps_var.set(True)
            if 'loop' in run_behaves: self.flag_loop_var.set(True)

            im_range = run_defaults['im_range'] if 'im_range' in run_defaults else ('','')
            if len(im_range)==1: range_min, range_max = im_range[0], ''
            if len(im_range)==2: range_min, range_max = im_range
            self.str_range_min.set(str(range_min))
            self.str_range_max.set(str(range_max))

        self.viewer_parent.pack(fill=tk.BOTH)
        self.fr_task_cmd.pack(fill=tk.BOTH, expand=False, side=tk.LEFT)
        self.fr_config.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.window.bind("<Key>", self.do_key_press)
        self.window.bind('<Up>', self.do_upKey)
        self.window.bind('<Down>', self.do_downKey)
        self.window.bind("<Button-4>", self.do_mouseRollUp)
        self.window.bind("<Button-5>", self.do_mouseRollDown)

        self.window.mainloop()
        # control_thread.join()

    def set_vision_viewer(self, tab_caption, viewer_parent=None):
        if viewer_parent is None: viewer_parent = self.viewer_parent
        imshow_tab = ttk.Frame(master=viewer_parent)  # add viewer to sensor
        fr_imshow = tk.Frame(master=imshow_tab)
        viewer_parent.add(imshow_tab, text=tab_caption)
        fr_imshow.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, anchor=tk.NW)
        viewer = tk.Label(fr_imshow)
        # viewer.place(x=0, y=0)
        viewer.bind("<Button 1>", self.do_mouse_click)
        viewer.bind("<Motion>", self.do_mouse_move)
        viewer.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, anchor=tk.NW)
        return viewer

    def set_textbox_viewer(self, name='unnamed_configs', viewer_parent=None):
        if viewer_parent is None: viewer_parent = self.viewer_parent
        conf_tab = ttk.Frame(master=viewer_parent)
        viewer_parent.add(conf_tab, text=name)

        conf_viewer = tk.Text(master=conf_tab, width=self.tab_width,
                              height=self.tab_height)  # , font=("Courier", 14))  # , height=10)
        # conf_viewer.grid(row=0, column=0)
        conf_viewer.pack(fill=tk.BOTH)

        return conf_viewer

    def layout_module(self):

        self.btn_w1, self.btn_w2, self.btn_w2_lb, self.btn_w3, self.btn_w4, self.btn_w5 = 17, 7, 5, 3, 2, 1

        hgl_frame = tk.Frame(master=self.fr_task_cmd)
        self.highlight_label = tk.Label(master=hgl_frame, text='   ',height=2)
        self.highlight_label.grid(row=0, column=0)
        hgl_frame.pack()

        btn_frame = tk.Frame(master=self.fr_task_cmd)
        tk.Button(master=btn_frame, text='img', command=self.open_image, width=self.btn_w5).grid(row=0, column=0)
        tk.Button(master=btn_frame, text='dir', command=self.open_dir, width=self.btn_w5).grid(row=0, column=1)
        tk.Button(master=btn_frame, text='root>', command=self.open_rgbd_dir, width=self.btn_w5).grid(row=0, column=2)
        tk.Button(master=btn_frame, text='cfg>', command=self.open_dir_from_args, width=self.btn_w5).grid(row=0,
                                                                                                          column=3)
        tk.Button(master=btn_frame, text='pdf>', command=self.open_image_from_pdf, width=self.btn_w5).grid(row=0,
                                                                                                           column=4)
        for j, test_dir in enumerate(self.test_dirs):
            root_dir, im_dir = os.path.split(test_dir)
            tk.Button(master=btn_frame, text='dir%d'%j,
                      command=partial(self.open_dir_,root_dir,os.path.join(im_dir,'*')),  width=self.btn_w5).\
                grid(row=1+(j//5), column=j-(j//5)*5)
        btn_frame.pack()

        self.viewers, viewer_ind = [], 0
        for j, module in enumerate(self.modules):
            if 'sensor' not in module.category: continue
            # tk.Label(master=self.fr_task_cmd, text='').pack(fill=tk.X)
            self.add_sensor(sensor=module)
            module.viewer_ind = viewer_ind
            viewer = self.set_vision_viewer(tab_caption=module.name)
            viewer_ind += 1
            self.viewers.append(viewer)

        self.viewers.append(self.set_vision_viewer(tab_caption='viewer'))

        # tk.Label(master=self.fr_task_cmd, text='').pack(fill=tk.X)
        fr_chk = tk.Frame(master=self.fr_task_cmd, highlightbackground='green', highlightcolor='green', highlightthickness=2)
        tk.Checkbutton(fr_chk, variable=self.flag_hold_var, text='Hold', width=self.btn_w3).grid(row=0, column=0)
        tk.Checkbutton(fr_chk, variable=self.flag_run_all_var, text='All', width=self.btn_w3).grid(row=0, column=1)
        tk.Checkbutton(fr_chk, variable=self.flag_run_acc_var, text='Acc', width=self.btn_w3).grid(row=0, column=2)
        tk.Checkbutton(fr_chk, variable=self.show_steps_var, text='Step', width=self.btn_w3).grid(row=1, column=0)
        tk.Checkbutton(fr_chk, variable=self.flag_loop_var, text='Loop', width=self.btn_w3).grid(row=1, column=1)
        # tk.ttk.Entry(fr_chk, textvariable=self.str_num_run, width=self.btn_w3).grid(row=0, column=3)
        tk.Label(fr_chk, text='Range', width=self.btn_w2_lb).grid(row=2, column=0)
        tk.ttk.Entry(fr_chk, textvariable=self.str_range_min, width=self.btn_w2_lb).grid(row=2, column=1)
        # tk.Label(fr_chk, text='-', width=self.btn_w5).grid(row=1, column=2)
        tk.ttk.Entry(fr_chk, textvariable=self.str_range_max, width=self.btn_w2_lb).grid(row=2, column=2)
        fr_chk.pack()

        for j, module in enumerate(self.modules):
            if 'sensor' in module.category: continue
            self.add_module(module_ind=j)

        tk.Label(master=self.fr_task_cmd, text=f'=====PROCESS=====').pack()
        prc_frame = tk.Frame(master=self.fr_task_cmd)
        self.str_process_module_inds = tk.StringVar()
        txt = f'{[j for j, mod in enumerate(self.modules) if "sensor" not in mod.category]}'[1:-1]
        self.str_process_module_inds.set(f'{txt}')
        tk.ttk.Entry(master=prc_frame, textvariable=self.str_process_module_inds,width=self.btn_w2).grid(row=0, column=0)
        self.process_ask_next_var=tk.BooleanVar()
        tk.Checkbutton(prc_frame, variable=self.process_ask_next_var, text='Ask_next', width=self.btn_w2).grid(row=0, column=1)
        tk.Button(master=prc_frame, text='Run', command=self.run_process,width=self.btn_w3).grid(row=0,column=2)
        prc_frame.pack()
        tk.Label(master=self.fr_task_cmd, text=f'{"="*16}').pack()



        btn_frame = tk.Frame(master=self.fr_task_cmd)
        j = 0
        tk.Button(master=btn_frame, text='Load cfg', width=self.btn_w2, command=self.reload_configs). \
            grid(row=j // 2, column=j - 2 * (j // 2))
        j += 1
        self.btn_select_workspace = tk.Button(btn_frame, text='Select ws', width=self.btn_w2,
                                              command=self.mouse_select_workspace)
        self.btn_select_workspace.grid(row=j // 2, column=j - 2 * (j // 2))
        j += 1
        tk.Button(btn_frame, text='Unset ws', width=self.btn_w2, command=self.unset_workspace). \
            grid(row=j // 2, column=j - 2 * (j // 2))
        j += 1
        tk.Button(master=btn_frame, text='Save im', width=self.btn_w2, command=self.save_image). \
            grid(row=j // 2, column=j - 2 * (j // 2))
        j += 1
        self.btn_save_video = tk.Button(master=btn_frame, text='Save vid', width=self.btn_w2, command=self.save_video)
        self.btn_save_video.grid(row=j // 2, column=j - 2 * (j // 2))
        j += 1
        tk.Button(master=btn_frame, text='Make Vid', width=self.btn_w2, command=self.create_video). \
            grid(row=j // 2, column=j - 2 * (j // 2))
        btn_frame.pack()

        # tk.Label(master=self.fr_task_cmd).pack()
        key_frame = tk.Frame(self.fr_task_cmd)
        self.key_command_viewer = tk.Label(master=key_frame, highlightbackground='green', highlightcolor='green', highlightthickness=2)
        self.key_command_viewer.pack(fill=tk.X)
        self.update_key_command_viewer()
        key_frame.pack()

        ################################################## EROOOOOOOOOOOOOOOOOOOORRRR
        self.tab_height, self.tab_width = 80, 200
        self.default_conf_viewer = self.set_textbox_viewer('default_configs', viewer_parent=self.viewer_parent)

        # conf_tab = ttk.Frame(master=self.viewer_parent)
        # self.default_conf_viewer = tk.Text(master=conf_tab, width=self.tab_width,
        #                                    height=self.tab_height)  # , font=("Courier", 14))  # , height=10)
        # self.default_conf_viewer.grid(row=0, column=0)
        # self.viewer_parent.add(conf_tab, text='default_configs')

        GuiUtils().update_textbox(self.default_conf_viewer, self.args.to_string())

    def get_module_args(self, module):
        # if module.args_ is not None:
        #     if module.args is None: module.args =  module.args_.flatten()   # args not None means there are pre-defined configs
        if module.args is not None: args = module.args.copy()
        else: args = CFG()
        for section_name in self.args.keys():
            default_section = getattr(self.args, section_name)
            if hasattr(args, section_name):
                section = getattr(args, section_name)
                for option_name in default_section.keys():
                    setattr(section, option_name, getattr(default_section, option_name))
            else:
                setattr(args, section_name, default_section)
        # [setattr(args_, key, getattr(self.args_, key)) for key in self.args_.keys()]
        # args = args_.flatten()
        if module.args is None:  module.args = args
        else: module.args.merge_with(args)


    def update_all_modules_args(self):
        if self.modules is None: return
        if len(self.modules) == 0: return
        for module in self.modules:
            if not hasattr(module, 'args'): module.args = CFG(cfg_path=module.cfg_path, separate=True)
            self.get_module_args(module)
            if hasattr(module, 'worker'):
                module.worker.args = module.args
                if hasattr(module.worker, 'reload_params'): module.worker.reload_params()

    def update_after_change_args_(self):
        # self.args = self.args_.flatten()
        self.update_all_modules_args()
        self.set_workspace()
        GuiUtils().update_textbox(self.default_conf_viewer, self.args.to_string())

    def init_params(self):
        modules = [module for module in self.modules if 'sensor' in module.category]  # arrange sensor first
        self.num_sensor = len(modules)

        self.update_all_modules_args()
        self.set_workspace()

        self.mouse_loc = (0, 0)
        self.click_locs = []

        self.detected = None
        self.detected_disp = None
        self.rgbds = [None] * (self.num_sensor + 1)
        self.num_viewer = self.num_sensor + 1
        self.view_scales = [(1, 1)] * self.num_viewer

        # flags
        self.flag_select_workspace = False
        self.flag_show_click_locs = False
        self.flag_show_roi = False
        self.flag_imshow_modes = ['rgb', 'depth', 'depth_norm', 'depth_jet']
        self.flag_show_roi = False
        self.flag_show_detected = False
        self.flag_90_rot = False
        self.flag_show_annotation = False
        self.flag_measure = False

    def open_image(self):
        control_thread = Thread(target=self.open_image_, daemon=True)
        control_thread.start()

    def open_image_(self):
        try:
            filepath = fd.askopenfilename(initialdir='data')
            if ProcUtils().isimpath(filepath):
                self.rgbd_path = [filepath, ]
                # self.rgb = cv2.imread(filepath)[:,:,::-1]
                self.rgbds[-1] = self.get_rgbd_from_path(use_rgb=True, use_depth=False)
                self.update_im_viewer()
                self.viewer_parent.select(self.num_viewer - 1)

            else:
                print('%s is not an image ...' % filepath)
        except:
            pass

    def open_image_from_pdf(self):
        from pdf2image import convert_from_path as read_pdf
        try:
            filepath = fd.askopenfilename(initialdir='data', title='Select a pdf file...',
                                          filetypes=(("pdf files", "*.pdf"), ("all files", "*.*")))
            self.data_root, filename = os.path.split(filepath)
            filename = filename.replace('.pdf', '')
            im_dir = os.path.join(self.data_root, 'image')
            ProcUtils().rmdir(im_dir)
            images = read_pdf(filepath, dpi=300)

            for j, im in enumerate(images):
                im_ind = str(j).rjust(3, '0')
                ArrayUtils().save_array_v3(np.array(im), os.path.join(im_dir, '%s_%s.png' % (filename, im_ind)))

            self.args.path.im_suffixes = ['image/*']
            if hasattr(self.args.path, 'depth_suffixes'): delattr(self.args.path, 'depth_suffixes')
            # self.update_after_change_args_()
            self.open_dir_(data_root=self.data_root)
        except:
            pass

    def open_dir_from_args(self, module=None):
        # if module is not None:
            # self.copy_args_sections_from_module(module_args_=module.args_, section_names=['path'])
            # self.update_after_change_args_()
        if module is not None:
            if hasattr(module.args_,'path'):
                self.copy_args_sections_from_module(module_args_=module.args_, section_names=['path'])
        # self.open_dir_(data_root=self.args_.path.root_dir)
        control_thread = Thread(target=self.open_dir_, args=(self.args.path.root_dir,), daemon=True)
        control_thread.start()

    def open_dir(self):
        self.args.path.im_suffixes = '*'
        if hasattr(self.args.path, 'depth_suffixes'): delattr(self.args.path, 'depth_suffixes')
        # self.update_after_change_args_()
        control_thread = Thread(target=self.open_dir_, daemon=True)
        control_thread.start()

    def open_rgbd_dir(self):
        self.args.path.im_suffixes = 'image/*'
        self.args.path.depth_suffixes = 'depth/*'
        # self.update_after_change_args_()
        control_thread = Thread(target=self.open_dir_, daemon=True)
        control_thread.start()

    def open_dir_(self, data_root=None, im_suffixes=None, depth_suffixes=None):
        if hasattr(self, 'viewer_parent'): self.viewer_parent.select(self.num_viewer - 1)
        try:
            if data_root is not None:
                self.data_root = data_root
            else:
                self.data_root = fd.askdirectory(initialdir='data', title='Select root directory')
            if im_suffixes is not None: self.args.path.im_suffixes = im_suffixes
            if depth_suffixes is not None: self.args.path.depth_suffixes = depth_suffixes

            self.args.path.root_dir = self.data_root
            self.update_after_change_args_()

            print('Root %s selected ...' % self.data_root)
            self.get_im_list()
            self.num_im = len(self.paths)
            if self.num_im == 0:
                print('No image in %s ...' % self.data_root)
                return

            print('%d images ...' % self.num_im)

            if not hasattr(self, 'im_ind'): self.im_ind = 0
            self.rgbd_path = self.paths[self.im_ind]
            self.rgbds[-1] = self.get_rgbd_from_path()
            self.update_im_viewer()

        except:
            pass

    def next_im(self, forward=True):
        try:
            self.flag_show_detected = False
            self.highlight_label.configure(text='   ')
            tab_ind = self.get_current_tab_ind()
            if tab_ind >= self.num_viewer: return
            if tab_ind < self.num_viewer - 1:
                self.tab2sensor(tab_ind).flag_get_data = True
                return

            if not hasattr(self, 'paths'): return
            # if self.im_ind==(self.num_im-1): return

            if forward: self.im_ind += 1
            else: self.im_ind -= 1
            self.im_ind = min(max(0, self.im_ind), self.num_im - 1)

            self.rgbd_path = self.paths[self.im_ind]
            print('[%d/%d] %s' % (self.im_ind, self.num_im, self.rgbd_path['base']))

            self.rgbds[-1] = self.get_rgbd_from_path()

            if self.hold_detect():
                # self.run_detect_dir_(self.just_executed_module, self.just_executed_method_ind)
                if 'module' in self.just_executed:
                    self.run_detect_(module=self.just_executed['module'],
                                     method_ind=self.just_executed['method_ind'])
                if 'process' in self.just_executed: self.run_process_()
                self.flag_show_detected = True
            else:self.flag_show_detected = False

            self.update_im_viewer()


        except:
            pass

    def hold_detect(self):
        if not self.flag_hold_var.get(): return False
        if not hasattr(self,'just_executed'): return False
        to_hold = False
        if 'module' in self.just_executed:
            to_hold = self.just_executed['module'].category=='detector'
        if 'process' in self.just_executed:
            to_hold = self.modules[self.just_executed['process'][-1]].category=='detector'
        return to_hold

    def mouse_select_workspace(self):
        if self.hold_detect(): return 
        self.click_locs = []
        self.flag_show_click_locs = True
        self.flag_measure = True
        self.btn_select_workspace.configure(text='Set ws', command=self.set_mouse_workspace)

    def set_mouse_workspace(self):
        self.flag_show_click_locs = False
        self.flag_measure = False
        self.btn_select_workspace.configure(text='Select ws', command=self.mouse_select_workspace)
        if len(self.click_locs) < 3: return
        self.args.sensor.crop_poly = self.click_locs
        self.update_after_change_args_()

        if self.rgbds[-1] is not None:
            self.rgbds[-1].workspace = self.workspace
            self.update_im_viewer()

    def unset_workspace(self):
        self.args.sensor.crop_poly, self.args.sensor.crop_rect = None, None
        self.update_after_change_args_()
        if self.rgbds[-1] is not None:
            self.rgbds[-1].workspace = self.workspace
            self.update_im_viewer()

    def update_im_viewer(self, im=None, viewer_ind=-1):
        viewer = self.viewers[viewer_ind]
        # im_on_image = self.im_on_viewers[viewer_ind]
        if im is None:
            im = self.rgbd_disp(rgbd=self.rgbds[viewer_ind])
        h, w = im.shape[:2]
        wo, ho = self.im_disp_scale
        wo, ho = ArrayUtils().fix_resize_scale(in_size=(w, h), out_size=(wo, ho))
        # self.im_disp_scale = (wo, ho)
        self.view_scales[viewer_ind] = (wo / w, ho / h)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.resize(im, (wo, ho))))
        viewer.configure(image=img, text=self.viewer_info, compound=tk.TOP, font=("Courier", 18), anchor=tk.NW)
        viewer.image = img
        # viewer.itemconfigure(im_on_image, image=img)

    def reload_configs(self):
        self.args = CFG().from_string(GuiUtils().textbox2text(self.default_conf_viewer), separate=True)
        self.args.write(path=self.default_cfg_path)
        self.update_after_change_args_()

    def draw_roi(self, im, pt=None):
        if pt is None: pt = self.mouse_loc
        height, width = im.shape[:2]
        out = np.copy(im)
        if ArrayUtils().pt_in_im_range(pt, height=height, width=width):
            roi = ArrayUtils().crop_array_patch(out, center=pt, pad_size=self.args.disp.pad_size)
            roi = cv2.resize(roi, self.args.disp.roi_disp_size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            h, w = roi.shape[:2]
            cv2.drawMarker(roi, (w // 2, h // 2), self.args.disp.marker_color, cv2.MARKER_CROSS,
                           self.args.disp.marker_size, self.args.disp.marker_thick)

            xm, ym = pt
            ss = 50

            y_end = ym + ss + h
            if y_end >= height: y_end = ym - ss
            x_end = xm + ss + w
            if x_end >= width: x_end = xm - ss
            out[y_end - h:y_end, x_end - w:x_end, :] = roi
        return out

    def do_show_detect(self):
        return self.flag_show_detected and self.detected is not None

    def rgbd_disp(self, rgbd=None):
        # timer = Timer()

        show_detect = self.do_show_detect()
        if show_detect:
            rgbd_disp = self.detected_disp
        else:
            if rgbd is None: return np.zeros((720, 1080), 'uint8')
            rgbd_disp = rgbd.disp(mode=self.flag_imshow_modes[0])

        if rgbd_disp is None: return np.zeros((720, 1080), 'uint8')
        if self.flag_show_roi: rgbd_disp = self.draw_roi(rgbd_disp)
        disp_height, disp_width = rgbd_disp.shape[:2]
        if not show_detect:
            self.viewer_info =f'Size: {rgbd.width}x{rgbd.height}, Mode: {self.flag_imshow_modes[0]}\n'
            if rgbd.pt_in_im_range(self.mouse_loc): self.viewer_info += str(self.mouse_loc) + rgbd.val_str(self.mouse_loc) + '\n'
        else:
            self.viewer_info = f'Size: {disp_width}x{disp_height}, Mode: {self.flag_imshow_modes[0]}\n'
        if ArrayUtils().pt_in_im_range(self.mouse_loc, height=disp_height, width=disp_width):
            self.viewer_info += f'{rgbd_disp[self.mouse_loc[::-1]]}'


        if self.flag_show_click_locs: rgbd_disp = PolyUtils().draw_poly(rgbd_disp, self.click_locs)

        if self.flag_measure:
            xm, ym = self.mouse_loc
            ss = self.args.disp.line_thick
            x0, y0 = ArrayUtils().truncate(xm - ss // 2, 0, rgbd.width), ArrayUtils().truncate(ym - ss // 2, 0,
                                                                                               rgbd.height)
            x1, y1 = ArrayUtils().truncate(x0 + ss, 0, rgbd.width), ArrayUtils().truncate(y0 + ss, 0, rgbd.height)
            rgbd_disp[:, x0:x1, :] = self.args.disp.line_color
            rgbd_disp[y0:y1, :, :] = self.args.disp.line_color

            if len(self.click_locs) > 0:
                x0, y0 = self.click_locs[-1]
                x1, y1 = self.mouse_loc
                xc, yc = (x0 + x1) // 2, (y0 + y1) // 2
                dx, dy = abs(x1 - x0), abs(y1 - y0)
                dd = np.linalg.norm((dx, dy))

                cv2.line(rgbd_disp, (x0, y0), (x1, y1), self.args.disp.line_color, self.args.disp.line_thick)
                cv2.putText(rgbd_disp, 'dx:%d-dy:%d-l:%.1f' % (dx, dy, dd), (xc, yc), cv2.FONT_HERSHEY_COMPLEX,
                            self.args.disp.text_scale, self.args.disp.text_color, self.args.disp.text_thick)

        # if self.flag_90_rot: rgbd_disp = np.rot90(rgbd_disp, -1)
        # timer.pin_time('option_disp')
        # print(timer.pin_times_str())
        return rgbd_disp

    def do_mouse_move(self, event):
        try:
            tab_ind = self.get_current_tab_ind()
            if tab_ind >= self.num_viewer: return
            x0, y0 = event.x, event.y

            rgbd = self.rgbds[tab_ind]
            if rgbd is None: return

            fx, fy = self.view_scales[tab_ind]
            x0, y0 = int(x0 / fx), int(y0 / fy)
            self.mouse_loc = (x0, y0)

            if tab_ind == (self.num_viewer - 1):
                self.update_im_viewer()
        except:
            pass

    def do_mouse_click(self, event):
        # mouse_loc = self.correct_mouse_loc((event.x, event.y))
        self.click_locs.append(self.mouse_loc)

    def do_mouseRollUp(self, event):
        if hasattr(self, 'flag_rollUp'):
            self.flag_rollUp = True
        else:
            self.__setattr__('flag_rollUp', True)
        self.do_mouseRoll_()

    def do_mouseRollDown(self, event):
        if hasattr(self, 'flag_rollUp'):
            self.flag_rollUp = False
        else:
            self.__setattr__('flag_rollUp', False)
        self.do_mouseRoll_()
    def do_mouseRollDown(self, event):
        if hasattr(self, 'flag_rollUp'):
            self.flag_rollUp = False
        else:
            self.__setattr__('flag_rollUp', False)
        self.do_mouseRoll_()

    def do_mouseRoll_(self):
        if self.flag_rollUp:
            rat = 1.05
        else:
            rat = 1 / 1.05

        self.im_disp_scale = (int(self.im_disp_scale[0] * rat), int(self.im_disp_scale[1] * rat))
        if self.get_current_tab_ind() != self.num_viewer - 1: return
        if self.rgbds[-1] is None: return
        self.update_im_viewer()

    def do_key_press(self, event):
        tab_ind = self.get_current_tab_ind()
        if tab_ind >= self.num_viewer: return
        self.pressed_key = event.char
        if self.pressed_key not in self.key_commands: return 1

        key_command = self.key_commands[self.pressed_key]

        if key_command == 'show_annotation':
            self.flag_show_annotation = True
            ProcUtils().change_cmd_dict(self.key_commands, prev='show_annotation', lat='stop_show_annotation')
            self.update_key_command_viewer()
            rgbd = self.rgbds[-1]
            if rgbd is None: return
            self.update_im_viewer(im=self.rgbd_disp(rgbd)[:, :, ::-1])

        if key_command == 'stop_show_annotation':
            self.flag_show_annotation = False
            ProcUtils().change_cmd_dict(self.key_commands, prev='stop_show_annotation', lat='show_annotation')
            self.update_key_command_viewer()
            rgbd = self.rgbds[-1]
            if rgbd is None: return
            self.update_im_viewer(im=self.rgbd_disp(rgbd))

        if key_command == 'switch_imshow':
            self.flag_imshow_modes = self.flag_imshow_modes[1:] + [self.flag_imshow_modes[0]]

            rgbd = self.rgbds[-1]
            if rgbd is None: return
            self.update_im_viewer(im=self.rgbd_disp(rgbd))

        if key_command == 'show_roi':
            self.flag_show_roi = True
            ProcUtils().change_cmd_dict(self.key_commands, prev='show_roi', lat='stop_show_roi')
            self.update_key_command_viewer()

        if key_command == 'stop_show_roi':
            self.flag_show_roi = False
            ProcUtils().change_cmd_dict(self.key_commands, prev='stop_show_roi', lat='show_roi')
            self.update_key_command_viewer()

        if key_command == '90_rot':
            self.flag_90_rot = True
            ProcUtils().change_cmd_dict(self.key_commands, prev='90_rot', lat='stop_90_rot')
            self.update_key_command_viewer()

        if key_command == 'stop_90_rot':
            self.flag_90_rot = False
            ProcUtils().change_cmd_dict(self.key_commands, prev='stop_90_rot', lat='90_rot')
            self.update_key_command_viewer()

        if key_command == 'crop_roi':
            self.mouse_crop_roi()
            ProcUtils().change_cmd_dict(self.key_commands, prev='crop_roi', lat='save_roi')

            self.update_key_command_viewer()
        if key_command == 'save_roi':
            self.mouse_save_roi(tab_ind)
            ProcUtils().change_cmd_dict(self.key_commands, prev='save_roi', lat='crop_roi')
            self.update_key_command_viewer()

        if key_command == 'measure':
            self.click_locs = []
            self.flag_measure = True
            ProcUtils().change_cmd_dict(self.key_commands, prev='measure', lat='stop_measure')
            self.update_key_command_viewer()

        if key_command == 'stop_measure':
            self.flag_measure = False
            ProcUtils().change_cmd_dict(self.key_commands, prev='stop_measure', lat='measure')
            self.update_key_command_viewer()

    def mouse_crop_roi(self):
        self.click_locs = []
        self.flag_show_click_locs = True
        self.flag_measure = True

    def mouse_save_roi(self, tab_ind):
        if len(self.click_locs)>2:
            ws = WorkSpace(pts=self.click_locs)
            left, top, right, bottom = ws.bbox
            rgbd_crop = self.rgbds[tab_ind].crop(left=left, right=right, top=top, bottom=bottom)
            filename = ProcUtils().get_current_time_str()
            ArrayUtils().save_array_v3(rgbd_crop.bgr(), os.path.join(self.args.path.root_dir, 'gui_crop', filename + '_rgb.png'))
            ArrayUtils().save_array_v3(rgbd_crop.depth,
                                       os.path.join(self.args.path.root_dir, 'gui_crop', filename + '_depth.png'), open_dir=True)

        self.click_locs = []
        self.flag_show_click_locs = False
        self.flag_measure = False

    def do_upKey(self, event):
        self.next_im(forward=False)

    def do_downKey(self, event):
        self.next_im(forward=True)

    def update_key_command_viewer(self):
        cmd_text = ''
        for key_str in self.key_commands:
            cmd_text += '%s: %s\n' % (key_str, self.key_commands[key_str])
        self.key_command_viewer.configure(text=cmd_text)

    def get_current_tab_ind(self):
        return self.viewer_parent.index(self.viewer_parent.select())

    def get_im_list(self):
        """ get image path lists """
        self.paths = []

        self.use_rgb = hasattr(self.args.path, 'im_suffixes')
        self.use_depth = hasattr(self.args.path, 'depth_suffixes')

        if not self.use_rgb and not self.use_depth:
            print(f'{"+"*10} Please check im_suffixes and depth_suffixes')
            return

        data_types = [k.replace('_suffixes','') for k in self.args.path.keys() if k.endswith('_suffixes')]
        data_suffixes = [self.args.path.__getattribute__(f'{p}_suffixes') for p in data_types ]
        data_dict = dict()
        for t,s in zip(data_types, data_suffixes) : data_dict.update({t: [s, ] if isinstance(s, str) else s})

        if self.use_rgb: base_suffixes = data_dict['im']
        else: base_suffixes=data_dict['depth']

        for j, b_suf in enumerate(base_suffixes):
            b_paths = glob(os.path.join(self.data_root, b_suf))
            for b_path in b_paths:
                apath = dict()
                b_path = b_path.replace('\\', '/')
                if not ProcUtils().isimpath(b_path) and not b_path.endswith('.npy'): continue

                for d_type in data_dict:
                    d_suf = data_dict[d_type][j]
                    d_path = b_path
                    for bb, dd in zip(b_suf.split('*'), d_suf.split('*')):
                        d_path = d_path.replace(bb, dd)
                    if not os.path.exists(d_path) and not d_path.endswith('.npy'): continue
                    apath.update({d_type: d_path})
                apath.update({'base': b_path})
                self.paths.append(apath)

        # if self.use_rgb and not self.use_depth:
        #     if isinstance(self.args.im_suffixes, str): self.args.im_suffixes = [self.args.im_suffixes]
        #     for im_suffix in self.args.im_suffixes:
        #         for path in glob(os.path.join(self.data_root, im_suffix), recursive=True):
        #             path = path.replace('\\', '/')
        #             if not ProcUtils().isimpath(path) and not path.endswith('.npy'): continue
        #             self.paths.append([path])
        # if not self.use_rgb and self.use_depth:
        #     if isinstance(self.args.depth_suffixes, str): self.args.depth_suffixes = [self.args.depth_suffixes]
        #     for depth_suffix in self.args.depth_suffixes:
        #         for path in glob(os.path.join(self.data_root, depth_suffix), recursive=True):
        #             path = path.replace('\\', '/')
        #             if not ProcUtils().isimpath(path) and not path.endswith('.npy'): continue
        #             self.paths.append([path])
        # if self.use_rgb and self.use_depth:
        #     if isinstance(self.args.im_suffixes, str): self.args.im_suffixes = [self.args.im_suffixes]
        #     if isinstance(self.args.depth_suffixes, str): self.args.depth_suffixes = [self.args.depth_suffixes]
        #     for im_suffix, depth_suffix in zip(self.args.im_suffixes, self.args.depth_suffixes):
        #         all_im_path = glob(os.path.join(self.data_root, im_suffix), recursive=True)
        #         for im_path in all_im_path:
        #             im_path = im_path.replace('\\', '/')
        #             if not ProcUtils().isimpath(im_path) and not im_path.endswith('.npy'): continue
        #             depth_path = im_path
        #             for im_suf, dp_suf in zip(im_suffix.split('*'), depth_suffix.split('*')):
        #                 depth_path = depth_path.replace(im_suf, dp_suf)
        #             if not os.path.exists(depth_path) and not depth_path.endswith('.npy'): continue
        #             self.paths.append([im_path, depth_path])

    def get_rgbd_from_path(self, rgbd_path=None, use_rgb=None, use_depth=None):
        """
        :param paths: list of rgbd path. paths[0]: image path, paths[-1]: depth path
        """
        if rgbd_path is None: rgbd_path = self.rgbd_path
        if use_rgb is None: use_rgb = self.use_rgb
        if use_depth is None: use_depth = self.use_depth
        rgb, depth = None, None
        if use_rgb:
            if ProcUtils().isimpath(rgbd_path['im']): rgb = cv2.imread(rgbd_path['im'])[:, :, ::-1]
            if rgbd_path['im'].endswith('.npy'): rgb = cv2.load(rgbd_path['im'])
            if self.flag_90_rot: rgb = np.rot90(rgb)
        if use_depth:
            if ProcUtils().isimpath(rgbd_path['depth']): depth = cv2.imread(rgbd_path['depth'], -1)
            if rgbd_path['depth'].endswith('.npy'):
                depth = np.load(rgbd_path['depth']).squeeze()
                if np.amax(depth)<10: depth = (1000*depth).astype('uint16')
            if self.flag_90_rot: depth = np.rot90(depth)

        extra=dict()
        for k in rgbd_path:
            if k in ['im', 'depth']: continue
            extra.update({f'{k}_path': rgbd_path[k]})

        rgbd = RGBD(rgb=rgb, depth=depth, workspace=self.workspace,
                    depth_min=self.args.sensor.depth_min, depth_max=self.args.sensor.depth_max, extra=extra)

        return rgbd

    def set_workspace(self):
        pts, crop_rect, bound_margin = None, None, None
        if hasattr(self.args, 'sensor'):
            if hasattr(self.args.sensor, 'crop_poly'):pts = self.args.sensor.crop_poly
            if hasattr(self.args.sensor, 'crop_rect'): crop_rect = self.args.sensor.crop_rect
            if hasattr(self.args.sensor, 'bound_margin'): bound_margin = self.args.sensor.bound_margin

        if pts is None and crop_rect is None:
            self.workspace = None
        else:
            self.workspace = WorkSpace(pts=pts, bbox=crop_rect, bound_margin=bound_margin)

    def set_workspace_from_cfg(self):
        if self.cfg_crop_poly is not None:
            ProcUtils().replace_confile_value(self.guiConfile, 'sensor', 'crop_poly', self.cfg_crop_poly)

        if self.cfg_crop_rect is not None:
            ProcUtils().replace_confile_value(self.guiConfile, 'sensor', 'crop_rect', self.cfg_crop_rect)

        self.guiConfile2textview()
        self.update_conf_viewer()
        self.reload_configs()

        if self.rgbds[-1] is not None:
            self.rgbds[-1].workspace = self.workspace
            self.update_im_viewer()

    def run_sensor(self, sensor):
        self.viewer_parent.select(sensor.viewer_ind)
        sensor.worker = sensor.module()
        sensor.worker.start(device_serial=sensor.serial)
        sensor.on = True

        sensor.btn_run.configure(text='Stop', command=partial(self.stop_sensor, sensor))
        sensor.flag_get_data = True
        control_thread = Thread(target=self.run_sensor_, args=(sensor,), daemon=True)
        control_thread.start()
        # self.run_sensor_(sensor_ind=sensor_ind)

    def run_sensor_(self, sensor):

        while sensor.on:
            timer = Timer()
            if sensor.flag_get_data:
                aa = \
                    sensor.worker.get_rgbd(workspace=self.workspace, rot_90=self.flag_90_rot,
                                           depth_min=self.args.sensor.depth_min, depth_max=self.args.sensor.depth_max)
                self.rgbds[sensor.viewer_ind]=aa

            if self.hold_detect():
                if 'module' in self.just_executed:
                    self.run_detect_(module=self.just_executed['module'],
                                     method_ind=self.just_executed['method_ind'], show_fps=True)
                if 'process' in self.just_executed:
                    self.run_process_(show_fps=True)
                sensor.flag_get_data = True
                continue

            rgbd_disp = self.rgbd_disp(self.rgbds[sensor.viewer_ind])
            if sensor.flag_get_data:
                cv2.putText(rgbd_disp, '%.2f FPS'%timer.fps(), (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX, self.args.disp.text_scale, self.args.disp.text_color, self.args.disp.text_thick)
            # self.rgbd_disp()
            self.update_im_viewer(rgbd_disp, viewer_ind=sensor.viewer_ind)


    def pause_sensor(self, sensor):
        if not hasattr(sensor, 'worker'): return
        self.viewer_parent.select(sensor.viewer_ind)
        sensor.on = False
        sensor.btn_pause.configure(text='Resume', command=partial(self.resume_sensor, sensor))

    def resume_sensor(self, sensor):
        if not hasattr(sensor, 'worker'): return
        self.viewer_parent.select(sensor.viewer_ind)

        sensor.on = True
        sensor.btn_pause.configure(text='Pause', command=partial(self.pause_sensor, sensor))
        control_thread = Thread(target=self.run_sensor_, args=(sensor,), daemon=True)
        control_thread.start()

    def stop_sensor(self, sensor):
        self.viewer_parent.select(sensor.viewer_ind)
        sensor.on = False
        sensor.worker.stop()
        sensor.btn_run.configure(text='Run', command=partial(self.run_sensor, sensor))

    def add_sensor(self, sensor):

        btn_frame = tk.Frame(master=self.fr_task_cmd)
        sensor_type = sensor.type

        lb = tk.Label(master=btn_frame, text=sensor.short_name)
        lb.grid(row=0, column=0)

        btn_run = tk.Button(master=btn_frame, text='Run', width=self.btn_w2_lb,
                            command=partial(self.run_sensor, sensor))
        btn_run.grid(row=0, column=1)

        btn_pause = tk.Button(master=btn_frame, text='Pause', width=self.btn_w2_lb,
                              command=partial(self.pause_sensor, sensor))
        btn_pause.grid(row=0, column=2)
        btn_frame.pack()

        sensor.btn_run = btn_run
        sensor.btn_pause = btn_pause

    def init_module(self, module, change_cfg_path=False):
        control_thread = Thread(target=self.init_module_, args=(module, change_cfg_path), daemon=True)
        control_thread.start()

    def init_module_(self, module, change_cfg_path=False):
        try:
            # if module.cfg_path is None: change_cfg_path = True
            if change_cfg_path:
                module.cfg_path = fd.askopenfilename(
                    initialdir='configs',title='Select a {} config file'.format(module.type))
                if len(module.cfg_path) == 0: module.cfg_path = None

            module.args = CFG(cfg_path=module.cfg_path, separate=True)
            self.get_module_args(module)
            worker = module.module(args=module.args)

            setattr(module, 'worker', worker)
            print('{} {} initialized ...'.format('+' * 10, module.name))

            if module.run_after_init_method_ind is not None:
                self.run_module(module,method_ind=module.run_after_init_method_ind)
        except:
            pass

    def run_module(self, module, method_ind=0, run_thread=None, record_module=True):
        print('{} {} run{} running ...'.format('+'*10, module.name, method_ind))
        if record_module: self.just_executed = {'module': module, 'method_ind': method_ind}
        mod_category = module.category
        if mod_category == 'detector':
            run_module_ = self.run_detect_
        elif mod_category == 'dataset':
            run_module_ = self.run_dataset_
        else: run_module_ = self.run_general_module_

        if run_thread is None: run_thread = module.run_thread
        if run_thread:
            control_thread = Thread(target=run_module_, args=(module, method_ind), daemon=True)
            control_thread.start()
        else:
            run_module_(module, method_ind)

    def run_general_module_(self, module, method_ind=0):
        if not hasattr(module, 'worker'): return
        module.worker.gui_run_module(method_ind=method_ind)

    def run_dataset_(self, module, method_ind=0):
        timer = Timer()
        tab_ind = self.get_current_tab_ind()
        if tab_ind >= self.num_viewer: return
        # self.viewer_parent.select(self.num_viewer-1)
        # module.worker.args.show_steps = self.show_steps_var.get()

        rgbd = self.rgbds[-1]
        _, filename = os.path.split(self.rgbd_path[0])
        self.detected = module.worker.gui_show_single(rgbd=rgbd, method_ind=method_ind, filename=filename)
        self.show_detected(rgbd, mod_type=module.type)

        print('{} {} executed in {}'.format('+' * 10, module.name, timer.pin_times_str()))
        self.flag_show_detected = True

    def tab2sensor(self, tab_ind):
        for mod in self.modules:
            if 'sensor' not in mod.category: continue
            if tab_ind == mod.viewer_ind:
                sensor = mod
                break
        return sensor

    def run_detect_(self, module, method_ind=0, show_fps=False):
        tab_ind = self.get_current_tab_ind()
        if tab_ind >= self.num_viewer: return
        if not hasattr(module, 'worker'): return
        # self.just_executed_type = 'detector'
        flag_select_sensor = tab_ind < (self.num_viewer - 1)
        module.worker.args.flag.show_steps = self.show_steps_var.get()

        if show_fps:timer=Timer()
        # input_name = module.input_var.get()
        # if input_name =='Img':
        if flag_select_sensor: self.run_detect_sensor_(module, method_ind, self.tab2sensor(tab_ind))
        else: self.run_detect_dir_(module, method_ind)
        # else:
        #     if self.detected is None: return
        #     if input_name != self.detected['module_name']: return
        #     self.execute_detect_and_show(module,method_ind,self.detected['rgbd'], self.detected['filename'], detected=self.detected)

        if show_fps: cv2.putText(self.detected_disp, '%.2f FPS' % timer.fps(), (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, self.args.disp.text_scale, self.args.disp.text_color, self.args.disp.text_thick)
        if self.detected_disp is not None: self.update_im_viewer(im=self.rgbd_disp(), viewer_ind=tab_ind)



    # def run_secondary_detect(self, module, method_ind, clean_detect=False):
    #     if self.detected is None: return
    #     if module.input_var.get()!=self.detected['module_name']: return
    #     self.execute_detect_and_show(module,method_ind,self.detected['rgbd'], self.detected['filename'])
    #     # self.update_im_viewer()
    #     if clean_detect: self.detected=None

    def show_detected(self, viewer_ind=0):
        # self.detected_disp = None
        if self.detected is None: return

        self.detected_disp = np.copy(self.detected['im'])
        # self.update_im_viewer(im=self.detected_disp, viewer_ind=viewer_ind)

    def execute_detect_and_show(self, module, method_ind, rgbd, filename='unnamed', viewer_ind=0, detected=None):
        self.detected = module.worker.gui_process_single(rgbd=rgbd, method_ind=method_ind, filename=filename,
                                                         disp_mode=self.flag_imshow_modes[0], detected=detected)
        if isinstance(self.detected, dict):
            self.detected.update({'module_name':module.short_name, 'filename': filename})
            if 'rgbd' not in self.detected: self.detected.update({'rgbd': RGBD(rgb=self.detected['im'],
                                                                           workspace=rgbd.workspace,
                                                                           depth_min=rgbd.depth_min,
                                                                           depth_max=rgbd.depth_max)})

            announce = self.detected[
                'announce'] if 'announce' in self.detected else f'{module.short_name} run{method_ind} finished'
            self.highlight_label.configure(text=announce)
        self.flag_show_detected = True
        self.show_detected(viewer_ind=viewer_ind)

    def run_detect_sensor_(self, module, method_ind, sensor):
        sensor.flag_get_data = False
        sensor_root_dir = os.path.join('data', sensor.type, ProcUtils().get_current_time_str('%m%d'))
        sensor_result_dir = os.path.join(sensor_root_dir, '{}_run{}'.format(module.short_name, method_ind + 1))

        rgbd = self.rgbds[sensor.viewer_ind]
        input_name = module.input_var.get()
        detected = None if input_name=='Img' else self.detected
        self.execute_detect_and_show(module,method_ind,rgbd, viewer_ind=sensor.viewer_ind, detected=detected)

        if not self.hold_detect() and self.detected_disp is not None:
            # save
            filename = ProcUtils().get_current_time_str() + '.png'
            ArrayUtils().save_array_v3(rgbd.bgr(), os.path.join(sensor_root_dir, 'image', filename))
            ArrayUtils().save_array_v3(rgbd.depth, os.path.join(sensor_root_dir, 'depth', filename))
            if len(self.detected_disp.shape)<3: out = self.detected_disp
            else: out = self.detected_disp[:,:,::-1]
            ArrayUtils().save_array_v3(out, os.path.join(sensor_result_dir, filename))

    def run_detect_dir_(self, module, method_ind=0):
        db_result_dir = os.path.join(self.args.path.root_dir, f'{module.short_name}_run{method_ind}')
        flag_run_all = self.flag_run_all_var.get() or self.flag_run_acc_var.get()
        flag_run_acc = self.flag_run_acc_var.get()


        if flag_run_all:
            paths = self.paths
        else:
            if hasattr(self, 'rgbd_path'):
                paths = [self.rgbd_path]
            else:
                print('{} No image loaded...'.format('+'*10))
                return

        num_im = len(paths)
        try: range_min = min(max(0,int(self.str_range_min.get())), num_im-1)
        except: range_min = 0
        try: range_max = min(max(0,int(self.str_range_max.get())), num_im-1)
        except: range_max = num_im
        range_max = max(range_min, range_max)

        if flag_run_acc: module.worker.init_acc()
        j=-1
        # for j, path in enumerate(paths):
        while True:
            j+=1
            if self.flag_loop_var.get():
                j %= num_im
                if hasattr(module.worker, 'reload_params') and j==0: module.worker.reload_params()
            if j>=num_im: break
            path = paths[j]
            if flag_run_all:
                if j<range_min: continue
                if j>=range_max: continue

                rgbd = self.get_rgbd_from_path(path)
                print('[%d/%d] %s' % (j, num_im, path['base']))
            else:
                rgbd = self.rgbds[-1]

            _, filename = os.path.split(path['base'])
            timer = Timer()
            input_name = module.input_var.get()
            detected = None if input_name == 'Img' else self.detected
            self.execute_detect_and_show(module,method_ind,rgbd,filename, detected=detected)

            timer.pin_time('predict_time')
            print(timer.pin_times_str())

            # save
            if self.detected_disp is not None:
                _, filename = os.path.split(path['base'])
                if len(self.detected_disp.shape)<3: out = self.detected_disp
                else: out = self.detected_disp[:,:,::-1]
                ArrayUtils().save_array_v3(out, os.path.join(db_result_dir, filename))
                # if module.run_thread:  self.update_im_viewer(im=self.detected_disp)

        if flag_run_acc:
            module.worker.finalize_acc()
            if hasattr(module.worker, 'print_acc'):
                self.highlight_label.configure(text=module.worker.print_acc())

    def popup_args(self, module, name='unnamed'):

        popup_root = tk.Toplevel(self.window)
        popup_root.geometry('{}x{}'.format(self.gui_size[0] // 2, self.gui_size[1]))
        popup_root.title(name)

        fr_button = tk.Frame(master=popup_root)
        fr_viewer = tk.Frame(master=popup_root)
        viewer_parent = ttk.Notebook(master=fr_viewer)

        # tbx_args_ = self.set_textbox_viewer(name='args_', viewer_parent=viewer_parent)
        # GuiUtils().update_textbox(tbx_args_, module.args.to_string())
        tbx_args = self.set_textbox_viewer(name='args', viewer_parent=viewer_parent)
        GuiUtils().update_textbox(tbx_args, module.args.to_string())

        def reload_args(module, tbx_args_, tbx_args):
            module.args = CFG().from_string(GuiUtils().textbox2text(tbx_args_), separate=True)
            self.get_module_args(module)
            if hasattr(module, 'worker'):
                module.worker.args = module.args
                if hasattr(module.worker,'reload_params'): module.worker.reload_params()
            GuiUtils().update_textbox(tbx_args, module.args.to_string())

        def save_args(module, tbx_args_, tbx_args):
            reload_args(module, tbx_args_, tbx_args)
            args = CFG().from_string(GuiUtils().textbox2text(tbx_args_), separate=True)
            if module.cfg_path is None:
                args.write()
            else:
                args.write(module.cfg_path)

        def reset_args(module, tbx_args_, tbx_args):
            module.args = CFG(cfg_path=module.cfg_path, separate=True)
            self.get_module_args(module)
            if hasattr(module, 'worker'): module.worker.args = module.args
            GuiUtils().update_textbox(tbx_args, module.args.to_string())
            GuiUtils().update_textbox(tbx_args_, module.args.to_string())

        tk.Button(master=fr_button, text='Reload configs', width=10,
                  command=partial(reload_args, module, tbx_args, tbx_args)).grid(row=0, column=0)
        tk.Button(master=fr_button, text='Save configs', width=10,
                  command=partial(save_args, module, tbx_args, tbx_args)).grid(row=0, column=1)
        tk.Button(master=fr_button, text='Reset configs', width=10,
                  command=partial(reset_args, module, tbx_args, tbx_args)).grid(row=0, column=2)

        viewer_parent.pack(fill=tk.BOTH)
        fr_button.pack(fill=tk.BOTH, expand=False)
        fr_viewer.pack(fill=tk.BOTH, expand=False)

    def copy_args_sections_from_module(self, module_args_, section_names=None):
        if module_args_ is None:
            print('{} config is None o...'.format('+' * 10))
            return
        if section_names is None: section_names = module_args_.keys()
        for section_name in section_names:
            if not hasattr(module_args_, section_name):
                print('{} config is None or has no {} section ...'.format('+' * 10, section_name))
                continue
            default_section = self.args_.__getattribute__(section_name)
            module_section = module_args_.__getattribute__(section_name)
            default_options = default_section.keys()
            for option_name in module_section.keys():
                default_section.__setattr__(option_name, module_section.__getattribute__(option_name))

    def set_preprocess_params_from_module(self, module):
        self.copy_args_sections_from_module(module.args_, section_names=['sensor'])
        self.update_after_change_args_()

    def run_process(self):
        control_thread = Thread(target=self.run_process_, daemon=True)
        control_thread.start()

    def run_process_(self, show_fps=False):
        timer = Timer()
        process_module_inds = eval(self.str_process_module_inds.get())
        if not hasattr(process_module_inds,'__iter__'): process_module_inds=[process_module_inds,]
        self.just_executed = {'process': process_module_inds}
        # self.just_executed_process_ind = process_ind
        modules = [self.modules[j] for j in process_module_inds]
        num_module = len(modules)
        for j, module in enumerate(modules):
            if j==0: module.input_var.set('Img')
            else: module.input_var.set(f'{modules[j-1].short_name}')
            self.run_module(module, eval(module.method_ind_var.get()), run_thread=False, record_module=False)
            timer.pin_time(module.name)
            if self.process_ask_next_var.get() and j<num_module-1:
                rep = msgbox.askquestion(title='Process', message='want to execute next?')
                if rep=='no': break
        print(timer.pin_times_str())

    def add_module(self, module_ind):
        module = self.modules[module_ind]
        names = [mod.short_name for j,mod in enumerate(self.modules) if j!=module_ind and mod.category!='sensor']
        names = ['Img',]+names
        input_sources = names[module_ind:] + names[:module_ind]

        tk.Label(master=self.fr_task_cmd, text=f'=====MODULE {module_ind}=====').pack()
        module_frame = tk.Frame(master=self.fr_task_cmd,
                                highlightbackground='green', highlightcolor='green', highlightthickness=2)
        lb_frame = tk.Frame(master=module_frame)
        col=0
        # if module.input_sources is not None:
        module.input_var = tk.StringVar()
        module.input_var.set(input_sources[0])
        tk.OptionMenu(lb_frame, module.input_var, *input_sources).grid(row=0, column=col)
        col+=1

        tk.Label(master=lb_frame, text=f'->{module.name}').grid(row=0, column=col)
        lb_frame.pack()
        col += 1

        module.method_ind_var = tk.StringVar()
        module.method_ind_var.set(0)
        tk.OptionMenu(lb_frame, module.method_ind_var, *list(range(module.num_method))).grid(row=0, column=col)



        btn_frame = tk.Frame(master=module_frame)
        btn_init = tk.Button(master=btn_frame, text='Int', width=self.btn_w5,
                             command=partial(self.init_module, module, True))
        btn_init.grid(row=0, column=0)
        btn_init_ = tk.Button(master=btn_frame, text='Int_', width=self.btn_w5,
                              command=partial(self.init_module, module, False))
        btn_init_.grid(row=0, column=1)
        btn_args_ = tk.Button(master=btn_frame, text='args_', width=self.btn_w5,
                              command=partial(self.popup_args, module, '{} args_'.format(module.name)))
        btn_args_.grid(row=0, column=2)
        # btn_args = tk.Button(master=btn_frame, text='args', width=self.btn_w5,
        #                     command=partial(self.popup_args, module.args, '{} args'.format(module.name)))
        # btn_args.grid(row=0, column=3)
        btn_prep = tk.Button(master=btn_frame, text='prep', width=self.btn_w5,
                             command=partial(self.set_preprocess_params_from_module, module))
        btn_prep.grid(row=0, column=3)
        btn_dir = tk.Button(master=btn_frame, text='dir', width=self.btn_w5,
                            command=partial(self.open_dir_from_args, module))
        btn_dir.grid(row=0, column=4)

        num_method = module.num_method

        btn_runs = []
        # if num_method == 1:
        #     btn_run = tk.Button(master=btn_frame, text='Run', width=self.btn_w3, command=partial(self.run_module, module_ind, 0))
        #     btn_run.grid(row=0, column=2)
        #     btn_runs.append(btn_run)
        #     btn_frame.pack()
        # else:
        btn_frame.pack()
        btn_frame = tk.Frame(master=module_frame)
        for i in range(num_method):
            btn_text = 'Run'
            if num_method > 1: btn_text += str(i)
            row = i // 3
            col = i - 3 * row
            btn_run = tk.Button(master=btn_frame, text=btn_text, width=self.btn_w3,
                                command=partial(self.run_module, module, i))
            btn_run.grid(row=row + 1, column=col)
            btn_runs.append(btn_run)
        btn_frame.pack()

        module_frame.pack()
        self.init_module(module, change_cfg_path=False)


    def save_image(self):
        tab_ind = self.get_current_tab_ind()
        if tab_ind >= self.num_viewer - 1: return
        rgbd = self.rgbds[tab_ind]
        sensor_name = self.viewer_ind_to_sensor(tab_ind).name
        subdir = ProcUtils().get_current_time_str('%m%d')
        filename = ProcUtils().get_current_time_str() + '.png'
        if rgbd.hasRgb:
            ArrayUtils().save_array_v3(rgbd.bgr(), filepath=os.path.join('data', sensor_name, subdir, 'rgb', filename))
            ArrayUtils().save_array_v3(rgbd.crop_rgb()[:, :, ::-1],
                                       filepath=os.path.join('data', sensor_name, subdir, '_crop', 'rgb', filename))
        if rgbd.hasDepth:
            ArrayUtils().save_array_v3(rgbd.depth,
                                       filepath=os.path.join('data', sensor_name, subdir, 'depth', filename))
            ArrayUtils().save_array_v3(rgbd.crop_depth(),
                                       filepath=os.path.join('data', sensor_name, subdir, '_crop', 'depth', filename))

    def viewer_ind_to_sensor(self, viewer_ind=0):
        for mod in self.modules:
            if not hasattr(mod, 'viewer_ind'): continue
            if mod.viewer_ind == viewer_ind: return mod

    def save_video(self):
        tab_ind = self.get_current_tab_ind()
        if tab_ind >= self.num_viewer - 1: return
        if self.rgbds[tab_ind] is None: return

        self.btn_save_video.configure(text='Stop save', command=partial(self.stop_save_video, tab_ind))
        self.acc_rgbd = []

        self.flag_save_video = True
        thread_control = Thread(target=self.acc_frame_, args=(tab_ind,), daemon=True)
        thread_control.start()

    def acc_frame_(self, sensor_ind):
        while self.flag_save_video:
            print('accumulating frame ...')
            self.acc_rgbd.append(self.rgbds[sensor_ind])
            time.sleep(1 / self.args.sensor.vid_fps)
            ProcUtils().clscr()

    def stop_save_video(self, tab_ind):
        if not hasattr(self, 'acc_rgbd'): return
        self.flag_save_video = False
        self.btn_save_video.configure(text='Save video', command=self.save_video)
        thread_control = Thread(target=self.stop_save_video_, args=(tab_ind,), daemon=True)
        thread_control.start()

    def stop_save_video_(self, viewer_ind):
        subdir = ProcUtils().get_current_time_str('%m%d')
        height, width = self.acc_rgbd[0].height, self.acc_rgbd[0].width
        writer = ProcUtils().init_vid_writer(size=(width, height),
                                             fold=os.path.join('data', self.viewer_ind_to_sensor(viewer_ind).name,
                                                               subdir),
                                             fps=self.args.sensor.vid_fps, isColor=True)
        for rgbd in self.acc_rgbd:
            writer.write(rgbd.bgr())

        self.acc_rgbd = []
        writer.release()
        print('video saved ...')

    def create_video(self):
        thread_control = Thread(target=self.create_video_, daemon=True)
        thread_control.start()

    def create_video_(self):
        if not hasattr(self, 'paths'): return
        num_im = len(self.paths)
        if num_im <= 1: return

        rgbs = [cv2.imread(path[0]) for path in self.paths]
        h, w = rgbs[0].shape[:2]
        writer = ProcUtils().init_vid_writer(size=(w, h), fold=self.data_root,
                                             fps=self.args.sensor.vid_fps, isColor=True)
        for rgb in rgbs:
            writer.write(rgb)

        rgbs = []
        writer.release()
        print('video saved ...')


class GuiModule():
    def __init__(self, module, type='detector', name='unnamed', category='detector', args=None,
                 cfg_path=None, num_method=1, short_name=None, run_thread=True, run_after_init_method_ind=None,
                 input_sources=None,**kwargs):
        self.module = module
        self.type = type
        self.name = name
        self.category = category
        self.args=args
        self.cfg_path = cfg_path
        self.num_method = num_method
        self.short_name = short_name
        self.run_thread = run_thread
        self.run_after_init_method_ind = run_after_init_method_ind
        self.input_sources = input_sources
        if self.short_name is None: self.short_name = self.name[:min(len(self.name), 3)]
        for key in kwargs:
            if not hasattr(self, key):setattr(self, key, kwargs[key])
            else: self.__setattr__(key, kwargs[key])

class GuiProcess():
    def __init__(self, gui_modules, module_inds=None, method_inds=None, input_sources=None, name='unnamed_process',
                 im_in_ret=True,ask_next=True):
        self.gui_modules=gui_modules
        self.method_inds = method_inds if method_inds is not None else [0,]*len(gui_modules)
        self.module_inds = module_inds if module_inds is not None else list(range(len(gui_modules)))
        self.input_sources = input_sources
        self.name=name
        self.ask_next = ask_next
        self.im_in_ret=im_in_ret






if __name__ == '__main__':

    pass


















