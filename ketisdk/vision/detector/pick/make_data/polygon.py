from libs.import_basic_utils import *
from libs.utils.makedb_utilts import *
from libs.basic.basic_onmouse import BasSeqImOnMouseLabel
from libs.basic.basic_objects import BasObj
from libs.basic.basic_objects import BasSeqImObj, BasPoly, BasJson
from libs.basic.basic_sequence import SeqAccumulator, BasSeqImCallBack
import os, cv2


def show_polygon(rgbd, json_path, args):
    if json_path is None: return rgbd.crop_rgb()
    return visualize_polygon(im=rgbd.crop_rgb(), json_path=json_path,
                             text_scale=args.text_scale, text_thick=args.text_thick,
                             rect=args.text_rect, space=args.text_space, alpha=args.text_alpha, up2down=args.up2down)

def polys2masks(polys, im_size):
    mask = np.zeros(im_size, np.uint16)
    num_poly = len(polys)
    for j in range(num_poly):
        poly = np.array(polys[j]).astype('int32')

        cv2.fillPoly(mask,poly.reshape((1, )+ poly.shape),j+1000)
    return mask

class DrawPolyon(BasObj):
    def get_ann(self, inputs, filename='unname', title='viewer'):
        json_path = os.path.join(self.args.root_dir, self.args.ann_dir, filename + '.json')
        if not os.path.exists(json_path): return None
        return json_path

class OnMousePolygonDrawer(BasSeqImOnMouseLabel):
    def start_on_mouse(self, filename, **kwargs):
        ann_dict = super().start_on_mouse(filename=filename)
        self.disp_colors = []
        if ann_dict is None: self.class_list, self.polygons = [], []
        else:
            self.class_list, self.polygons = ann_dict['Classes'], ann_dict['Polygons']
            for cls in self.class_list:
                for key in self.key_dict:
                    if self.key_dict[key]['cls'] != cls: continue
                    self.disp_colors.append(self.key_dict[key]['color'])


    def show_draws(self,rgbd):
        self.im = PolyUtils().draw_poly(self.im, self.click_locs, thick=self.args.line_thick,
                        color=self.disp_colors[-1])  # permenantly show


    def process_click_locs(self, rgbd):
        self.polygons.append(self.click_locs)

        # region class
        promtstr = 'Class? \n'
        for key in self.key_dict:
            promtstr += '%s: \t%s\n' % (key, self.key_dict[key]['cls'])
        print(promtstr)

        while True:
            key = chr(cv2.waitKey())
            if key in self.key_dict:
                break
            else:
                print('wrong key, press other key ... ')
        self.class_list.append(self.key_dict[key]['cls'])
        self.disp_colors.append(self.key_dict[key]['color'])
        ProcUtils().clscr()

        # grip = GRIP(pts = self.click_locs[:2])
        # self.im = rgbd.disp(args=self.args, grips=GRIPS(grips=grip))

        self.show_draws(rgbd)

    def stop_on_mouse(self, rgbd, filename):
        super().stop_on_mouse(rgbd=rgbd, filename=filename)
        json_name, _ = os.path.splitext(filename)
        json_name += '.json'
        JsonUtils().save_poly_json(os.path.join(self.args.root_dir, self.args.ann_poly,json_name), self.im_info, self.class_list, self.polygons, )


class PolygonShower(BasSeqImCallBack, BasPoly):
    def process_single(self):
        super().process_single()
        self.show_polys(rgbd=self.rgbd, filename=self.filename)

    def show_polys(self, rgbd, filename='unnamed'):
        ann_polygon = os.path.join(self.args.root_dir, self.args.ann_poly)
        json_path = os.path.join(ann_polygon, filename + '.json')
        if not os.path.exists(json_path):
            print('%s does not exist ...' %json_path)
            rgbd.show(args=self.args)
        else:
            out = self.draw_polys_from_json(json_path, rgbd.bgr())
            cv2.imshow('viewer', out)
            # return self.draw_polys_on_rgbd(json_path, rgbd)


class PolygonAnalyzer(SeqAccumulator, BasJson):
    def process_single(self):
        super().process_single()
        in_dir, self.filename = os.path.split(self.rgbd_path[0])
        self.get_rgbd_from_path(self.rgbd_path)
        # ----------------------------->>>>>>>>>> Main
        ann_polygon = os.path.join(self.args.root_dir, self.args.ann_polygon)
        json_path = os.path.join(ann_polygon, self.filename).replace('.png', '.json')
        if not os.path.exists(json_path):
            print('%s does not exist ...' % json_path)
            return

        ann_dict = self.read_json(json_path=json_path)
        self.analyze_classes(ann_dict['Classes'])


    def analyze_classes(self, classes):
        anly = dict()
        for cls in classes:
            if cls not in anly: anly.update({cls: 1})
            else : anly[cls] += 1

        for cls in anly:
            print('cls %s: %d' %(cls, anly[cls]))




