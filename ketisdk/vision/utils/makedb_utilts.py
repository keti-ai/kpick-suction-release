import json
from ..import_basic_utils import *
import numpy as np
import cv2


def draw_poly(im, poly, color=(0,255,0), thick=2):
    out = np.copy(im)
    for i in range(len(poly) - 1):
        cv2.line(out, tuple(poly[i]), tuple(poly[i + 1]), color, thick)
    cv2.line(out, tuple(poly[-1]), tuple(poly[0]), color, thick)

    return out

def save_polyogon_json(json_path, im_info, class_list, polygons):
    # val json
    Ann_dict = dict()

    Ann_dict.update({'Info': im_info})
    Ann_dict.update({'Classes': class_list})
    Ann_dict.update({'Polygons': polygons})


    instance_json_obj = open(json_path, 'w+')
    instance_json_obj.write(json.dumps(Ann_dict))
    instance_json_obj.close()

def visualize_polygon(json_path, im=None, im_path=None, text_scale=1.5, text_thick=3,
                      rect=(0,0,100,100),space=10,alpha=0.5, up2down=True):
    assert im is not None or im_path is not None
    if ProcUtils().isimpath(im_path): im = cv2.imread(im_path)

    with open(json_path) as json_file:
        ann_dict = json.load(json_file)
        json_file.close()

    info = ann_dict['Info']
    classes = ann_dict['Classes']
    polygons = ann_dict['Polygons']

    classes_unq = list(np.unique(classes))
    colors = ProcUtils().get_color_list(len(classes_unq))
    color_dict=dict()
    for cls, color in zip(classes_unq, colors):
        color_dict.update({cls:color})

    out = np.copy(im)
    for cls, poly in zip(classes, polygons):
        out = draw_poly(out, poly, color=color_dict[cls])

    out = VisUtils().draw_texts_blend(out, texts=classes_unq, colors=colors, scale=text_scale,thick=text_thick, text_rect=rect,space=space,alpha=alpha, up2down=up2down)

    return out

def get_rect_from_poly(poly):
    poly_array = np.array(poly)

    left, top = np.amin(poly_array, axis=0)
    right, bottom = np.amax(poly_array, axis=0)

    return (left, top, right, bottom)