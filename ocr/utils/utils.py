import numpy as np


def pil_to_np(pil_image):
    return np.array(pil_image)

def normalizer(i):
    x0 = i['box'][0][0]
    y0 = i['box'][0][1]
    x1 = i['box'][2][0]
    y1 = i['box'][2][1]
    bbox = [x0, y0, abs(x1 - x0), abs(y1 - y0)]
    i['bbox'] = bbox
    return i

