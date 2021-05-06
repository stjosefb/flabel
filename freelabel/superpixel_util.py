import numpy as np
import sys, os

from PIL import Image


def draw_boundaries(img_np,labels):
    try:
        print(img_np.shape)
        print(img_np)
        #img = Image.open(imgname)
        #img = np.array(img)

        ht,wd = labels.shape

        for y in range(1,ht-1):
            for x in range(1,wd-1):
                if labels[y,x-1] != labels[y,x+1] or labels[y-1,x] != labels[y+1,x]:
                    img_np[y,x,:] = 0

        return img_np
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)      