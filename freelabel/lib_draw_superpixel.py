import sys, os
from skimage import color

import numpy as np

from PIL import Image
from PIL import ImageDraw
#from PIL import ImageCms
#import pyCMS
#import cv2


def draw_boundaries(img_np,labels):
    try:
        #print(img_np.shape)
        #print(img_np)
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

    
def drawBoundariesOnly(img_np,labels,numlabels,dict_label_pos={},is_draw_label=False):
    try:    
        #print(img_np.shape)
        width, height = labels.shape

        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        if is_draw_label:
            d1 = ImageDraw.Draw(img)
            for label, label_pos in dict_label_pos.items():
                d1.text(label_pos[::-1], str(label), fill=(0, 0, 0))

        img = np.array(img)

        ht,wd = labels.shape

        for y in range(1,ht-1):
            for x in range(1,wd-1):
                if labels[y,x-1] != labels[y,x+1] or labels[y-1,x] != labels[y+1,x]:
                    img[y,x,:] = (255, 0, 0, 255)

        return img
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
        
def dictLabelLabToRgb(dict_label_color_lab):
    try:
        dict_label_color_rgb = {}
        
        lab_values = list(dict_label_color_lab.values());
        #print([rgb_values])
        lab_keys = dict_label_color_lab.keys();
        #print(list(rgb_keys))
        rgb_values = color.lab2rgb([lab_values])
        #srgb_p = ImageCms.createProfile("sRGB")
        #lab_p  = ImageCms.createProfile("LAB")        
        #ImageCms.applyTransform(rgb_values, ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB"))
        #lab_values = pyCMS.profileToProfile(rgb_values, pyCMS.createProfile("sRGB"), pyCMS.createProfile("LAB"))
        #print(lab_values)
        #lab_values = []
        #for rgb in rgb_values:
        #    lab = rgb2lab(rgb)
        #    lab_values.append(lab)
        #lab_values = [lab_values]
        
        #BGR = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
        
        for idx, key in enumerate(list(lab_keys)):        
            dict_label_color_rgb[key] = (rgb_values[0][idx][0]*255,rgb_values[0][idx][1]*255, rgb_values[0][idx][2]*255)
        
        return dict_label_color_rgb
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)     
        

"""        
def rgb2lab ( inputColor ) :

    num = 0
    RGB = [0, 0, 0]

    for value in inputColor :
        value = float(value) / 255

    if value > 0.04045 :
        value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
    else :
        value = value / 12.92

    RGB[num] = value * 100
    num = num + 1

    XYZ = [0, 0, 0,]

    X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
    Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
    Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
    XYZ[ 0 ] = round( X, 4 )
    XYZ[ 1 ] = round( Y, 4 )
    XYZ[ 2 ] = round( Z, 4 )

    XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
    XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
    XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

    num = 0
    for value in XYZ :

        if value > 0.008856 :
            value = value ** ( 0.3333333333333333 )
        else :
            value = ( 7.787 * value ) + ( 16 / 116 )

    XYZ[num] = value
    num = num + 1

    Lab = [0, 0, 0]

    L = ( 116 * XYZ[ 1 ] ) - 16
    a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
    b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

    Lab [ 0 ] = round( L, 4 )
    Lab [ 1 ] = round( a, 4 )
    Lab [ 2 ] = round( b, 4 )

    return Lab        
"""
        
        
def draw_superpixels(img_np,labels,dict_label_color_rgb):
    img_np_sp_color = np.copy(img_np)

    ht,wd = labels.shape

    for y in range(0,ht):
        for x in range(0,wd):
            img_np_sp_color[y,x,:] = list(dict_label_color_rgb[labels[y,x]])
            #img_np_sp_color[y,x,:] = [dict_label_color_rgb[labels[y,x]][0]*255, ]
    
    return img_np_sp_color