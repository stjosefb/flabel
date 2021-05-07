import numpy as np
import urllib.request as ur
import base64
import io
import cv2 as cv
import sys, os

from PIL import Image


def img_bytearr_to_np(img_byte_arr):
    img_np_bgr = cv.imdecode(img_byte_arr, cv.IMREAD_COLOR)
    img_np = cv.cvtColor(img_np_bgr, cv.COLOR_BGR2RGB)
    return img_np
    
    
def img_bytearr_to_pil(img_byte_arr):
    #img = cv.imdecode(img_arr, cv.IMREAD_COLOR)
    img_pil = Image.open(io.BytesIO(img_byte_arr))
    return img_pil
    
    
def img_url_to_bytearr(url):
    resp = ur.urlopen(url)
    img_byte_arr = np.asarray(bytearray(resp.read()), dtype="uint8")
    #print(img_byte_arr.shape)
    return img_byte_arr


def img_np_to_pil(img_np):
    img_pil = Image.fromarray(img_np)
    return img_pil
    

def img_pil_to_base64(img_pil):
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()) 
    return img_base64
    
    
def img_np_to_file(img_np, file_path):
    print('img_np_to_file')
    try:
        #print(img_np)
        img_np2 = img_np.astype(np.uint8)
        #print(img_np2)
        img_pil = Image.fromarray(img_np2)
        img_pil.save(file_path, format="png")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)     
    
