import numpy as np
import urllib.request as ur
import base64
import io
import cv2 as cv

from PIL import Image


def create_superpixel(url):
    #print('create_superpixel')
    
    img_byte_arr = img_url_to_bytearr(url)
    #print('after img_url_to_bytearr')
    
    img_np = img_bytearr_to_np(img_byte_arr)
    print(img_np)
    
    #img_pil = img_bytearr_to_pil(img_byte_arr)
    #print('after img_bytearr_to_pil')
    
    #img_np = img_pil_to_np(img_pil)
    img_pil = img_np_to_pil(img_np)
    
    #img_pil = img_np_to_pil(img_np)
    img_base64 = img_pil_to_base64(img_pil) 
    #print('after img_pil_to_base64')
    
    return img_base64


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