import callRGR
import img_convert as ic
import math
import numpy as np
import sys, os

def create_superpixel(url, m):
    #print('create_superpixel')
    
    img_byte_arr = ic.img_url_to_bytearr(url)
    #print('after img_url_to_bytearr')
    
    img_np = ic.img_bytearr_to_np(img_byte_arr)
    #print(img_np)
    #print(img_np.shape)
    
    img_sp_np = get_superpixel_snic(img_np, m)
    
    #img_pil = img_bytearr_to_pil(img_byte_arr)
    #print('after img_bytearr_to_pil')
    
    #img_np = img_pil_to_np(img_pil)
    img_pil = ic.img_np_to_pil(img_sp_np)
    
    #img_pil = img_np_to_pil(img_np)
    img_base64 = ic.img_pil_to_base64(img_pil) 
    #print('after img_pil_to_base64')
    
    return img_base64


def get_superpixel_snic(img_np, m):
    try:
        height, width, channels = img_np.shape
        # allocate memory for output returned by reg.growing C++ code
        RGRout = np.zeros((width*height), dtype=int)        
        img_b = img_np[:,:,2].flatten()
        img_g = img_np[:,:,1].flatten()
        img_r = img_np[:,:,0].flatten()    
        preSeg = np.int32(np.zeros((height,width))).flatten() # not used
        num_superpixel = 200
        S, num_superpixel = get_snic_seeds(height,width,num_superpixel)
        
        # call RGR
        print(type(img_r))
        print(type(img_g))
        print(type(img_b)) 
        print(img_r.dtype)
        print(img_g.dtype)
        print(img_b.dtype)        
        print(preSeg.astype(np.int32).shape)
        print(S.astype(np.int32).shape)
        print(width)
        print(height)    
        print(int(num_superpixel))
        print(m)
        print(RGRout.shape)
        out_ = callRGR.callRGR(img_r.astype(np.int32), img_g.astype(np.int32), img_b.astype(np.int32), preSeg.astype(np.int32), S.astype(np.int32), width, height, int(num_superpixel), m, RGRout.astype(np.int32))
        #PsiMap = np.asarray(out_)
        #print(PsiMap.shape)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno) 
        
    return img_np
    
    
    
def get_snic_seeds(height,width,num_superpixel):
    S = np.zeros((height, width))
    #print(S)
    #print(S.shape)    
    try:    
        sz = width * height
        gridstep = math.sqrt(sz/num_superpixel) + 0.5
        halfstep = gridstep / 2
        h = float(height)
        w = float(width)
        
        xstep = int(width/gridstep)
        ystep = int(height/gridstep)
        
        err1 = abs(xstep*ystep-num_superpixel)
        err2 = abs(int(width/(gridstep-1))*int(height/(gridstep-1))-num_superpixel)
        
        if (err2 < err1):
            gridstep = gridstep - 1
            xstep = width/gridstep
            ystep = height/gridstep
        
        num_superpixel_actual = xstep * ystep
        n = 0
        
        y = halfstep
        rowstep = 0
        while (y < height) and (n < num_superpixel):
            x = halfstep
            while (x < width) and (n < num_superpixel):
                if (y <= h-halfstep) and (x <= w-halfstep):
                    #print(y,x)
                    S[int(y),int(x)] = 255
                    n = n + 1
                x = x + gridstep
            y = y + gridstep
            rowstep = rowstep + 1
        
        #print(S)
        #print(S.shape)
        
        # test save seeds as image
        #ic.img_np_to_file(S, 'static/'+'dummy1'+'/superpixel_seeds'+''+'.png')
        
        S = S.flatten(order='F')
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)   
    return S, num_superpixel_actual        