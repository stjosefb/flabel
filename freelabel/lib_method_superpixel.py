import callRGR
import math
import numpy as np
import sys, os

import lib_img_convert as ic
import lib_superpixel_util as su
import lib_grow_selection

from PIL import Image
from PIL import ImageDraw

def create_superpixel(url, m, in_traces):
    try:
        #print('create_superpixel')
        
        img_byte_arr = ic.img_url_to_bytearr(url)
        #print('after img_url_to_bytearr')
        
        img_np_orig = ic.img_bytearr_to_np(img_byte_arr)
        img_np = ic.img_bytearr_to_np(img_byte_arr)
        #print(img_np)
        #print(img_np.shape)
        
        # SUPERPIXEL
        # snic
        labels, numlabels, labimg = get_superpixel_snic(img_np, m)
        # pixels in each label
        #print(labels.shape)
        #print(numlabels)
        dict_label_pixels = su.create_label_pixels(labels,numlabels)
        # lokasi pada tiap area (label) untuk menempatkan tulisan
        dict_label_pos = su.create_dict_label_pos(dict_label_pixels)
        # average color
        dict_label_color = su.create_dict_label_color(dict_label_pixels, labimg)
        # graph of labels
        adjacent_adaptels = su.get_adjacent_adaptels(labels,numlabels)
        
        # TRACES
        # draw foreground traces
        #fg_traces = su.create_traces_canvas(1, labels)
        #fg_traces = su.draw_trace_line(fg_traces, (240,260), (250,270))
        # draw background traces
        #bg_traces = su.create_traces_canvas(0, labels)
        #bg_traces = su.draw_trace_line(bg_traces, (10,0), (400,10))
        #bg_traces = su.draw_trace_line(bg_traces, (10,0), (20,400))
        #bg_traces = su.draw_trace_line(bg_traces, (500,500), (10,400))
        # draw traces
        #traces = [bg_traces, fg_traces]
        traces = get_traces(in_traces, labels)
        #print(traces)
        
        # LABEL CLASSIFICATION
        # classify selected labels
        dict_adaptel_classes = su.find_adaptel_class(traces, labels, dict_label_pixels)  
        # grow selection
        dict_adaptel_classes = lib_grow_selection.grow_selection(dict_adaptel_classes, adjacent_adaptels, dict_label_color)
        # get image mask	
        mask_img = su.drawMask(labels, dict_adaptel_classes, dict_label_pixels)	
        # save image mask
        #Image.fromarray(maskimg).save(mask_rslt)        
        
        # TEST
        # TEST SHOW BOUNDARIES
        img_np_with_boundaries = draw_boundaries(img_np,labels)
        img_np_boundaries = drawBoundariesOnly(img_np,labels,numlabels,dict_label_pos,True)
        
        # RESULT
        img_pil = ic.img_np_to_pil(img_np_orig)
        img_base64 = ic.img_pil_to_base64(img_pil) 
        
        img_pil_mask = ic.img_np_to_pil(mask_img)
        mask_base64 = ic.img_pil_to_base64(img_pil_mask)
        
        img_pil_superpixel = ic.img_np_to_pil(img_np_with_boundaries)
        img_base64_superpixel = ic.img_pil_to_base64(img_pil_superpixel)
        
        img_pil_boundaries = ic.img_np_to_pil(img_np_boundaries)
        img_base64_boundaries = ic.img_pil_to_base64(img_pil_boundaries)         
        
        return img_base64, mask_base64, img_base64_superpixel, img_base64_boundaries
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)    


def get_superpixel_snic(img_np, m):
    try:
        height, width, channels = img_np.shape
        # allocate memory for output returned by reg.growing C++ code
        RGRout = np.zeros((width*height), dtype=int)
        lOut = np.zeros((width*height), dtype=np.float64)
        aOut = np.zeros((width*height), dtype=np.float64)
        bOut = np.zeros((width*height), dtype=np.float64)        
        img_b = img_np[:,:,2].flatten()
        img_g = img_np[:,:,1].flatten()
        img_r = img_np[:,:,0].flatten()    
        preSeg = np.int32(np.zeros((height,width))).flatten() # not used
        num_superpixel = 800
        S, num_superpixel = get_snic_seeds(height,width,num_superpixel)
        m = 1
        
        # call RGR
        #print(type(img_r))
        #print(type(img_g))
        #print(type(img_b)) 
        #print(img_r.dtype)
        #print(img_g.dtype)
        #print(img_b.dtype)        
        #print(preSeg.astype(np.int32).shape)
        #print(S.astype(np.int32).shape)
        #print(width)
        #print(height)    
        #print(int(num_superpixel))
        #print(m)
        #print(RGRout.shape)
        out_, _, _, _ = callRGR.callRGR2(img_r.astype(np.int32), img_g.astype(np.int32), img_b.astype(np.int32), preSeg.astype(np.int32), S.astype(np.int32), width, height, int(num_superpixel), m, RGRout.astype(np.int32), lOut, aOut, bOut)
        PsiMap = np.asarray(out_)
        #print(lOut)
        #print(PsiMap.shape)
        PsiMap = np.reshape(PsiMap, (height, width), order='F')
        l = np.reshape(lOut, (height, width), order='C')
        a = np.reshape(aOut, (height, width), order='C')
        b = np.reshape(bOut, (height, width), order='C')
        #PsiMap = np.reshape(PsiMap, (width, height), order='C')
        #l = np.reshape(lOut, (width, height), order='C')
        #a = np.reshape(aOut, (width, height), order='C')
        #b = np.reshape(bOut, (width, height), order='C')        
        #lab = np.dstack((l,a,b))
        lab = (l,a,b)
        #print(lab)
        #print(lab.shape)
        #print(PsiMap)
        #print(PsiMap.shape)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno) 
        
    return PsiMap, np.amax(PsiMap)+1, lab
    
    
    
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
        ic.img_np_to_file(S, 'static/'+'dummy1'+'/superpixel_seeds'+''+'.png')
        
        S = S.flatten(order='F')
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)   
    return S, num_superpixel_actual        
    
    
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
    
def get_traces(in_traces,labels):   
    traces = []
    dict_canvas = dict()

    for trace in in_traces:
        trace_elmts = trace.split(',')
        class_id = trace_elmts[-1]
        if trace_elmts[-1] not in dict_canvas:
            # create if not exist
            canvas = su.create_traces_canvas(int(class_id), labels)
            dict_canvas[class_id] = canvas
        #if (trace_elmts[0]==trace_elmts[-4]) and (trace_elmts[1]==trace_elmts[-3]):
            # polygon
            #pass
        #else:
            # polyline
            #pass
        for i in range(0,len(trace_elmts)-5,4):
            # draw
            c0 = int(trace_elmts[i]) # i.e. x0
            r0 = int(trace_elmts[i+1]) # i.e. y0                
            c1 = int(trace_elmts[i+4])
            r1 = int(trace_elmts[i+5])            
            dict_canvas[class_id] = su.draw_trace_line(dict_canvas[class_id], (r0,c0), (r1,c1))
        
    # draw foreground traces
    #fg_traces = su.create_traces_canvas(1, labels)
    #fg_traces = su.draw_trace_line(fg_traces, (240,260), (250,270))
    # draw background traces
    #bg_traces = su.create_traces_canvas(0, labels)
    #bg_traces = su.draw_trace_line(bg_traces, (10,0), (400,10))
    #bg_traces = su.draw_trace_line(bg_traces, (10,0), (20,400))
    #bg_traces = su.draw_trace_line(bg_traces, (500,500), (10,400))
    # draw traces
    #traces = [bg_traces, fg_traces]
    
    for canvas in dict_canvas:
        traces.append(dict_canvas[canvas]);

    return traces