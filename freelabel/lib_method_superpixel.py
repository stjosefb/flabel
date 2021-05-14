import callRGR
import math
import numpy as np
import sys, os

import lib_draw_superpixel as ds
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
        if False:
            # restore from file
            # ...
            pass
        else:
            # snic
            labels, numlabels, labimg = get_superpixel_snic(img_np, m)
            # pixels in each label
            #print(labels.shape)
            #print(numlabels)
            dict_label_pixels = su.create_label_pixels(labels,numlabels)
            #print(dict_label_pixels)
            # labels validation
            #if True:
            if False: 
                labels, numlabels = su.validate_labels(labels,numlabels,dict_label_pixels)
                dict_label_pixels = su.create_label_pixels(labels,numlabels)	
            # lokasi pada tiap area (label) untuk menempatkan tulisan
            dict_label_pos = su.create_dict_label_pos(dict_label_pixels)
            # average color
            dict_label_color = su.create_dict_label_color(dict_label_pixels, labimg)
            # graph of labels
            adjacent_adaptels = su.get_adjacent_adaptels(labels,numlabels)
            # save
            # ...
        
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
        dict_adaptel_classes_init = su.find_adaptel_class(traces, labels, dict_label_pixels)  
        # grow selection
        dict_adaptel_classes_temp, conflicting_labels = lib_grow_selection.grow_selection(dict_adaptel_classes_init, adjacent_adaptels, dict_label_color)
        #print(conflicting_labels)
        # get image mask	
        mask_img = su.drawMask(labels, dict_adaptel_classes_temp, dict_label_pixels)	
        if len(conflicting_labels) > 0:
            # resolve conflict: get superpixel
            dict_class_pixels = get_superpixel_snic_for_conflicting_labels(img_np_orig, m, conflicting_labels, dict_label_pixels, traces)
            # draw image mask for conflicting centroids
            #mask_img = su.drawMaskConflictingLabels(dict_class_pixels, mask_img)
            #dict_adaptel_classes_final = lib_grow_selection.resolve_selection_conflict(dict_adaptel_classes_temp, conflicting_labels, traces)        
        # save image mask
        #Image.fromarray(maskimg).save(mask_rslt)        
        
        # TEST
        # TEST SHOW BOUNDARIES
        img_np_with_boundaries = ds.draw_boundaries(img_np,labels)
        img_np_labels = ds.drawBoundariesOnly(img_np,labels,numlabels,dict_label_pos,True)
        # test superpixels color
        dict_label_color_rgb = ds.dictLabelLabToRgb(dict_label_color)
        #print(dict_label_color)
        #print(dict_label_color)
        #print(dict_label_color_rgb)
        img_np_sp_color = ds.draw_superpixels(img_np,labels,dict_label_color_rgb)
        
        # RESULT
        img_pil = ic.img_np_to_pil(img_np_orig)
        img_base64 = ic.img_pil_to_base64(img_pil) 
        
        img_pil_mask = ic.img_np_to_pil(mask_img)
        mask_base64 = ic.img_pil_to_base64(img_pil_mask)
        
        img_pil_boundaries = ic.img_np_to_pil(img_np_with_boundaries)
        img_base64_boundaries = ic.img_pil_to_base64(img_pil_boundaries)
        
        img_pil_labels = ic.img_np_to_pil(img_np_labels)
        img_base64_labels = ic.img_pil_to_base64(img_pil_labels)         
 
        img_pil_superpixel = ic.img_np_to_pil(img_np_sp_color)
        img_base64_superpixel = ic.img_pil_to_base64(img_pil_superpixel)
  
        return img_base64, mask_base64, img_base64_boundaries, img_base64_labels, img_base64_superpixel
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
        #num_superpixel = 800
        num_superpixel = int(width*height/327.5)
        #print('num seed',num_superpixel)
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
        #print(preSeg)
        print(RGRout)
        out_, _, _, _ = callRGR.callRGR2(img_r.astype(np.int32), img_g.astype(np.int32), img_b.astype(np.int32), preSeg.astype(np.int32), S.astype(np.int32), width, height, int(num_superpixel), m, RGRout.astype(np.int32), lOut, aOut, bOut)
        #print(preSeg)
        print(RGRout)
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
    
    
def get_superpixel_snic_for_conflicting_labels(img_np, m, conflicting_labels, dict_label_pixels, traces):
    try:
        #labels_, numlabels_, labimg_ = None
        
        height, width, channels = img_np.shape
        # allocate memory for output returned by reg.growing C++ code
        RGRout = np.zeros((width*height), dtype=int)
        lOut = np.zeros((width*height), dtype=np.float64)
        aOut = np.zeros((width*height), dtype=np.float64)
        bOut = np.zeros((width*height), dtype=np.float64)        
        img_b = img_np[:,:,2].flatten()
        img_g = img_np[:,:,1].flatten()
        img_r = img_np[:,:,0].flatten()    
        m = 1        
        #preSeg = np.int32(np.zeros((height,width))).flatten() # not used
        
        #num_superpixel = 800
        #num_superpixel = int(width*height/327.5)
        #print('num seed',num_superpixel)
        #S, num_superpixel = get_snic_seeds(height,width,num_superpixel)
        
        S, num_superpixel, preSeg = get_snic_seeds_for_conflicting_labels(height,width,conflicting_labels, dict_label_pixels, traces)
        
        #print(num_superpixel)        
        #print(preSeg)
        #print(preSeg.astype(np.int32))
        out_, _, _, _ = callRGR.callRGR2(img_r.astype(np.int32), img_g.astype(np.int32), img_b.astype(np.int32), preSeg.astype(np.int32), S.astype(np.int32), width, height, int(num_superpixel), m, RGRout.astype(np.int32), lOut, aOut, bOut)
        #print(preSeg)
    
        dict_class_pixels = {}
        
        return dict_class_pixels
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)     


def get_snic_seeds_for_conflicting_labels(height,width,conflicting_labels, dict_label_pixels, traces):
    # init with -1 (centroids will not expand to these pixels)
    S = np.full((height, width), -1)
    #S = np.full((height, width), 255)  # test
    canvas_conflicting_labels = np.full((height, width), 0)
    
    # set conflicting areas as 0 (centroids will expand to these pixels)
    for conflicting_label in conflicting_labels:
        #print(conflicting_label)
        for pixel in dict_label_pixels[conflicting_label]:
            h,w = pixel
            S[h,w] = 0
            #S[h,w] = 128  # test
            canvas_conflicting_labels[h,w] = 1
    
    # set traces on conflicting areas based on class id (centroid will expand from these pixels)
    for trace in traces:        
        class_id = trace['class_id']
        canvas = trace['canvas']
        
        canvas1 = np.array(canvas_conflicting_labels, dtype=bool)
        canvas2 = canvas.astype(bool)
        canvas_intersect = np.logical_and(canvas1, canvas2)
        idx_intersect = np.where(canvas_intersect == True)
        S[idx_intersect] = class_id
        #S[idx_intersect] = class_id * 80  # test
    
    #ic.img_np_to_file(S, 'static/'+'dummy1'+'/superpixel_seeds_conflict'+''+'.png')
    
    # num_superpixel and preSeg
    idx_traces = np.where( S > 0 )
    num_superpixel = len(idx_traces[0])
    preSeg = np.copy(S).flatten()
    S = S.flatten(order='F')
    
    return S, num_superpixel, preSeg


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
        while (y < height) and (n < num_superpixel_actual):
            x = halfstep
            while (x < width) and (n < num_superpixel_actual):
                if (y <= h-halfstep) and (x <= w-halfstep):
                    #print(y,x)
                    S[int(y),int(x)] = 255  # anything greater than 0
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