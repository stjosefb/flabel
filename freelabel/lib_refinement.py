
import sys, os

import numpy as np

import callRGR


def get_refined_dict_class_indexes(img_np, m, need_refinement_labels, dict_label_pixels, traces):
    dict_class_indexes = {}
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
        m = 1        
        #preSeg = np.int32(np.zeros((height,width))).flatten() # not used
        
        #num_superpixel = 800
        #num_superpixel = int(width*height/327.5)
        #print('num seed',num_superpixel)
        #S, num_superpixel = get_snic_seeds(height,width,num_superpixel)
        
        S, num_superpixel, preSeg = get_snic_seeds_for_refinement(height,width,need_refinement_labels, dict_label_pixels, traces)
        
        #print(num_superpixel)        
        #print(preSeg)
        #print(np.where(preSeg == 2))
        #print(preSeg.astype(np.int32))
        label_out_, class_out_ = callRGR.callRGR3(img_r.astype(np.int32), img_g.astype(np.int32), img_b.astype(np.int32), preSeg.astype(np.int32), S.astype(np.int32), width, height, int(num_superpixel), m, RGRout.astype(np.int32))
        class_out = np.asarray(class_out_)
        class_out = np.reshape(class_out, (height, width), order='C')
        #print(class_out)
        #print(np.where(class_out == 2))
            
        for trace in traces:        
            class_id = trace['class_id']
            #canvas = trace['canvas']
            dict_class_indexes[class_id] = np.where(class_out == class_id)
            #print(class_id)
            #print(dict_class_indexes[class_id])    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno) 
        
    return dict_class_indexes
    
    
def get_snic_seeds_for_refinement(height,width,need_refinement_labels, dict_label_pixels, traces):
    # init with -1 (centroids will not expand to these pixels)
    S = np.full((height, width), -1)
    #S = np.full((height, width), 255)  # test
    canvas_refinement = np.full((height, width), 0)
    
    # set conflicting areas as 0 (centroids will expand to these pixels)
    for dict_refinement in need_refinement_labels:
        #print(conflicting_label)
        for pixel in dict_label_pixels[dict_refinement['label']]:
            h,w = pixel
            S[h,w] = 0
            #S[h,w] = 128  # test
            canvas_refinement[h,w] = 1
    
    # set traces on conflicting areas based on class id (centroid will expand from these pixels)
    for trace in traces:        
        class_id = trace['class_id']
        canvas = trace['canvas']
        
        canvas1 = np.array(canvas_refinement, dtype=bool)
        canvas2 = canvas.astype(bool)
        canvas_intersect = np.logical_and(canvas1, canvas2)
        idx_intersect = np.where(canvas_intersect == True)
        S[idx_intersect] = class_id
        #S[idx_intersect] = class_id * 80  # test
        
    # set first pixel adjacent to nearest_label centroid as seed
    for pixel in dict_label_pixels[dict_refinement['label']]:
        h,w = pixel
        pixels = []
        if h < height - 1:
            pixel_adj = h+1, w
            pixels.append(pixel_adj)
        if w < width - 1:
            pixel_adj = h, w+1 
            pixels.append(pixel_adj)            
        if h > 0:
            pixel_adj = h-1, w
            pixels.append(pixel_adj)
        if w > 0:
            pixel_adj = h, w-1     
            pixels.append(pixel_adj)
        for adjacent_pixel in pixels:
            if adjacent_pixel in dict_label_pixels[dict_refinement['nearest_label']]:
                S[adjacent_pixel] = dict_refinement['class_candidate']
                break
    
    #ic.img_np_to_file(S, 'static/'+'dummy1'+'/superpixel_seeds_conflict'+''+'.png')
    
    # num_superpixel and preSeg
    idx_traces = np.where( S > 0 )
    num_superpixel = len(idx_traces[0])
    preSeg = np.copy(S).flatten()
    S = S.flatten(order='F')
    
    return S, num_superpixel, preSeg
    