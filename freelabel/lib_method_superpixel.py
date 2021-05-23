import callRGR
import math
import numpy as np
import sys, os
import os.path

import lib_draw_superpixel as ds
import lib_img_convert as ic
import lib_superpixel_util as su
import lib_grow_selection
import lib_refinement
import lib_resolve_conflict

from PIL import Image
from PIL import ImageDraw
import pickle
import time


def create_superpixel(url, m, in_traces, ID, init_only=False):
    try:
        # variables
        ts0 = time.time()
        is_test = True
    
        # file name
        np_save_file_init = 'static/'+'dummy1'+'/'+ID+'-init.npz'
        np_save_file_process = 'static/'+'dummy1'+'/'+ID+'-process.npz'
        pickle_save_file_init = 'static/'+'dummy1'+'/'+ID+'-init.pkl'        
        pickle_save_file_process = 'static/'+'dummy1'+'/'+ID+'-process.pkl'        
        
        # image to process - 1
        img_byte_arr = ic.img_url_to_bytearr(url)
        #print('after img_url_to_bytearr')
        if not init_only:
            img_np_orig = ic.img_bytearr_to_np(img_byte_arr)
            ht, wd, _ = img_np_orig.shape
        
        if not init_only:            
            # TRACES
            traces, sum_traces = get_traces(in_traces, ht, wd) 
            # check traces difference                           
            if os.path.isfile(np_save_file_process):
                
                npzfile_process = np.load(np_save_file_process)
                mask_img_prev = npzfile_process['mask_img']
                sum_traces_prev = npzfile_process['sum_traces']
                sum_traces_diff = np.subtract(sum_traces_prev, sum_traces)
                count_nonzero_sum_traces_diff = np.count_nonzero(sum_traces_diff)
                #print(count_nonzero_sum_traces_diff)
                if count_nonzero_sum_traces_diff == 0:
                    img_pil_mask = ic.img_np_to_pil(mask_img_prev)
                    mask_base64 = ic.img_pil_to_base64(img_pil_mask)                
                    ts1 = time.time()
                    time_diff = ts1 - ts0
                    print(time_diff)
                    return mask_base64, mask_base64, mask_base64, mask_base64, mask_base64, time_diff
    
        #print('create_superpixel')

        # image to process - 2
        img_np = ic.img_bytearr_to_np(img_byte_arr)
        #print(img_np)
        #print(img_np.shape)
        
        # SUPERPIXEL
        is_save = True
        if os.path.isfile(np_save_file_init):
        #if False:
            is_save = False
            # restore from file
            # restore numpy: labels
            npzfile = np.load(np_save_file_init)
            labels = npzfile['labels']
            # restore others: numlabels, dict_label_pos, dict_label_color, adjacent_adaptels
            with open(pickle_save_file_init, 'rb') as f:  # Python 3: open(..., 'rb')
                dict_label_pixels, numlabels, dict_label_pos, dict_label_color, adjacent_adaptels = pickle.load(f)
            if is_test:
                if not dict_label_pos:
                    dict_label_pos = su.create_dict_label_pos(dict_label_pixels)
                    is_save = True
            #print(labels)
            #print(dict_label_pixels)
        else:
            # snic
            labels, numlabels, labimg, dict_centroid_center = get_superpixel_snic(img_np, m)
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
            if is_test:
                dict_label_pos = su.create_dict_label_pos(dict_label_pixels)
            else:
                dict_label_pos = {}
            # average color
            dict_label_color = su.create_dict_label_color(dict_label_pixels, labimg)
            # graph of labels
            adjacent_adaptels = su.get_adjacent_adaptels(labels,numlabels)

        #if False:         
        if is_save:
            # save
            # save numpy: labels
            np.savez(np_save_file_init, labels=labels)
            # save others: numlabels, dict_label_pos, dict_label_color, adjacent_adaptels
            with open(pickle_save_file_init, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([dict_label_pixels, numlabels, dict_label_pos, dict_label_color, adjacent_adaptels], f)
                 
        if not init_only:
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
            #traces, sum_traces = get_traces(in_traces, labels)
            #print(traces)
            # test draw sum_traces
            #ic.img_np_to_file(sum_traces * 80, 'static/'+'dummy1'+'/sum_traces'+''+'.png')
            
            # LABEL CLASSIFICATION
            # classify selected labels
            dict_adaptel_classes_init = su.find_adaptel_class(traces, labels, dict_label_pixels)  
            # grow selection
            dict_adaptel_classes_temp, conflicting_labels, need_refinement_labels = lib_grow_selection.grow_selection(dict_adaptel_classes_init, adjacent_adaptels, dict_label_color)
            #print(need_refinement_labels)
            #print(conflicting_labels)
            # get image mask	
            mask_img = su.drawMask(labels, dict_adaptel_classes_temp, dict_label_pixels)	
            #print(conflicting_labels)
            # RESOLVE CONFLICT
            if len(conflicting_labels) > 0:
                # resolve conflict: get superpixel
                dict_class_indexes = get_superpixel_snic_for_conflicting_labels(img_np_orig, m, conflicting_labels, dict_label_pixels, traces)
                #print(dict_class_indexes)
                # draw image mask for conflicting centroids
                mask_img = su.drawMaskAdd(dict_class_indexes, mask_img)
                #dict_adaptel_classes_final = lib_grow_selection.resolve_selection_conflict(dict_adaptel_classes_temp, conflicting_labels, traces)        
            # save image mask
            #Image.fromarray(maskimg).save(mask_rslt)
            
            # REFINE
            if len(need_refinement_labels) > 0:
                dict_class_indexes_refine = get_superpixel_snic_for_refinement(img_np_orig, m, need_refinement_labels, dict_label_pixels, traces)
                #mask_img = su.drawMaskAdd(dict_class_indexes_refine, mask_img)
            
            is_save_2 = True              
            #if is_save:
            if is_save_2:
                #pass
                #for trace in traces:
                    #pickle.dump([traces], f)
                
                # save
                # save numpy: mask_img
                np.savez(np_save_file_process, mask_img=mask_img, sum_traces=sum_traces)
                # save others: traces
                #with open(pickle_save_file_process, 'wb') as f:  # Python 3: open(..., 'wb')
                #    pass

            # TEST        
            if is_test:
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
            
            if is_test:
                img_pil_boundaries = ic.img_np_to_pil(img_np_with_boundaries)
                img_base64_boundaries = ic.img_pil_to_base64(img_pil_boundaries)
                
                img_pil_labels = ic.img_np_to_pil(img_np_labels)
                img_base64_labels = ic.img_pil_to_base64(img_pil_labels)         
         
                img_pil_superpixel = ic.img_np_to_pil(img_np_sp_color)
                img_base64_superpixel = ic.img_pil_to_base64(img_pil_superpixel)
            else:
                img_base64_boundaries = img_base64_labels = img_base64_superpixel = mask_base64 
      
            ts1 = time.time()
            time_diff = ts1 - ts0
            return img_base64, mask_base64, img_base64_boundaries, img_base64_labels, img_base64_superpixel, time_diff
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
        #num_superpixel = 40  # test
        #print('num seed',num_superpixel)
        S, num_superpixel = get_snic_seeds(height,width,num_superpixel)
        m = 1
        #m = 10  # test
        
        dict_centroid_center = get_dict_centroid_center(S,height,width)
        
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
        #print(RGRout)
        out_, _, _, _ = callRGR.callRGR2(img_r.astype(np.int32), img_g.astype(np.int32), img_b.astype(np.int32), preSeg.astype(np.int32), S.astype(np.int32), width, height, int(num_superpixel), m, RGRout.astype(np.int32), lOut, aOut, bOut)
        #print(preSeg)
        #print(RGRout)
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
        
    return PsiMap, np.amax(PsiMap)+1, lab, dict_centroid_center


def get_superpixel_snic_for_refinement(img_np, m, need_refinement_labels, dict_label_pixels, traces):
    try:
        #labels_, numlabels_, labimg_ = None
        dict_class_indexes = {}
        
        is_per_label = True
        if is_per_label:
            for need_refinement_label in need_refinement_labels:
                dict_class_indexes_tmp = lib_refinement.get_refined_dict_class_indexes(img_np, m, [need_refinement_label], dict_label_pixels, traces)
                #print(dict_class_indexes_tmp)                
                #print('')
                for key in dict_class_indexes_tmp:
                    #print(type(dict_class_indexes_tmp[key][0]))
                    if key not in dict_class_indexes:
                        dict_class_indexes[key] = dict_class_indexes_tmp[key]
                    else:
                        #dict_class_indexes[key][0].append(dict_class_indexes_tmp[key][0])
                        #dict_class_indexes[key][1].append(dict_class_indexes_tmp[key][1])
                        tup_y = np.append(dict_class_indexes[key][0], dict_class_indexes_tmp[key][0])
                        tup_x = np.append(dict_class_indexes[key][1], dict_class_indexes_tmp[key][1])
                        dict_class_indexes[key] = (tup_y, tup_x)                        
        else:
            dict_class_indexes = lib_refinement.get_refined_dict_class_indexes(img_np, m, need_refinement_labels, dict_label_pixels, traces)
            """
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
            """
            
        return dict_class_indexes
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)     

    
def get_superpixel_snic_for_conflicting_labels(img_np, m, conflicting_labels, dict_label_pixels, traces):
    try:
        dict_class_indexes = {}
        
        is_per_label = True
        if is_per_label:
            for idx, conflicting_label in enumerate(conflicting_labels):
                dict_class_indexes_tmp = lib_resolve_conflict.get_resolved_dict_class_indexes(img_np, m, [conflicting_label], dict_label_pixels, traces)
                #print(dict_class_indexes_tmp)                
                #print('')
                #dict_class_indexes = dict_class_indexes_tmp
                #if idx == 1:
                #    break
                #break
                for key in dict_class_indexes_tmp:
                    #if idx == 2:
                    #if idx in [1]:
                    if True:
                        #print(type(dict_class_indexes_tmp[key][0]))
                        #print(idx, dict_class_indexes_tmp[key][0].shape)
                        if key not in dict_class_indexes:
                            #print(idx, key, 'if')
                            dict_class_indexes[key] = dict_class_indexes_tmp[key]
                        else:
                            #print(idx, key, 'else')
                            tup_y = np.append(dict_class_indexes[key][0], dict_class_indexes_tmp[key][0])
                            tup_x = np.append(dict_class_indexes[key][1], dict_class_indexes_tmp[key][1])
                            dict_class_indexes[key] = (tup_y, tup_x)
                            #dict_class_indexes[key][0].append(dict_class_indexes_tmp[key][0])
                            #dict_class_indexes[key][1].append(dict_class_indexes_tmp[key][1])
                        #break
            #print(dict_class_indexes[1][0].shape)
            #print(dict_class_indexes[2][0].shape)
        else:
            dict_class_indexes = lib_resolve_conflict.get_resolved_dict_class_indexes(img_np, m, conflicting_labels, dict_label_pixels, traces)    
        
        """
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
        #print(np.where(preSeg == 2))
        #print(preSeg.astype(np.int32))
        label_out_, class_out_ = callRGR.callRGR3(img_r.astype(np.int32), img_g.astype(np.int32), img_b.astype(np.int32), preSeg.astype(np.int32), S.astype(np.int32), width, height, int(num_superpixel), m, RGRout.astype(np.int32))
        class_out = np.asarray(class_out_)
        class_out = np.reshape(class_out, (height, width), order='C')
        #print(class_out)
        #print(np.where(class_out == 2))
    
        dict_class_indexes = {}
        for trace in traces:        
            class_id = trace['class_id']
            canvas = trace['canvas']
            dict_class_indexes[class_id] = np.where(class_out == class_id)
            #print(class_id)
            #print(dict_class_indexes[class_id])
        """
        
        return dict_class_indexes
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)     


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
        
        #dict_centroid_center = {}
        #indices = np.where(S == 255)
        #print(indices)
        #for key,index in enumerate(indices):
        #    dict_centroid_center[key] = index        
        
        S = S.flatten(order='F')
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)   
    return S, num_superpixel_actual       
    
    
def get_traces(in_traces,height,width):   
    traces = []
    dict_canvas = dict()

    #ht,wd = labels.shape
    sum_traces = np.zeros((height, width), dtype=np.float64)

    for trace in in_traces:
        trace_elmts = trace.split(',')
        class_id = trace_elmts[-1]
        if trace_elmts[-1] not in dict_canvas:
            # create if not exist
            canvas = su.create_traces_canvas(int(class_id), height, width)
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
            sum_traces = su.draw_trace_line(sum_traces, (r0,c0), (r1,c1), class_id)
        
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

    return traces, sum_traces
    
    
def get_dict_centroid_center(S,height,width):
    dict_centroid_center = {}
    indices = np.where(S == 255)
    #print(indices)
    tup_coordinates = np.unravel_index(indices, (height,width), order='F')
    #print(tup_coordinates)
    ravelled = np.ravel(tup_coordinates, order='F')
    #print(ravelled)
    #print(len(ravelled)/2)
    reshaped = ravelled.reshape((int(len(ravelled)/2), 2))
    #print(reshaped)
    for key,index in enumerate(reshaped):
        dict_centroid_center[key] = index
    #print(dict_centroid_center)
    return dict_centroid_center