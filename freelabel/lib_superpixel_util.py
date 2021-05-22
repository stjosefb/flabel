import numpy as np
import sys, os
import math
from skimage.draw import line  # pip install scikit-image

from PIL import Image

import lib_img_convert as ic


# # BEGIN - MASK
DICT_CLASS_COLOR = {
    1: (255,255,255,0), 
    2: (200,0,0,128),
}    


def drawMask(labels, dict_adaptel_classes, dict_label_pixels):
    try:
        ht,wd = labels.shape
        maskimg = np.zeros((ht, wd, 4), dtype=np.uint8)
        
        for label, _classes in dict_adaptel_classes.items():
            #if _class is not None:
            if len(_classes) == 1:
                idx = tuple(zip(*dict_label_pixels[label]))
                maskimg[idx] = DICT_CLASS_COLOR[_classes[0]]
        
        return maskimg
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno) 


def drawMaskAdd(dict_class_indexes, mask_img):
    try:
        ht,wd,_ = mask_img.shape
        
        for _class, indexes in dict_class_indexes.items():
            #if _class == 3:
            #print(indexes)
            mask_img[indexes] = DICT_CLASS_COLOR[_class]
        
        #ic.img_np_to_file(mask_img, 'static/'+'dummy1'+'/superpixel_conflict'+''+'.png')
        
        return mask_img
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno) 
# # END - MASK    
        
        
# # BEGIN - ADAPTELS SELECTION    
def simplify_adaptel_class(dict_adaptel_classes):
    dict_adaptel_classes_2 = {}
    for label, classes in dict_adaptel_classes.items():
        if len(classes) > 0:
            dict_adaptel_classes_2[label] = classes[0]['class_id']
        else:
            dict_adaptel_classes_2[label] = None
    
    return dict_adaptel_classes_2
    
    
def find_adaptel_class(traces, labels, dict_label_pixels):
    dict_adaptel_classes = {}
    
    for label, pixels in dict_label_pixels.items():
        dict_adaptel_classes[label] = []
    
    for trace in traces:
        unique_labels, count_unique_labels = select_adaptels(trace, labels)
        #print(unique_labels)
        for key, unique_label in enumerate(unique_labels):
            dict_adaptel_classes = append_adaptel_class(dict_adaptel_classes, unique_label, trace['class_id'], count_unique_labels[key])            
            
    return dict_adaptel_classes
    

def append_adaptel_class(dict_adaptel_classes, label, class_id, count_trace_pixels):
    is_inserted = False
    for index, value in enumerate(dict_adaptel_classes[label]):
        if value['count_trace_pixels'] < count_trace_pixels:
            dict_adaptel_classes[label].insert(index, {'class_id': class_id, 'count_trace_pixels': count_trace_pixels})
            is_inserted = True
            break    
    if not is_inserted: 
        dict_adaptel_classes[label].append({'class_id': class_id, 'count_trace_pixels': count_trace_pixels})
        
    return dict_adaptel_classes
    
    
def select_adaptels(trace, labels):
    #set_selected_labels = set()
    #pixel_counts = []
    intersect = trace['canvas'] * labels
    list_labels = intersect[intersect > 0]
    unique_labels, count_unique_labels = np.unique(list_labels, return_counts=True)

    return [int(l) for l in unique_labels], count_unique_labels
# # END - ADAPTELS SELECTION
        
        
# # BEGIN - TRACES    
def create_traces_canvas(class_id, labels):
    ht,wd = labels.shape
    dict_canvas = {'class_id': class_id}
    dict_canvas['canvas'] = np.zeros((ht, wd), dtype=np.float64)

    return dict_canvas

    
def draw_trace_line(fg_traces, begin, end):
    try:
        y1,x1 = begin
        y2,x2 = end
        #print(y1, x1, y2, x2)
        rr, cc = line(y1, x1, y2, x2)
        fg_traces['canvas'][rr, cc] = 1
        
        return fg_traces
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)      
# # END - TRACES
        
        
# # BEGIN - ADAPTEL COLOR    
def create_dict_label_color(dict_label_pixels, labimg):
    try:
        dict_label_color = {}
        l_img, a_img, b_img = labimg    
        for label, pixels in dict_label_pixels.items():        
            l_sum = a_sum = b_sum = 0
            for pixel in pixels:
                y, x = pixel
                l_sum += l_img[y,x]
                a_sum += a_img[y,x]
                b_sum += b_img[y,x]
            len_pixels = len(pixels)
            l_avg = l_sum/len_pixels
            a_avg = a_sum/len_pixels
            b_avg = b_sum/len_pixels
                
            dict_label_color[label] = (l_avg, a_avg, b_avg)

        return dict_label_color
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)       
# # END - ADAPTEL COLOR    

    
# # BEGIN - ADAPTELS PIXELS    
def create_label_pixels(labels,numlabels):
    try:
        label_pixels = dict()
        for i in range(numlabels):
            label_pixels[i] = []
        ht,wd = labels.shape
        for y in range(0,ht):
            for x in range(0,wd):
                #print(x, y)
                #print(labels[y,x])  # 799
                label_pixels[labels[y,x]].append((y,x))
        return label_pixels        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)   


def create_dict_label_pos(dict_label_pixels):
    try:
        dict_label_pos = {}
        #print(dict_label_pixels)
        for label, pixels in dict_label_pixels.items():
            dict_label_pos[label] = pixels[int(math.floor(len(pixels)/2))]
        return dict_label_pos  
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)       
# # END - ADAPTELS PIXELS    


# # BEGIN - ADAPTELS ADJACENCY
def get_adjacent_adaptels(labels,numlabels):
    adjacent_adaptels = dict()
    ht,wd = labels.shape

    for y in range(0,ht):
        for x in range(0,wd):
            adjacent_pixel_labels = get_adjacent_pixel_labels(y,x,labels)
            for label in adjacent_pixel_labels:
                if labels[y,x] != label:
                    if label not in adjacent_adaptels:
                        adjacent_adaptels[label] = set()
                    adjacent_adaptels[label].add(labels[y,x])

    return adjacent_adaptels

    
def get_adjacent_pixel_labels(y,x,labels):
    adjacent_pixel_labels = set()
    ht,wd = labels.shape            
    if x < (wd - 1):  # right                       
        if labels[y,x] != labels[y,x+1]:
            adjacent_pixel_labels.add(labels[y,x+1])
    if x > 0:  # left            
        if labels[y,x] != labels[y,x-1]:
            adjacent_pixel_labels.add(labels[y,x-1])
    if y < (ht - 1):  # down                       
        if labels[y,x] != labels[y+1,x]:
            adjacent_pixel_labels.add(labels[y+1,x])
    if y > 0:  # up            
        if labels[y,x] != labels[y-1,x]:
            adjacent_pixel_labels.add(labels[y-1,x])
            
    if x < (wd - 1) and y > 0:  # up right                       
        if labels[y,x] != labels[y-1,x+1]:
            adjacent_pixel_labels.add(labels[y-1,x+1])
    if x > 0 and y > 0:  # up left          
        if labels[y,x] != labels[y-1,x-1]:
            adjacent_pixel_labels.add(labels[y-1,x-1])
    if y < (ht - 1) and x < (wd - 1):  # down right                
        if labels[y,x] != labels[y+1,x+1]:
            adjacent_pixel_labels.add(labels[y+1,x+1])
    if y > 0 and x > 0:  # down left
        if labels[y,x] != labels[y-1,x-1]:
            adjacent_pixel_labels.add(labels[y-1,x-1])
            
    return adjacent_pixel_labels
# # END - ADAPTELS ADJACENCY
    
    
# # BEGIN - ADAPTELS VALIDATION    
def validate_labels(labels,numlabels,dict_label_pixels):
    validated_labels = labels
    num_validated_labels = numlabels
    
    for label, pixels in dict_label_pixels.items():
        # debug: check only adaptels 1
        #if label == 1:
        if True:
        #if False: # feature not active, takes too much time to check for all labels
            list_separated_pixels = validate_label(labels,numlabels,label,pixels)
            if len(list_separated_pixels) > 0:
                #num_validated_labels += len(list_separated_pixels)
                validated_labels, num_validated_labels = update_labels(list_separated_pixels, validated_labels, num_validated_labels)
    
    return validated_labels,num_validated_labels
    
    
def update_labels(list_separated_pixels, validated_labels, num_validated_labels):
    for pixels in list_separated_pixels:
        for pixel in pixels:
            y,x = pixel
            validated_labels[y,x] = num_validated_labels
        num_validated_labels += 1
        
    return validated_labels, num_validated_labels

    
def validate_label(labels,numlabels,label,pixels):
    list_separated_pixels = []

    list_pixels_done = []
    list_pixels_all = pixels
    list_pixels_processing = []        

    while len(list_pixels_all) > 0:
        separated_pixels = []
        
        # set pixel to process
        pixel_processed = list_pixels_all.pop(0)
        list_pixels_processing.append(pixel_processed)        
        while True:    
            # set pixel to process
            pixel_processed = list_pixels_processing.pop(0)            

            # pixel processed
            list_pixels_done.append(pixel_processed)
            separated_pixels.append(pixel_processed)
            #list_pixels_processing.remove(pixel_processed)
            
            # determine pixel candidates to process
            list_adjacent_pixels = get_adjacent_pixels_with_same_label(labels, label, pixel_processed)
            for adjacent_pixel in list_adjacent_pixels:
                if adjacent_pixel not in list_pixels_done and adjacent_pixel not in list_pixels_processing:
                    list_pixels_processing.append(adjacent_pixel)
                if adjacent_pixel in list_pixels_all:
                    list_pixels_all.remove(adjacent_pixel)
                        
            if len(list_pixels_processing) == 0:
                break        
            
        if len(list_pixels_all) > 0:
            list_separated_pixels.append(separated_pixels)
    
    return list_separated_pixels
    
    
def get_adjacent_pixels_with_same_label(labels, label, pixel_processed):
    list_adjacent_pixels = []
    
    ht,wd = labels.shape
    y,x = pixel_processed
    if y > 0:
        if labels[y-1,x] == label:
            list_adjacent_pixels.append((y-1,x))
    if y < (ht - 1):            
        if labels[y+1,x] == label:
            list_adjacent_pixels.append((y+1,x))
    if x > 0:            
        if labels[y,x-1] == label:
            list_adjacent_pixels.append((y,x-1))
    if x < (wd - 1):                        
        if labels[y,x+1] == label:
            list_adjacent_pixels.append((y,x+1))        
    return list_adjacent_pixels

# # END - ADAPTELS VALIDATION
    