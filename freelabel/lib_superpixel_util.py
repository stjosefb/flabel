import numpy as np
import sys, os
import math
from skimage.draw import line  # pip install scikit-image

from PIL import Image


# # BEGIN - MASK
DICT_CLASS_COLOR = {
    1: (255,255,255,255), 
    2: (200,0,0,128),
}	


def drawMask(labels, dict_adaptel_classes, dict_label_pixels):
	ht,wd = labels.shape
	maskimg = np.zeros((ht, wd, 4), dtype=np.uint8)
	
	for label, _class in dict_adaptel_classes.items():
		if _class is not None:
			idx = tuple(zip(*dict_label_pixels[label]))
			maskimg[idx] = DICT_CLASS_COLOR[_class]
	
	return maskimg
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
	dict_label_pos = {}
	for label, pixels in dict_label_pixels.items():
		dict_label_pos[label] = pixels[int(math.floor(len(pixels)/2))]
	return dict_label_pos        
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
	if x < (wd - 1):						
		if labels[y,x] != labels[y,x+1]:
			adjacent_pixel_labels.add(labels[y,x+1])
	if x > 0:			
		if labels[y,x] != labels[y,x-1]:
			adjacent_pixel_labels.add(labels[y,x-1])
	if y < (ht - 1):						
		if labels[y,x] != labels[y+1,x]:
			adjacent_pixel_labels.add(labels[y+1,x])
	if y > 0:			
		if labels[y,x] != labels[y-1,x]:
			adjacent_pixel_labels.add(labels[y-1,x])
	return adjacent_pixel_labels
# # END - ADAPTELS ADJACENCY
    