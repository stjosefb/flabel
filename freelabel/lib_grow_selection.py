import heapq
import sys, os


class Queue(object):
    def __init__(self):
        self.heap = []
        self._sub_idx = 0

    def add(self, priority, value):
        heapq.heappush(self.heap, (priority, self._sub_idx, value))
        self._sub_idx += 1

    def is_empty(self):
        return len(self.heap) == 0

    def pop_value(self):
        return heapq.heappop(self.heap)[2]

    def pop(self):
        return heapq.heappop(self.heap)

    def length(self):
        return len(self.heap)
                
                
def norm_nd_sqr_arr(a, b):
    def sub_sqr(x, y):
        d = x - y
        return d * d
    return sum(map(sub_sqr, a, b))                
                
                
# # BEGIN - GROW TRACES
def grow_selection(dict_adaptel_classes, adjacent_adaptels, dict_label_color):
    try:
        queue = Queue()
        q_add = queue.add  # cache some functions
        q_pop = queue.pop
        q_empty = queue.is_empty
            
        dict_adaptel_class_classified = {}
        need_refinement_labels = []
        dict_most_adjacent_label = {}
        
        # adaptels with known classes 
        for label, list_class_info in dict_adaptel_classes.items():
            #if label not in dict_adaptel_class_classified:
                #dict_adaptel_class_classified[label] = []
            #if len(list_class_info) > 0:
            #    dict_adaptel_class_classified[label] = list_class_info[0]['class_id']
            for class_info in list_class_info:
                #if label not in dict_adaptel_class_classified:
                #    dict_adaptel_class_classified[label] = []        
                #dict_adaptel_class_classified[label].append(class_info['class_id'])
                dict_adaptel_class_classified.setdefault(label, []).append(class_info['class_id'])
                
        #print(dict_adaptel_classes)
        #print(dict_adaptel_class_classified)
        #print(dict_label_color)
        
        # init priority queue based on adaptels with known classes
        level = 1
        for label_ref in dict_adaptel_class_classified:   
            for label in adjacent_adaptels[label_ref]:
                #print(label_ref)
                #if label not in dict_adaptel_class_classified:
                if True:
                    #print(label_ref)
                    color_label = dict_label_color[label]
                    color_label_ref = dict_label_color[label_ref]
                    color_diff = norm_nd_sqr_arr(color_label, color_label_ref)
                    for potential_class in dict_adaptel_class_classified[label_ref]:
                        #print(label_ref, potential_class)       
                        q_add(color_diff, [label_ref, label, potential_class, level])
                    #q_add(color_diff, [label_ref, label, dict_adaptel_class_classified[label_ref]])
                    #print(label_ref, label, dict_adaptel_class_classified[label_ref], color_diff)
                    
        #print(dict_adaptel_class_classified.keys())
        
        # process queue
        while not q_empty():
            item = q_pop()
            value = item[2]
            source_label = value[0]
            current_label = value[1]
            class_ = value[2]
            level = value[3]        
            if current_label not in dict_adaptel_class_classified:
                #dict_adaptel_class_classified[value[1]] = value[2]
                #dict_adaptel_class_classified[value[1]].append(value[2])
                dict_adaptel_class_classified.setdefault(current_label, []).append(class_)
                label_ref = current_label
                for label in adjacent_adaptels[current_label]:
                    if label not in dict_adaptel_class_classified:
                        color_label = dict_label_color[label]
                        color_label_ref = dict_label_color[label_ref]
                        color_diff = norm_nd_sqr_arr(color_label, color_label_ref)
                        for potential_class in dict_adaptel_class_classified[label_ref]:
                            q_add(color_diff, [label_ref, label, potential_class, level+1])                    
                        #q_add(color_diff, [label_ref, label, dict_adaptel_class_classified[label_ref]])            
            else:
                pass

            if level == 1:
                if source_label not in dict_most_adjacent_label:
                    dict_most_adjacent_label[source_label] = current_label
                        #need_refinement_labels.append({'label':source_label, 'class_trace':list_class_info[0], 'class_candidate': class_)                    
                
        #print(dict_most_adjacent_label)
        
        # determine which labels need refinement
        for source_label in dict_most_adjacent_label:
            if source_label in dict_adaptel_classes:
                list_class_info = dict_adaptel_classes[source_label]
                if len(list_class_info) == 1:
                    adjacent_label = dict_most_adjacent_label[source_label]                    
                    adjacent_class = dict_adaptel_class_classified[adjacent_label][0]
                    source_class = list_class_info[0]['class_id']
                    #print(source_label, adjacent_label, source_class, adjacent_class)
                    if adjacent_class != source_class:
                        need_refinement_labels.append({'label': source_label, 'class_trace': source_class, 'class_candidate': adjacent_class})
                        #print(source_label, adjacent_label, source_class, adjacent_class)
        """
        for label, list_class_info in dict_adaptel_classes.items():
            if  len(list_class_info) == 1:
                pass
                need_refinement_labels.append({'label':label, 'class_trace':list_class_info[0], 'class_candidate': list_class_info[0])
        """
        
        # ignore conflicting selections
        conflicting_classes = []
        for key in dict_adaptel_class_classified:
            if len(dict_adaptel_class_classified[key]) > 1:
                #pass
                #dict_adaptel_class_classified.pop(key)
                dict_adaptel_class_classified[key] = []
                conflicting_classes.append(key)
                
        return dict_adaptel_class_classified, conflicting_classes, need_refinement_labels

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)     
# # END - GROW TRACES    


# # BEGIN - RESOLVE CONFLICTS
"""
def resolve_selection_conflict(dict_adaptel_classes_temp, conflicting_classes, traces):
    # TBD
    
    return dict_adaptel_classes_temp
"""
# # END - RESOLVE CONFLICTS