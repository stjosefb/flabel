import heapq


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
	queue = Queue()
	q_add = queue.add  # cache some functions
	q_pop = queue.pop
	q_empty = queue.is_empty
		
	dict_adaptel_class_classified = {}
	
	# adaptels with known classes 
	for label, list_class_info in dict_adaptel_classes.items():
		if len(list_class_info) > 0:
			dict_adaptel_class_classified[label] = list_class_info[0]['class_id']
            
	print(dict_adaptel_class_classified)
	print(dict_label_color)
    
	# init priority queue based on adaptels with known classes
	for label_ref in dict_adaptel_class_classified:		
		for label in adjacent_adaptels[label_ref]:
			if label not in dict_adaptel_class_classified:
				color_label = dict_label_color[label]
				color_label_ref = dict_label_color[label_ref]
				color_diff = norm_nd_sqr_arr(color_label, color_label_ref)
				q_add(color_diff, [label_ref, label, dict_adaptel_class_classified[label_ref]])
				print(label_ref, label, dict_adaptel_class_classified[label_ref], color_diff)
    
	# process queue
	while not q_empty():
		item = q_pop()
		value = item[2]
		if value[1] not in dict_adaptel_class_classified:
			dict_adaptel_class_classified[value[1]] = value[2]
			label_ref = value[1]
			for label in adjacent_adaptels[value[1]]:
				if label not in dict_adaptel_class_classified:
					color_label = dict_label_color[label]
					color_label_ref = dict_label_color[label_ref]
					color_diff = norm_nd_sqr_arr(color_label, color_label_ref)
					q_add(color_diff, [label_ref, label, dict_adaptel_class_classified[label_ref]])			

	return dict_adaptel_class_classified
# # END - GROW TRACES	
