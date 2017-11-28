def set_rect(x_tl,y_tl, x_br,y_br):
	# приводит координаты прямоугольника из вида (x_topleft, y_topleft, x_bottomright, y_bottomright) в вид (left, right, top, bottom)
	bound = dict()
	bound['left']=x_tl
	bound['right']=x_br
	bound['top']=y_tl
	bound['bottom']=y_br
	return bound

def rect_intersection(tup_1, tup_2):
	# площадь пересечения прямоугольников
	# задаются в виде  (x_topleft, y_topleft, x_bottomright, y_bottomright) - кортежей
  rect1 = set_rect(*tup_1)
  rect2 = set_rect(*tup_2)
  x_overlap = max(0, min(rect1['right'], rect2['right']) - max(rect1['left'], rect2['left']))
  y_overlap = max(0, min(rect1['bottom'], rect2['bottom']) - max(rect1['top'], rect2['top']))
  return x_overlap*y_overlap

def rect_union(tup_1, tup_2):
	# площадь объединения прямоугольников
	# задаются в виде  (x_topleft, y_topleft, x_bottomright, y_bottomright) - кортежей
  s1 = rect_square(tup_1)
  s2 = rect_square(tup_2)
  return s1+s2-rect_intersection(tup_1, tup_2)

def rect_square(tup):
    rect = set_rect(*tup)
    s = (rect['right']-rect['left'])*(rect['bottom']-rect['top'])
    if s<0:
        raise ValueError('bad coordinates in rect_square, negative S')
    return s

def IoU(tup_1, tup_2):
 	# площадь (пересечения прямоугольников)/(площадь объединения)
	# задаются в виде  (x_topleft, y_topleft, x_bottomright, y_bottomright) - кортежей
  intersection = rect_intersection(tup_1, tup_2)
  union = rect_union(tup_1, tup_2)
  return (intersection/union)
