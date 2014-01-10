import persistent
import codecs
import math
import scipy.io as sio
import numpy
d = persistent.load_var('tag_support')
t = sorted(d.iteritems(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
res = []
template = u'{}: {} photos by {} users over {} days'
numeric = numpy.zeros((len(t), 3), dtype=numpy.int32)
i = 0
for tag, info in t:
    days = int(math.ceil((info[3] - info[2]).total_seconds()/3600))
    res.append(template.format(tag, info[0], info[1], days))
    numeric[i, :] = [info[0], info[1], days]
    i += 1

sio.savemat('tag_support_num', {'t': numeric})
# with codecs.open('tag_support.txt', 'w', 'utf8') as f:
#     f.write('\n'.join(res))
