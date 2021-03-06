import pickle,glob
import numpy as np
bitmap_list = glob.glob('/home/yuval/Documents/ReliableProj/afl-2.52b/afl_out/fuzzer01/bitmaps/*')
import time
with open('new_int','rb') as f:
	label = f.read().split('\n')[:-1]
label = filter(None, label)
label = [int(f) for f in label]
MAX_BITMAP_SIZE = len(label)
t0 = time.time()
for i in bitmap_list:
    bitmap = np.zeros((1,MAX_BITMAP_SIZE))
    tmp = open(i,'rb').read().split('\n')[:-1]
    for j in tmp:
        bitmap[0][label.index((int(j)))] = 1
    file_name = "/home/yuval/Documents/ReliableProj/afl-2.52b/afl_out/convert_bitmaps/"+i.split('/')[-1]
    np.save(file_name,bitmap)

print(time.time()-t0)

