import glob
import subprocess
file_list = glob.glob('/home/dongdong/Project/size/adv_gen_seeds/*')
#seed_list = glob.glob('/home/dongdong/Project/objdump/afl_out/afl_out/queue/*')
base_list = glob.glob('/home/dongdong/Project/size/afl_out/seeds/*')
import os
import time
stdout = subprocess.STDOUT
call = subprocess.call
FNULL = open(os.devnull, 'w')
t0 = time.time()
lee= len(file_list)*51000
cnt = 0

idx = 0
for fll in file_list:
    seed_list = glob.glob(fll+"/*")
    for fl in seed_list:
        if not (idx % 5000):
            print(100*idx/float(lee))
        ret = call(["/home/dongdong/Project/size/binutils-2.30_gcov/binutils/size",fl], stdout=FNULL, stderr=stdout)
        if ret < 0:
            print("crash " + str(ret) + fl)
        idx+=1
print(time.time()-t0)

'''
for idx,fl in enumerate(base_list):
    if not (idx % 1000):
        print(100*idx/float(len(base_list)))
    ret = call(["/home/dongdong/Project/size/binutils-2.30_gcov/binutils/size", fl], stdout=FNULL, stderr=stdout)
'''
