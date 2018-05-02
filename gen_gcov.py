import glob
import subprocess
file_list = glob.glob('/home/reliableProj/ReliableProj/adv_gen_seeds_old/*')
#seed_list = glob.glob('/home/dongdong/Project/objdump/afl_out/afl_out/queue/*')
base_list = glob.glob('/home/reliableProj/ReliableProj/afl-2.52b/afl_out/seeds/*')
import os
import time
import sys
stdout = subprocess.STDOUT
call = subprocess.call
FNULL = open(os.devnull, 'w')
t0 = time.time()
lee= len(file_list)*51000
cnt = 0

idx = 0
for fll in file_list:
    seed_list = glob.glob(fll+"/*")
    #print(seed_list)
    for fl in seed_list:
	#print(fl)
        if not (idx % 10):#changed from 5000
            print(100*idx/float(lee))
        try:
            ret = call(["/home/reliableProj/ReliableProj/afl_gcov/binutils-2.30/binutils/objdump","-D",fl], stdout=FNULL, stderr=stdout, timeout=3600)
        
            if ret < 0:
                print("crash " + str(ret) + fl)
            sys.stdout.flush()
            idx+=1
        except subprocess.TimeoutExpired: 
            print(ret) 
print(time.time()-t0)
'''
for idx,fl in enumerate(base_list):
    if not (idx % 1000):
        print(100*idx/float(len(base_list)))
    ret = call(["/home/reliableProj/ReliableProj/afl_gcov/binutils-2.30/binutils/objdump", fl], stdout=FNULL, stderr=stdout)

print(time.time()-t0)
'''
