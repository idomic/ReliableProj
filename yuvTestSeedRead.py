import struct
import re


# this one is for reading seeds and translating them into ints
'''fin = open("C:\Users\yuval\Downloads\id_000000,orig_hello", "rb")
print(struct.unpack('i', fin.read(4)))
fin.close()

fin = open("C:\Users\yuval\Downloads\id_000001,src_000000,op_flip1,pos_0,+cov", "rb")
print(struct.unpack('i', fin.read(4)))
fin.close()

fin = open("C:\Users\yuval\Downloads\id_000001,src_000000,op_havoc,rep_2,+cov", "rb")
print(struct.unpack('i', fin.read(4)))
fin.close()
'''

# this one is for reading bitmaps and getting them into arrays
with open('yuvBitmapTest\myBitmap') as myFile:
  text = myFile.read()
myBitmap = re.sub(r':.','',text)
result = myBitmap.split("\n")

print(result) 