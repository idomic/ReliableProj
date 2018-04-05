#!/bin/sh
# Iterate over files and generate a bitmap file with afl-showmap tool 
find ./fuzzer01/queue -type f -size +10k -delete
find ./fuzzer02/queue -type f -size +10k -delete
mkdir ./fuzzer02/bitmaps ./fuzzer01/bitmaps

# Run over fuzzer01 and then fuzzer 02
for file in ./fuzzer01/queue/id*
do
   # take action on each file.
   #fname=`basename $file`
   #echo "fileName: $fname"
   #../afl-showmap -q -o ./fuzzer01/bitmaps/$fname -- /home/ubuntu/binafl/binutils-2.30/binutils/objdump -D $file
   # cat ./fuzzer01/bitmaps/$fname | cut -d ':' -f1 > ./fuzzer01/bitmaps/$fname
done

for file in ./fuzzer02/queue/id*
do
   # take action on each file.
   #fname=`basename $file`
   #echo "fileName: $fname"
   #../afl-showmap -q -o ./fuzzer02/bitmaps/$fname -- /home/ubuntu/binafl/binutils-2.30/binutils/objdump -D $file
   #cat ./fuzzer02/bitmaps/$fname | cut -d ':' -f1 > ./fuzzer02/bitmaps/$fname
done
date
#TODO iterate with for loop over bitmaps
# Remove second coloumn from the bitmap file
#cat tester | cut -d ':' -f1 > bit

# End of script
