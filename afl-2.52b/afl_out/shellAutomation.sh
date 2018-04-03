#!/bin/sh
# Iterate over files and generate a bitmap file with afl-showmap tool 
for file in ./queue/id*
do
  echo "Processing config file: $file"
  # take action on each file.
  fname=`basename $file`
  echo "fileName: $fname"
  ../afl-showmap -o ./bitmaps/$fname -- /home/ubuntu/binafl/binutils-2.30/binutils/objdump -D $file
  done
date


# End of script
