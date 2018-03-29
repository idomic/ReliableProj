#!/bin/sh
# Iterate over files and generate a bitmap file with afl-showmap tool 
for file in ./queue/id*
do
  echo "Processing config file: $file"
  # take action on each file.
  ../afl-showmap -o  ./bitmaps/$file
        sleep 35
    done
done
date


# End of script
