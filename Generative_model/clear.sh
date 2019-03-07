#!/bin/bash
DIRS=$(find . * -type d | grep -v "\.")
echo ${DIRS}
a=0
for dir in ${DIRS}
do
    rm ${dir}/*.png
    a=`expr ${a} + 1` && echo "Clear the ${a} times dir"
done