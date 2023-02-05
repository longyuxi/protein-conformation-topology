#!/bin/bash

# code for downloading pdbbind into current directory

wget https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_refined.tar.gz
tar -xzvf PDBbind_v2020_refined.tar.gz
cd refined_set

rm -rf index/
rm -rf readme/