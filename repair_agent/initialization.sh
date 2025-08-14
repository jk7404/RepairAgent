#!/bin/bash
sudo apt-get update 
sudo apt-get install -y openjdk-11-jdk subversion dos2unix 
cd defects4j 
sudo apt-get install -y cpanminus 
cpanm --installdeps . 
./init.sh
sudo apt-get install -y libdbi-perl
cd ..
python3 -m pip install -r requirements.txt