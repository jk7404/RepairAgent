#!/bin/bash
sudo apt-get update 
sudo apt-get install -y openjdk-11-jdk subversion dos2unix 
git clone https://github.com/rjust/defects4j
cd defects4j
sudo apt-get install -y cpanminus 
cpanm --installdeps . 
./init.sh
sudo apt-get install -y libdbi-perl
cd ..
python3 -m pip install -r requirements.txt
python3 get_defects4j_list.py