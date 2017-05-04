#!/bin/bash

echo 'Downloading dataset...'

#download dataset first,
wget http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/en.tar.gz

echo 'Moving to res subdirectory...'
mkdir -p ../data/osdb_raw_data/set1/
mv download.php?f=OpenSubtitles2016%2Fen.tar.gz ../data/osdb_raw_data/set1/metadata.tar.gz

cd ../data/osdb_raw_data/set1

echo 'Unzipping dataset...'

#Making assumption that user hasn't put any other tar files in folder
#Two tar ball extractions because during testing it downloaded as .gz once?
tar -xvf *.tar
tar -xvf *.gz

echo 'Extracting dataset...'

#extract all filesin sub-directories
find . -name '*.gz' -exec gunzip '{}' \;

cd ../../../src

echo 'Running python preprocessor...'

#run python pre-processor
python opensubtitleparser.py

