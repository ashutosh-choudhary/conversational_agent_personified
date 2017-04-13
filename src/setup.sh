#!/bin/bash

echo 'Downloading dataset...'

#download dataset first,
wget http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/en.tar.gz

echo 'Moving to res subdirectory...'
mkdir -p ../res/osdb/set1/
mv download.php?f=OpenSubtitles%2Fen.tar.gz ../res/osdb/set1/metadata.tar.gz

cd ../res/osdb/set1/

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

