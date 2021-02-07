#!/bin/bash

pwd
cd dataset
curl http://www.stud.fit.vutbr.cz/~xforto00/dataset1.zip -o dataset1.zip
unzip dataset1.zip -d "dataset1"
curl http://www.stud.fit.vutbr.cz/~xforto00/dataset2.zip -o dataset2.zip
unzip dataset2.zip -d "dataset2"

cd ..
cd markedImg
curl http://www.stud.fit.vutbr.cz/~xforto00/dataset1Examples.zip -o dataset1Examples.zip
unzip dataset1Examples.zip -d "dataset1Examples"
curl http://www.stud.fit.vutbr.cz/~xforto00/dataset2Examples.zip -o dataset2Examples.zip
unzip dataset2Examples.zip -d "dataset2Examples"
