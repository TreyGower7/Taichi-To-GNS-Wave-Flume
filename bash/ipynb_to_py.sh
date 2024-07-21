#!/bin/bash
for arg in "$*"
do
	echo $arg
	jupyter nbconvert --to python $arg	
done

for f in "."/*.py
do
	mv "${f}" ./scripts/
done

for f in "./taichi-ex"/*.py
do
	mv "${f}" ./scripts/
done
for f in "./taichi-ex/.ipynb_checkpoints"/*.py
do
	mv "${f}" ./scripts/
done
