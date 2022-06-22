#!/bin/bash

conda activate innotis

if [[ $(basename $(pwd)) = "docker" ]];then
	cd ..
fi

DEBUG=$1

python3 app.py $DEBUG
