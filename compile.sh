#!/bin/bash

g++ -I ./include/ ./src/videoStab.cpp ./src/vidStab.cpp -o ./bin/main `pkg-config --cflags --libs opencv`
