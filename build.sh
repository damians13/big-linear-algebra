#!/usr/bin/env bash
rm dist/*
gcc -c main.c -g -std=c99 -o dist/main.o
gcc -c lib/matrix.c -g -std=c99 -o dist/matrix.o
gcc -c lib/csv.c -g -std=c99 -o dist/csv.o
gcc -c lib/layer.c -g -std=c99 -o dist/layer.o
gcc dist/* -g -std=c99 -o dist/main