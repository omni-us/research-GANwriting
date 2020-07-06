#!/bin/sh

find $1 -mindepth 2 -type f -exec mv -t $1 -i '{}' +
find $1 -type d -empty -delete
