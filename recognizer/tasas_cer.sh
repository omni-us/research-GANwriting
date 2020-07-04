#!/bin/bash
source /home/lkang/utils/htrsh.inc.sh
tasas <( htrsh_prep_tasas $1 $2 -f tab -c yes ) \
    -ie -s " " -f "|"
# $1: ground truth   $2: decoded
