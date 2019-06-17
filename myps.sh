#!/bin/bash
ps axu | grep $1 | awk 'match($0, /--mode /) {print $2 "\t" substr($0, RSTART, RLENGTH+70) }'
