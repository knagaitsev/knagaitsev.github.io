#!/bin/bash

for img in ./*.{jpg,jpeg,png,tif,tiff}; do
    [ -f "$img" ] || continue
    magick "$img" -strip "$img"
done
