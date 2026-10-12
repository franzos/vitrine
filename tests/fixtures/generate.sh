#!/bin/sh
# Each format has its own size so tests can tell them apart. Run from the repo root:
#   guix shell ffmpeg imagemagick libheif libjxl perl-image-exiftool python -- sh tests/fixtures/generate.sh
set -eu

cd "$(dirname "$0")"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

src() {
    convert -size "$1" gradient:'#c04020-#2040c0' -depth 8 -strip "$tmp/$1.png"
    echo "$tmp/$1.png"
}

convert "$(src 64x48)" -quality 85 jpeg.jpg
rm -f orientation-6.jpg
exiftool -q -Orientation#=6 -o orientation-6.jpg jpeg.jpg
# Lisbon
rm -f gps.jpg
exiftool -q -GPSLatitude=38.7223 -GPSLatitudeRef=N -GPSLongitude=9.1393 -GPSLongitudeRef=W -o gps.jpg jpeg.jpg
convert "$(src 60x44)" -define png:include-chunk=none PNG24:png.png
convert "$(src 56x40)" -define webp:lossless=true webp.webp
convert "$(src 52x36)" gif.gif
convert "$(src 48x32)" -type TrueColor -compress lzw -define tiff:rows-per-strip=8 tiff.tif
heif-enc -q 80 -o heic.heic "$(src 68x52)"
heif-enc -A -q 80 -o avif.avif "$(src 70x54)"
cjxl -q 90 "$(src 74x58)" jxl.jxl
convert "$(src 80x60)" -quality 85 "$tmp/preview.jpg"
python3 dng.py "$tmp/preview.jpg" dng.dng
convert "$(src 84x56)" -quality 85 "$tmp/preview-6.jpg"
python3 dng.py "$tmp/preview-6.jpg" dng-orientation-6.dng 6
# 1024x768 stored, turned upright to 768x1024, with a 320x240 EXIF thumbnail
convert "$(src 1024x768)" -quality 20 "$tmp/large.jpg"
convert "$(src 320x240)" -quality 50 "$tmp/thumb.jpg"
rm -f exif-thumb.jpg
exiftool -q -Orientation#=6 '-ThumbnailImage<='"$tmp/thumb.jpg" -o exif-thumb.jpg "$tmp/large.jpg"
# 1 s of testsrc at 72x56, and the same stream tagged to display turned 90 degrees
ffmpeg -nostdin -v error -y -f lavfi -i testsrc=size=72x56:rate=5:duration=1 -c:v libx264 -pix_fmt yuv420p \
    -metadata creation_time=2024-05-01T12:00:00Z clip.mp4
ffmpeg -nostdin -v error -y -display_rotation 90 -i clip.mp4 -c copy \
    -metadata creation_time=2024-05-01T12:00:00Z clip-rotated.mp4
