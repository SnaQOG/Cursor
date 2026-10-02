#!/bin/bash
# Webversion für den Chat-Upload (Grenze 30 MB pro Datei): H.264 High, Zwei-Pass, faststart, ohne Ton.
#   ./web_version.sh master.mp4 web.mp4 <Bitrate, z. B. 10M>   (ffmpeg aus imageio-ffmpeg oder System)
#   Bitrate so wählen, dass Dauer × Bitrate / 8 unter 30 MB bleibt.
FF=${FFMPEG:-$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || echo ffmpeg)}
IN=$(realpath "$1"); OUT=$(realpath -m "$2"); BR="$3"
cd "$(dirname "$OUT")"
"$FF" -y -loglevel error -i "$IN" -c:v libx264 -preset slow -profile:v high -b:v "$BR" -maxrate "$BR" -bufsize "$BR" \
  -pass 1 -an -f mp4 /dev/null && \
"$FF" -y -loglevel error -i "$IN" -c:v libx264 -preset slow -profile:v high -b:v "$BR" -maxrate "$BR" -bufsize "$BR" \
  -pass 2 -pix_fmt yuv420p -movflags +faststart -an "$OUT" && rm -f ffmpeg2pass-0.log*
ls -la "$OUT"
