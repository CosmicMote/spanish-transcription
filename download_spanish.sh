#!/usr/bin/env bash
# Download the audio of a YouTube video as an mp3 (macOS/Linux equivalent of download_spanish.ps1).
set -euo pipefail

usage() {
    echo "Usage: $0 [-b browser] [-o output_dir] <url>" >&2
    echo "  -b  Browser to read cookies from (default: firefox)" >&2
    echo "  -o  Folder to save the mp3 (default: ~/Downloads)" >&2
    exit 1
}

browser="firefox"
output_dir="$HOME/Downloads"

while getopts ":b:o:h" opt; do
    case "$opt" in
        b) browser="$OPTARG" ;;
        o) output_dir="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

[ $# -eq 1 ] || usage
url="$1"

yt-dlp \
    --cookies-from-browser "$browser" \
    --extract-audio \
    --audio-format mp3 \
    --audio-quality 0 \
    --output "$output_dir/%(title)s.%(ext)s" \
    "$url"
