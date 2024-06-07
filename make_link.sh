#!/bin/bash

# Source folder
SOURCE_FOLDER="/WAVE/datasets/dmlab/aicity/aic24/track1"

# Destination folder 
DEST_FOLDER="./data/aic24"

# Create destination folder
mkdir -p "$DEST_FOLDER"

# Find all files in source folder recursively
find "$SOURCE_FOLDER" -type f -print0 | while read -d $'\0' file; do

  # Get relative path 
  rel_path=$(realpath --relative-to="$SOURCE_FOLDER" "$file")

  # Create parent folders in destination
  dest_dir="$DEST_FOLDER/$rel_path"
  mkdir -p "${dest_dir%/*}"

  # Create symbolic link
  ln -s "$file" "$dest_dir"

done