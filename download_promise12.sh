#!/bin/bash

mkdir -p data/PROMISE12

echo "Downloading PROMISE12 data..."
curl -L -o data/PROMISE12_live_challenge_test_data.zip "https://zenodo.org/records/8026660/files/livechallenge_test_data.zip?download=1"
curl -L -o data/PROMISE12_test_data.zip "https://zenodo.org/records/8026660/files/test_data.zip?download=1"
curl -L -o data/PROMISE12_training_data.zip "https://zenodo.org/records/8026660/files/training_data.zip?download=1"

echo "Unzipping..."
unzip -o data/PROMISE12_live_challenge_test_data.zip -d data/PROMISE12

echo "Done! Data is in ./data/PROMISE12/"
