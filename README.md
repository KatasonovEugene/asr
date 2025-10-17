# Automatic Speech Recognition

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#how-to-train">How To Train</a> •
  <a href="#inference">Inference</a> •
  <a href="#custom-dataset-inference">Custom Dataset Inference</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## Installation

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```
2. Install model weights & LM

   ```bash
   python3 scripts/download_everything.py
   ```
## How To Train

For training the same model, you need to train in 2 phases.

1. Phase 1

   ```bash
   python3 train.py -cn=train_phase_1
   ```
2. Phase 2

   ```bash
   python3 train.py -cn=train_phase_2
   ```

## Inference

To reproduce original result:

   ```bash
   python3 inference.py -cn=inference_lm
   ```

## Custom Dataset Inference

To evaluate on custom dataset:
   ```bash
   python3 inference.py -cn=inference_custom datasets.test.0.audio_dir=<YOUR_PATH>
   ```
WER/CER metrics will be automatically evaluated

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
