# Uzbek TTS (Text-to-Speech)

A high-quality text-to-speech system for Uzbek language based on Conditional Flow Matching (CFM) architecture.

## 🎯 Overview

This project implements a neural text-to-speech system specifically designed for the Uzbek language. It uses a Conditional Flow Matching approach with a DiT (Diffusion Transformer) backbone to generate natural-sounding speech from Uzbek text.

## ✨ Features

- 🎵 **High-quality voice synthesis** for Uzbek language
- 🎭 **Voice cloning** capabilities using reference audio
- ⚡ **Configurable speech speed** and generation parameters
- 🚀 **GPU acceleration** with automatic device detection
- 🎧 **Multiple audio formats** support (WAV, OGG)
- 🔒 **Thread-safe** implementation with caching

## 📁 Project Structure

```
Uzbek_TTS/
├── ckpts/                      # Model checkpoints directory
│   └── model.safetensors       # Pre-trained model file
├── src/                        # Source code
│   ├── models/                 # Model architectures
│   ├── utils/                  # Utility functions
│   └── inference.py            # Inference pipeline
├── examples/                   # Usage examples
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── setup.py                    # Installation script
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Uzbek_TTS.git
   cd Uzbek_TTS
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the pre-trained model:**
   
   Download the model from [Google Drive](https://drive.google.com/file/d/1RIqMhxux-RDT9RxbiMDKpYkJlE70Mq4A/view?usp=sharing) and place it in the `ckpts/` folder:
   
   ```bash
   # Create checkpoints directory
   mkdir -p ckpts
   
   # Place the downloaded model.safetensors file in ckpts/
   # The file structure should be: ckpts/model.safetensors
   ```
## Quick Start

### Basic Usage

```python
from omegaconf import OmegaConf
from hydra.utils import get_class
from tts import TTS

# Load configuration
model_cfg = OmegaConf.load('config/UZTTS_conf.yaml')

# Initialize TTS
tts = TTS(
    ref_audio_path="test_data/test_erkak.wav",
    ref_text="Jizzax kollejlarida infraqizil aniqlagichli turniketlar o'rnatilmoqda.",
    model_cfg=model_cfg,
    model_cls=get_class(f"uz_tts.model.{model_cfg.model.backbone}"),
    vocab='config/uz_vocab.txt',
    ckpt_path="ckpts/UZ.safetensors",
    device="auto",
    speed=1.0
)

# Generate speech
audio, sample_rate = tts.generate_speech("Assalomu alaykum! Bu Uzbek TTS tizimidir.")

# Save audio
tts.save_audio(audio, "output.wav")
```
