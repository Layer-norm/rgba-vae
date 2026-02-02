# RGBA-VAE (WIP)

## Overview

**RGBA-VAE** will be a Variational Autoencoder (VAE) model specifically designed for processing and generating RGBA images. Now it is in a very early stage of development, feel free to contribute or provide feedback!
## Features

- **VAE Architecture**: Implements a variational autoencoder specifically adapted for RGBA image data
- **Transparency Handling**: Properly processes and reconstructs alpha/transparency channel in both hard and soft modes.

## Requirements

- Python >= 3.12
- matplotlib >= 3.10.8
- numpy
- pillow >= 10.0.0
- torch
- torchvision
- transformers >= 4.57.6

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd rgba-vae

# Install dependencies using uv
uv sync
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

*Project maintained by Faxuan Cai*  
*Contact: magicianaio@gmail.com*