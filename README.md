# SpQuant-SNN: ultra-low precision membrane potential with sparse activations unlock the potential of on-device spiking neural network applications

SpQuant-SNN introduces an innovative quantization strategy for spiking neural networks (SNNs), enabling **ultra-low precision membrane potentials** combined with **sparse activations**. This approach significantly improves efficiency and unlocks the potential for **on-device spiking neural network applications**, such as energy-efficient edge AI.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Ultra-Low Precision Quantization**: Employs novel quantization techniques tailored for spiking neural networks.
- **Sparse Activations**: Reduces computational overhead by leveraging sparsity in activations.
- **Energy Efficiency**: Optimized for on-device inference, ensuring low energy consumption for edge devices.
- **Flexible Framework**: Easy integration with popular deep learning libraries like PyTorch.
- **Scalability**: Demonstrates high performance across various SNN architectures and applications.

---

## Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.12
- CUDA (Optional, for GPU acceleration)

### Install Required Packages
Clone the repository and install dependencies:

```bash
git clone https://github.com/Ahmedhasssan/IM_SNN-SpQuant_SNN.git
cd IM_SNN-SpQuant_SNN
pip install -r requirements.txt
```

## Usage

### Example Scripts
The repository includes examples for training and evaluating SNNs on popular Event (**DVS-MNIST**, **DVS-CIFAR-10**) and Static image (**MNIST**, **CIFAR-10** and **Caltech-101**) datasets:

```bash
bash scripts/vgg9_dvs_cifar.sh
python examples/vgg9_dvs_caltech.sh
```

## Methodology

SpQuant-SNN and IM-SNN introduces:

1. **Membrane Potential Quantization**:
   - Reduces the precision of membrane potential representation to as low as 1.5 bits without degrading performance.

2. **Sparse Activations**:
   - Utilizes sparsity in spiking activations to minimize redundant computations, achieving significant energy savings.

3. **End-to-End Pipeline**:
   - Combines quantization and sparsity in a unified framework for seamless training and inference.

For a detailed explanation, refer to our [paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1440000/full).

## Results

SpQuant-SNN achieves state-of-the-art performance on spiking neural network benchmarks while dramatically reducing resource usage.

## Dataset: DVS-CIFAR10

| **Method**                   | **Architecture**       | **Weight Precision** | **Umem Precision** | **Weight Memory (MB)** | **Umem Memory (MB)** | **Total Memory (MB)** | **FLOPs Reduction** | **Top-1 Accuracy**       |
|------------------------------|------------------------|-----------------------|--------------------|-------------------------|-----------------------|-----------------------|--------------------|--------------------------|
| Our work (SNN-BL)            | VGG-9                 | 32-bit               | 32-bit             | 41.12                  | 3.68                 | 48.58                | 1×                | 78.45%                  |
| Our work (Quant-SNN)         | VGG-9                 | 2-bit                | 1.58-bit           | 2.57                   | 0.23                 | 3.75                 | 1×                | 77.94% (-0.51)          |
| Our work (SpQuant-SNN)       | VGG-9                 | 2-bit                | 1.58-bit           | 2.57                   | 0.23                 | 3.75                 | 5.0×              | 76.80% (-1.14)          |              |

## Dataset: CIFAR10

| **Method**                   | **Architecture**       | **Weight Precision** | **Umem Precision** | **Weight Memory (MB)** | **Umem Memory (MB)** | **Total Memory (MB)** | **FLOPs Reduction** | **Top-1 Accuracy**       |
|------------------------------|------------------------|-----------------------|--------------------|-------------------------|-----------------------|-----------------------|--------------------|--------------------------|
| Our work (SNN-BL)  | ResNet-19 | 32-bit | 32-bit | 49.94 | 5.5 | 60.94 | 1× | 94.56%|
| Our work (Quant-SNN) | ResNet-19 | 4-bit | 1.58-bit | 6.24 | 0.25 | 7.49 | 1× | 94.11% (-0.45)|
| Our work (Quant-SNN) | Spikformer-4-256 | 8-bit | 1.58-bit | 9.62 | 0.25 | 15.26 | 1× | 94.99% (-0.52)|
| Our work (SpQuant-SNN) | ResNet-19 | 4-bit | 1.58-bit | 6.24 | 0.25 | 7.49 | 5.1× | 93.09% (-1.48)|


Experimental results of Quant-SNN and SpQuant-SNN on DVS datasets using T = 10. These results highlight the effectiveness of SpQuant-SNN in achieving high accuracy and energy efficiency for edge AI applications.

## Contact

For any inquiries or collaboration opportunities, feel free to reach out:

- **Email**: [ah2288.@cornell.edu](mailto:ah2288@cornell.edu)
- **GitHub**: [Ahmedhasssan](https://github.com/Ahmedhasssan)

We welcome feedback, suggestions, and contributions to enhance SpQuant-SNN!

