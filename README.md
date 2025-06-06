# Deep Learning Tutorial for Beginners 🧠

## Overview

This project provides an educational introduction to deep learning concepts through hands-on examples with extensive logging and explanations. It includes:

- **Simple Demo** (`simple_demo.py`): A pure Python implementation that teaches a single neuron to learn the AND gate, demonstrating core concepts like weights, bias, forward pass, and backpropagation without any external libraries.

- **PyTorch Implementation** (`neural_network.py`): A full neural network example using PyTorch that creates synthetic classification data, builds a 2-layer network, and achieves ~93% accuracy with detailed educational logging throughout the process.

- **Test Suite** (`test_neural_network.py`): Unit tests following Test-Driven Development (TDD) principles to verify all functionality.

All code includes console logs with natural language explanations of what's happening, what concepts mean, and what's considered good vs bad performance.

## Getting Started

1. **Clone the repository and navigate to the project directory**

2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the examples**:
   ```bash
   # Run the simple demo (no external libraries needed)
   python simple_demo.py

   # Run the full PyTorch neural network example
   python neural_network.py

   # Run the test suite
   python test_neural_network.py
   ```

5. **Experiment and learn**:
   - Try changing the learning rate in the examples
   - Modify the network architecture (hidden layer size)
   - Adjust the number of training epochs
   - Observe how these changes affect performance!

💡 **Tip**: Start with `simple_demo.py` to understand the basics, then move to `neural_network.py` for a more realistic implementation. 