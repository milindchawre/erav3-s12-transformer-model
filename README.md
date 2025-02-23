# Transformer Model Training

This project implements a transformer-based language model using PyTorch. The model is designed to learn from a text corpus and can be trained and fine-tuned for various natural language processing tasks.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Actual Training](#actual-training)
- [Checkpointing](#checkpointing)
- [Model Compression](#model-compression)
- [Working Demo](#working-demo)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- Transformer architecture with causal self-attention and feedforward layers.
- Efficient data loading and batching.
- Checkpointing to resume training.
- Support for multiple devices (CPU, CUDA, MPS).
- Model compression for reduced file size.
- Streamlit application for text generation using the trained model.

## Requirements
- Python 3.6 or higher
- PyTorch 1.7 or higher
- tqdm
- tiktoken
- streamlit
- transformers

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/transformer-model-training.git
   cd transformer-model-training
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your text data in a file named `input.txt`. The model will read this file to load tokens for training.

2. To train the model, run the training script:
   ```bash
   python train.py
   ```

3. The model will save checkpoints after each epoch in `checkpoint.pt` and the final model in `trained_model_quantized.pt`.

4. To generate text using the trained model, run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

5. Enter your text and specify the length of additional text to generate in the Streamlit interface.

## Training
- The model is trained using a batch size of 4 and a learning rate of 3e-4.
- The training loop includes loss calculation, backpropagation, and optimizer steps.
- The loss is monitored, and checkpoints are saved to allow for resuming training.
- The training process is logged in `training.log`, which contains detailed statistics for each epoch, including loss values and checkpointing information.
- The training process supports early stopping if the loss falls below a specified threshold (0.099999), which can help prevent overfitting and reduce training time.

## Actual Training
The model was trained for a total of **91 epochs**. The training process involved the following steps:
- **Data Preparation**: The model reads and encodes text data from `input.txt`, loading a total of **338,025 tokens**.
- **Batch Processing**: Each epoch consists of **82 batches**, with each batch containing sequences of tokens for training.
- **Loss Monitoring**: The loss is calculated at each step, and the model's performance is tracked throughout the training process.
- **Checkpointing**: The model state and current epoch are saved in a single checkpoint file (`checkpoint.pt`) after each epoch, allowing for recovery in case of interruptions.
- **Final Model**: After training, the model is saved with quantization and compression as `trained_model_quantized.pt`, reducing the file size for easier deployment. The final loss achieved at the end of training was approximately 0.089421.

The training log file contains detailed statistics for each epoch, including loss values and checkpointing information. You can find the log file named `training.log` in the project directory.

## Checkpointing
- The model state and current epoch are saved in a single checkpoint file (`checkpoint.pt`).
- To resume training from the last checkpoint, simply run the training script again. The model will automatically load the latest checkpoint.

## Model Compression
- The final model is saved with compression to reduce file size. The model file will be saved as `trained_model_quantized.pt`.

## Working Demo
You can try out the working demo of the model on Hugging Face Spaces:

![Hugging Face Spaces Demo](https://link-to-your-image.com/demo-image.png)

[Play with the Demo Here](https://huggingface.co/spaces/yourusername/your-demo)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- This project is inspired by the original GPT architecture and various resources available in the NLP community.
