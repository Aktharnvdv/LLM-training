# Fine-Tuning 1-B Falcon Language Model

This repository contains a script for fine-tuning the 1B Falcon Language Model on a custom dataset using the Hugging Face Transformers library. The model is fine-tuned for a conversational language understanding task with a dataset provided in the `train_data.json` file.

## Requirements

- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)

Install the required libraries using the following command:

```bash
pip install torch transformers
```
# Dataset
The training data is loaded from the train_data.json file, which should contain samples with 'user' and 'assistant' roles. The script creates a custom dataset class, MyCustomDataset, for handling the data.

# Model Fine-Tuning
The Bling Falcon Language Model is fine-tuned using the specified tokenizer and the provided training data. The model is trained for a specified number of epochs.

# Tokenizer
The script uses the Hugging Face Transformers library to load and customize the tokenizer for the specified Bling Falcon model. Special tokens are added, and the tokenizer is used to preprocess input data for training.

# Training
The train_epoch function is responsible for training the model for each epoch. It prints examples of input text, generated text, and gold label (assistant input) during training.

# Configuration
You can modify the script's configuration, such as batch size, learning rate, and the number of epochs, according to your specific requirements.

# Save Fine-Tuned Model
After training, the fine-tuned Bling Falcon model and its tokenizer are saved to the fine_tuned_model directory.
