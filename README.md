# maximkaGPT

A project for training and chatting with a neural network based on PyTorch.

# Requirements

- Python 3.9
- PyTorch
- tokenizers

# Files and Usage

- `main.py`  
  Used for training the neural network.

- `chat.py`
  Used to interact with saved neural network.

- `clear.py`  
  Cleans the `dataset.txt` file by removing numbers and empty lines. The `txt.txt` file contains poems used in the project.

- `txttokenizer.py`  
  Tokenizer script. **Do not modify this file.**

# Setup

1. Install Python 3.9.13 if you haven't already.
2. Install dependencies:

   `pip install torch tokenizers`

3. If you need, change `dataset.txt` file with poems.
4. Run `clear.py` to clean the data
5. Train the model to start chatting:

   `python main.py`

6. Now you can chat with the model:
   
   `python chat.py`

# With current dataset it can only speak russian and has troubles with sensible speech
