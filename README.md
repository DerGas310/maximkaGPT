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
  Cleans the `dataset.txt` file by removing numbers and empty lines. The `datast.txt` file contains dataset of poems.

- `txttokenizer.py`  
  Tokenizer script. **Do not modify this file.**

# Setup

1. Install Python 3.9.13 if you haven't already.
2. Install dependencies:

   `pip install torch tokenizers`

3. If you need, change `dataset.txt`.
4. Then run `clear.py` to preparr the dataset. 
5. Train the model to start chatting:

   `python main.py`

6. Now you can chat with the model:
   
   `python chat.py`

P.S: it speaks only Russian
