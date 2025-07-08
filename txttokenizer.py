from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["dataset.txt"], vocab_size=50000, min_frequency=2)
tokenizer.save_model("tokenizer")