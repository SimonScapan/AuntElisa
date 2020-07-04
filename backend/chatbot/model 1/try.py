import numpy as np
VOCAB_SIZE = 67520
MAX_LEN = 20
num_samples = 8000
decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")