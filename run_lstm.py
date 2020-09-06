# Hyper-parameters
from lstm_next_sequence_prediction.generate_dateset import generate_dataset, sequences_to_dicts, create_datasets, \
    Dataset, one_hot_encode, one_hot_encode_sequence
from lstm_next_sequence_prediction.lstm_network import LSTM_Net
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 200
hidden_size = 50
vocab_size = 4
# Initialize a new network
z_size = hidden_size + vocab_size  # Size of concatenated hidden + input vector
lstm_net = LSTM_Net(hidden_size, vocab_size)
params = lstm_net.init_lstm()
# Initialize hidden state as zeros
hidden_state = np.zeros((hidden_size, 1))
sequences = generate_dataset()
"""
print('samples from the generated dataset:')
for s in sequences:
    print(s)
"""
word_to_idx, idx_to_word, num_sequences, vocab_size = sequences_to_dicts(sequences)

print('We have {num_sequences} sentences and {len(word_to_idx)} unique tokens in our dataset (including UNK).\n')
print('The index of \'b\' is', word_to_idx['b'])
print('The word corresponding to index 1 is \'{idx_to_word[1]}\'')

assert idx_to_word[word_to_idx['b']] == 'b', \
    'Consistency error: something went wrong in the conversion.'
training_set, validation_set, test_set = create_datasets(sequences, Dataset)

print('We have {len(training_set)} samples in the training set.')
print('We have {len(validation_set)} samples in the validation set.')
print('We have {len(test_set)} samples in the test set.')
test_word = one_hot_encode(word_to_idx['a'], vocab_size)
print('Our one-hot encoding of \'a\' has shape {test_word.shape}.')

test_sentence = one_hot_encode_sequence(['a', 'b'], vocab_size)
print('Our one-hot encoding of \'a b\' has shape {test_sentence.shape}.')
# Track loss
training_loss, validation_loss = [], []

# For each epoch
for i in range(num_epochs):

    # Track loss
    epoch_training_loss = 0
    epoch_validation_loss = 0

    # For each sentence in validation set
    for inputs, targets in validation_set:
        # One-hot encode input and target sequence
        inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
        targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

        # Initialize hidden state and cell state as zeros
        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))

        # Forward pass
        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = lstm_net.forward(inputs_one_hot, h, c, params)

        # Backward pass
        loss, _ = lstm_net.backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params)

        # Update loss
        epoch_validation_loss += loss

    # For each sentence in training set
    for inputs, targets in training_set:
        # One-hot encode input and target sequence
        inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
        targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

        # Initialize hidden state and cell state as zeros
        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))

        # Forward pass
        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = lstm_net.forward(inputs_one_hot, h, c, params)

        # Backward pass
        loss, grads = lstm_net.backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params)

        # Update parameters
        params = lstm_net.update_parameters(params, grads, lr=1e-1)

        # Update loss
        epoch_training_loss += loss

    # Save loss for plot
    training_loss.append(epoch_training_loss / len(training_set))
    validation_loss.append(epoch_validation_loss / len(validation_set))

    # Print loss every 10 epochs
    if i % 10 == 0:
        print('Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

# Get first sentence in test set
inputs, targets = test_set[1]

# One-hot encode input and target sequence
inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

# Initialize hidden state as zeros
h = np.zeros((hidden_size, 1))
c = np.zeros((hidden_size, 1))

# Forward pass
z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = lstm_net.forward(inputs_one_hot, h, c, params)

# Print example
print('Input sentence:')
print(inputs)

print('\nTarget sequence:')
print(targets)

print('\nPredicted sequence:')
print([idx_to_word[np.argmax(output)] for output in outputs])

# Plot training and validation loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss', )
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()