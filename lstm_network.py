import numpy as np

from lstm_next_sequence_prediction.activation_funcs import sigmoid, tanh, softmax
from lstm_next_sequence_prediction.initializer import init_orthogonal, clip_gradient_norm


class LSTM_Net():
    def __init__(self, hidden_size, vocab_size):
        self.hidden_size, self.vocab_size = hidden_size, vocab_size

    def init_lstm(self):
        """
        Initializes our LSTM network.

        Args:
         `hidden_size`: the dimensions of the hidden state
         `vocab_size`: the dimensions of our vocabulary
         `z_size`: the dimensions of the concatenated input
        """
        # Weight matrix (forget gate)
        # YOUR CODE HERE!
        # Size of concatenated hidden + input vector
        self.z_size = self.hidden_size + self.vocab_size

        self.W_f = np.random.randn(self.hidden_size, self.z_size)

        # Bias for forget gate
        self.b_f = np.zeros((self.hidden_size, 1))

        # Weight matrix (input gate)
        # YOUR CODE HERE!
        self.W_i = np.random.randn(self.hidden_size, self.z_size)

        # Bias for input gate
        self.b_i = np.zeros((self.hidden_size, 1))

        # Weight matrix (candidate)
        # YOUR CODE HERE!
        self.W_g = np.random.randn(self.hidden_size, self.z_size)

        # Bias for candidate
        self.b_g = np.zeros((self.hidden_size, 1))

        # Weight matrix of the output gate
        # YOUR CODE HERE!
        self.W_o = np.random.randn(self.hidden_size, self.z_size)
        self.b_o = np.zeros((self.hidden_size, 1))

        # Weight matrix relating the hidden-state to the output
        # YOUR CODE HERE!
        self.W_v = np.random.randn(self.vocab_size, self.hidden_size)
        self.b_v = np.zeros((self.vocab_size, 1))

        # Initialize weights according to https://arxiv.org/abs/1312.6120
        self.W_f = init_orthogonal(self.W_f)
        self.W_i = init_orthogonal(self.W_i)
        self.W_g = init_orthogonal(self.W_g)
        self.W_o = init_orthogonal(self.W_o)
        self.W_v = init_orthogonal(self.W_v)

        return self.W_f, self.W_i, self.W_g, self.W_o, self.W_v, self.b_f, self.b_i, self.b_g, self.b_o, self.b_v


    def forward(self, inputs, h_prev, C_prev, p):
        """
        Arguments:
        x -- your input data at timestep "t", numpy array of shape (n_x, m).
        h_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
        C_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
        p -- python list containing:
                            W_f -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            b_f -- Bias of the forget gate, numpy array of shape (n_a, 1)
                            W_i -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            b_i -- Bias of the update gate, numpy array of shape (n_a, 1)
                            W_g -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                            b_g --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                            W_o -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            b_o --  Bias of the output gate, numpy array of shape (n_a, 1)
                            W_v -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_v, n_a)
                            b_v -- Bias relating the hidden-state to the output, numpy array of shape (n_v, 1)
        Returns:
        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s -- lists of size m containing the computations in each forward pass
        outputs -- prediction at timestep "t", numpy array of shape (n_v, m)
        """
        assert h_prev.shape == (self.hidden_size, 1)
        assert C_prev.shape == (self.hidden_size, 1)

        # First we unpack our parameters
        W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = p

        # Save a list of computations for each of the components in the LSTM
        x_s, z_s, f_s, i_s, = [], [], [], []
        g_s, C_s, o_s, h_s = [], [], [], []
        v_s, output_s = [], []

        # Append the initial cell and hidden state to their respective lists
        h_s.append(h_prev)
        C_s.append(C_prev)

        for x in inputs:
            # YOUR CODE HERE!
            # Concatenate input and hidden state
            z = np.row_stack((h_prev, x))
            z_s.append(z)

            # YOUR CODE HERE!
            # Calculate forget gate
            f = sigmoid(np.dot(W_f, z) + b_f)
            f_s.append(f)

            # Calculate input gate
            i = sigmoid(np.dot(W_i, z) + b_i)
            i_s.append(i)

            # Calculate candidate
            g = tanh(np.dot(W_g, z) + b_g)
            g_s.append(g)

            # YOUR CODE HERE!
            # Calculate memory state
            C_prev = f * C_prev + i * g
            C_s.append(C_prev)

            # Calculate output gate
            o = sigmoid(np.dot(W_o, z) + b_o)
            o_s.append(o)

            # Calculate hidden state
            h_prev = o * tanh(C_prev)
            h_s.append(h_prev)

            # Calculate logits
            v = np.dot(W_v, h_prev) + b_v
            v_s.append(v)

            # Calculate softmax
            output = softmax(v)
            output_s.append(output)

        return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s



    def backward(self, z, f, i, g, C, o, h, v, outputs, targets, p):
        """
        Arguments:
        z -- your concatenated input data  as a list of size m.
        f -- your forget gate computations as a list of size m.
        i -- your input gate computations as a list of size m.
        g -- your candidate computations as a list of size m.
        C -- your Cell states as a list of size m+1.
        o -- your output gate computations as a list of size m.
        h -- your Hidden state computations as a list of size m+1.
        v -- your logit computations as a list of size m.
        outputs -- your outputs as a list of size m.
        targets -- your targets as a list of size m.
        p -- python list containing:
                            W_f -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            b_f -- Bias of the forget gate, numpy array of shape (n_a, 1)
                            W_i -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            b_i -- Bias of the update gate, numpy array of shape (n_a, 1)
                            W_g -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                            b_g --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                            W_o -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            b_o --  Bias of the output gate, numpy array of shape (n_a, 1)
                            W_v -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_v, n_a)
                            b_v -- Bias relating the hidden-state to the output, numpy array of shape (n_v, 1)
        Returns:
        loss -- crossentropy loss for all elements in output
        grads -- lists of gradients of every element in p
        """

        # Unpack parameters
        W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = p

        # Initialize gradients as zero
        W_f_d = np.zeros_like(W_f)
        b_f_d = np.zeros_like(b_f)

        W_i_d = np.zeros_like(W_i)
        b_i_d = np.zeros_like(b_i)

        W_g_d = np.zeros_like(W_g)
        b_g_d = np.zeros_like(b_g)

        W_o_d = np.zeros_like(W_o)
        b_o_d = np.zeros_like(b_o)

        W_v_d = np.zeros_like(W_v)
        b_v_d = np.zeros_like(b_v)

        # Set the next cell and hidden state equal to zero
        dh_next = np.zeros_like(h[0])
        dC_next = np.zeros_like(C[0])

        # Track loss
        loss = 0

        for t in reversed(range(len(outputs))):
            # Compute the cross entropy
            loss += -np.mean(np.log(outputs[t]) * targets[t])
            # Get the previous hidden cell state
            C_prev = C[t - 1]

            # Compute the derivative of the relation of the hidden-state to the output gate
            dv = np.copy(outputs[t])
            dv[np.argmax(targets[t])] -= 1

            # Update the gradient of the relation of the hidden-state to the output gate
            W_v_d += np.dot(dv, h[t].T)
            b_v_d += dv

            # Compute the derivative of the hidden state and output gate
            dh = np.dot(W_v.T, dv)
            dh += dh_next
            do = dh * tanh(C[t])
            do = sigmoid(o[t], derivative=True) * do

            # Update the gradients with respect to the output gate
            W_o_d += np.dot(do, z[t].T)
            b_o_d += do

            # Compute the derivative of the cell state and candidate g
            dC = np.copy(dC_next)
            dC += dh * o[t] * tanh(tanh(C[t]), derivative=True)
            dg = dC * i[t]
            dg = tanh(g[t], derivative=True) * dg

            # Update the gradients with respect to the candidate
            W_g_d += np.dot(dg, z[t].T)
            b_g_d += dg

            # Compute the derivative of the input gate and update its gradients
            di = dC * g[t]
            di = sigmoid(i[t], True) * di
            W_i_d += np.dot(di, z[t].T)
            b_i_d += di

            # Compute the derivative of the forget gate and update its gradients
            df = dC * C_prev
            df = sigmoid(f[t]) * df
            W_f_d += np.dot(df, z[t].T)
            b_f_d += df

            # Compute the derivative of the input and update the gradients of the previous hidden and cell state
            dz = (np.dot(W_f.T, df)
                  + np.dot(W_i.T, di)
                  + np.dot(W_g.T, dg)
                  + np.dot(W_o.T, do))
            dh_prev = dz[:self.hidden_size, :]
            dC_prev = f[t] * dC

        grads = W_f_d, W_i_d, W_g_d, W_o_d, W_v_d, b_f_d, b_i_d, b_g_d, b_o_d, b_v_d

        # Clip gradients
        grads = clip_gradient_norm(grads)

        return loss, grads

    def update_parameters(self, params, grads, lr=1e-3):
        # Take a step
        for param, grad in zip(params, grads):
            param -= lr * grad
        return params