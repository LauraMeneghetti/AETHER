'''Module implementing the Bidirectional LSTM employed and developed.'''


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BILSTMModel(nn.Module):
    '''
    Class that handles the implementation of a BiLSTM for binary classification.
    '''
    def __init__(self, input_size):
        '''
        Initialize the network layers.
        :param torch.Tensor input_size: input tensor
        '''
        super(BILSTMModel, self).__init__()

        self.lstm_layers = nn.ModuleList([
            # Layer 1: Input size is the orginal input size, output size is 2*128
            nn.LSTM(input_size=input_size, hidden_size=128, batch_first=True, bidirectional=True),
            # Layer 2: Input size is the size of the output of Layer 1 (2*128), output size is 2*64
            nn.LSTM(input_size=2*128, hidden_size=64, batch_first=True, bidirectional=True),
            # Layer 3: Input size is the size of the output of Layer 2 (2*64), output size is 2*32
            nn.LSTM(input_size=2*64, hidden_size=32, batch_first=True, bidirectional=True),
        ])
        
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(p=0.4),
            nn.Dropout(p=0.3),
            nn.Dropout(p=0.3),
        ])

        # Classification layer (Fully Connected)
        # Input size is the output of the last layer of the BiLSTM (2*32)
        # Output size è 1 per classificazione binaria
        self.fc = nn.Linear(2 * 32, 1)


    def forward(self, x, seq_lengths):
        """
        Function performing the forward step.
        :param torch.Tensor x: Padded input tensor of shape (batch_size, max_seq_length, input_size)
        :param torch.Tensor seq_lengths: Actual lengths of each sequence before padding
        :return: output forward pass
        :rtype: torch.Tensor
        """
        output = x
        # need to move seq_lenghts to cpu for the pack_padded_sequence function
        seq_lengths_cpu = seq_lengths.cpu()

        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            
            # Pack sequences before passing to BiLSTM
            # use the output of the previous output as input
            packed_output = pack_padded_sequence(
                output, 
                seq_lengths_cpu, 
                batch_first=True, 
                enforce_sorted=True
            )

            # BiLSTM forward pass
            packed_output, _ = lstm(packed_output)
            
            # Unpack sequence to apply dropout
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            
            # 4. Apply Dropout
            output = dropout(output)

        # the final unpacked output
        unpacked_x = output

        # Select the last valid time step (without padding) for each sequence
        # find idx of last valid element 
        # tensor of dim [batch_size, 1, hidden_size * 2]
        # view(-1, 1) -> [batch_size, 1]
        # unsqueeze(2) -> [batch_size, 1, 1]
        idx = (seq_lengths.to(device).long() - 1).view(-1, 1).unsqueeze(2) 
        
        # expand pon the feature dimension (last)
        # unpacked_x.size(2) is hidden_size * 2 for the output of BiLSTM
        idx = idx.expand(unpacked_x.size(0), 1, unpacked_x.size(2)) 
        
        # gather is introduced for selecting the final output
        # last_outputs has dim [batch_size, 1, hidden_size*2] dopo gather
        # squeeze(1) reduced it to [batch_size, hidden_size*2]
        last_outputs = unpacked_x.gather(1, idx).squeeze(1)

        # Fully connected layer
        x = self.fc(last_outputs)
        
        return x
    


