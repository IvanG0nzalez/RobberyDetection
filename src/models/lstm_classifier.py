import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_layers=1, bidirectional=True, dropout_fc=0.5):
        super(LSTMClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_fc = dropout_fc

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional)
        
        direction_factor = 2 if bidirectional else 1
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * direction_factor, 128), #primera capa densa
            nn.ReLU(),
            nn.Dropout(self.dropout_fc), #capa de dropout, ayuda contra overfitting
            nn.Linear(128, 1), #salida a una sola neurona
            nn.Sigmoid()
        )

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        if self.bidirectional:
            # Para LSTM bidireccional: concatenar forward y backward
            # hn shape: [num_layers * num_directions, batch, hidden_size]
            # Necesitamos: [batch, hidden_size * 2]
            forward_hidden = hn[-2]  # forward direction del último layer
            backward_hidden = hn[-1]  # backward direction del último layer
            out = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            # Para LSTM unidireccional: solo el último hidden state
            out = hn[-1]
            
        out = self.fc_layers(out)
        return out