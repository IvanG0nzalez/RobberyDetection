import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Mecanismo de atención temporal para identificar qué clips son más importantes.
    Calcula un peso de atención para cada clip en la secuencia.
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_outputs):
        """
        Args:
            lstm_outputs: Tensor de shape (batch, seq_len, hidden_size)
        
        Returns:
            context: Tensor de shape (batch, hidden_size) - representación ponderada
            attention_weights: Tensor de shape (batch, seq_len) - pesos de atención
        """
        # Calcular scores de atención para cada timestep
        # shape: (batch, seq_len, 1)
        attention_scores = self.attention(lstm_outputs)
        
        # Aplicar softmax para obtener pesos normalizados
        # shape: (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Aplicar pesos de atención a los outputs de LSTM
        # shape: (batch, seq_len, hidden_size)
        weighted_output = lstm_outputs * attention_weights
        
        # Sumar sobre la dimensión temporal para obtener contexto
        # shape: (batch, hidden_size)
        context = torch.sum(weighted_output, dim=1)
        
        # Squeeze para obtener shape (batch, seq_len)
        attention_weights = attention_weights.squeeze(-1)
        
        return context, attention_weights


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_layers=1, bidirectional=True, dropout_fc=0.5, use_attention=True):
        super(LSTMClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_fc = dropout_fc
        self.use_attention = use_attention

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional)
        
        direction_factor = 2 if bidirectional else 1
        lstm_output_size = hidden_size * direction_factor
        
        # Mecanismo de atención temporal para identificar qué clips son más importantes
        if self.use_attention:
            self.attention = TemporalAttention(lstm_output_size)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, 128),  # primera capa densa
            nn.ReLU(),
            nn.Dropout(self.dropout_fc),  # capa de dropout, ayuda contra overfitting
            nn.Linear(128, 1),  # salida a una sola neurona
            nn.Sigmoid()
        )
        
        # Variable para almacenar pesos de atención (para explicabilidad)
        self.last_attention_weights = None

    def forward(self, x):
        # LSTM procesa toda la secuencia
        # lstm_outputs shape: (batch, seq_len, hidden_size * directions)
        lstm_outputs, (hn, cn) = self.lstm(x)
        
        if self.use_attention:
            # Usar mecanismo de atención para obtener representación ponderada
            context, attention_weights = self.attention(lstm_outputs)
            
            # Guardar pesos de atención para análisis posterior
            self.last_attention_weights = attention_weights.detach()
            
            # Usar el contexto ponderado para clasificación
            out = context
        else:
            # Método tradicional: usar último hidden state
            if self.bidirectional:
                # Para LSTM bidireccional: concatenar forward y backward
                # hn shape: [num_layers * num_directions, batch, hidden_size]
                # Se necesita [batch, hidden_size * 2]
                forward_hidden = hn[-2]  # forward direction del último layer
                backward_hidden = hn[-1]  # backward direction del último layer
                out = torch.cat((forward_hidden, backward_hidden), dim=1)
            else:
                # Para LSTM unidireccional: solo el último hidden state
                out = hn[-1]
            
            self.last_attention_weights = None
            
        out = self.fc_layers(out)
        return out
    
    def get_attention_weights(self):
        """
        Retorna los pesos de atención de la última inferencia.
        Útil para explicabilidad: muestra qué clips fueron más importantes.
        
        Returns:
            Tensor de shape (batch, seq_len) con los pesos de atención,
            o None si no se usa atención o no se ha ejecutado forward aún.
        """
        return self.last_attention_weights