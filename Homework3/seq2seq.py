import numpy as np
import pandas as pd
import random 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)




class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size, 
                           num_layers=num_layers, 
                           batch_first=True,
                           dropout = 0.35)
    
    def forward(self, x):
        outputs, (hidden, cell) = self.rnn(x)
        return outputs, hidden, cell
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(2 * hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        # Ajustar el tamaño de hidden para que tenga la forma (batch_size, seq_len, hidden_size)
        hidden = hidden[-1].unsqueeze(1)  # Solo la capa superior de los estados ocultos
        hidden = hidden.repeat(1, encoder_outputs.size(1), 1)  # (batch_size, seq_len, hidden_size)
        
        # Concatenar hidden con encoder_outputs
        energy = torch.cat((hidden, encoder_outputs), dim=2)  # (batch_size, seq_len, 2 * hidden_size)
        attention = self.attention(energy).squeeze(2)  # (batch_size, seq_len)
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=output_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.35)
        # Capa de salida
        self.fc_out = nn.Linear(hidden_size*2, output_size)
        self.attention = Attention(hidden_size)
        
    def forward(self, input_step, hidden, cell, encoder_outputs):
        output, (hidden, cell) = self.lstm(input_step, (hidden, cell))
        output = output.squeeze(1)
        
        attention_weights = self.attention(hidden, encoder_outputs)
        attention_weights = attention_weights.unsqueeze(1)
        
        context_vector = torch.bmm(attention_weights, encoder_outputs).squeeze(1)
        
        output = torch.cat((output, context_vector), 1)
        output = self.fc_out(output)
        
        return output, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, x, y=None, teacher_forcing_ratio=0.5):
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        output_size = self.decoder.fc_out.out_features
        
        encoder_outputs, hidden, cell = self.encoder(x)
        outputs = torch.zeros(batch_size, 5, output_size).to(self.device)
        
        input_step = torch.zeros(batch_size, 1, output_size).to(self.device)
        
        for t in range(5):
            output, hidden, cell = self.decoder(input_step, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            # Decidir si se usará el valor real como entrada para el siguiente paso
            use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False
            if use_teacher_forcing and y is not None:
                input_step = y[:, t, :].unsqueeze(1)  # Usar el valor real
            else:
                input_step = output.unsqueeze(1)  # Usar la predicción generada
            
        return outputs
    
def train_seq2seq(model, dataloader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            
            # Convertir a torch.float32
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            y_batch = y_batch.unsqueeze(-1)
            
            # Forward pass
            output = model(x_batch, y_batch, teacher_forcing_ratio=0.5)
            
            # Calcular la pérdida
            loss = criterion(output, y_batch)
            
            # Backward y optimización
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')