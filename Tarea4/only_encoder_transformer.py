import math
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class ForecastingModel(nn.Module):
    def __init__(self, 
                 seq_len=95,  # Ajuste para tus secuencias de 100 pasos
                 pred_len=5,
                 embed_size=16,
                 nhead=4,
                 dim_feedforward=2048,
                 dropout=0.1,
                 conv1d_emb=True,
                 conv1d_kernel_size=3,
                 device="cuda"):
        super(ForecastingModel, self).__init__()
        
        # Fijar los parámetros
        self.device = device
        self.conv1d_emb = conv1d_emb
        self.conv1d_kernel_size = conv1d_kernel_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed_size = embed_size
        
        # Capa de embedding
        if conv1d_emb:
            if conv1d_kernel_size % 2 == 0:
                raise Exception("conv1d_kernel_size must be an odd number to preserve dimensions.")
            self.conv1d_padding = conv1d_kernel_size - 1
            self.input_embedding = nn.Conv1d(5, 
                                             embed_size, 
                                             kernel_size=conv1d_kernel_size)  # 5 características de entrada
        else: 
            self.input_embedding = nn.Linear(5, 
                                             embed_size)  # 5 características de entrada

        # Capa de positional encoding
        self.position_encoder = PositionalEncoding(d_model=embed_size, 
                                                   dropout=dropout, 
                                                   max_len=seq_len)
        
        # Capa de Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout,
                                                   batch_first=True)
        # Se define el número de capas en el modelo de Transformer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=3)
        
        # Componente de Regresión
        self.linear1 = nn.Linear(seq_len * embed_size, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, int(dim_feedforward / 2))
        self.linear3 = nn.Linear(int(dim_feedforward / 2), int(dim_feedforward / 4))
        self.linear4 = nn.Linear(int(dim_feedforward / 4), self.pred_len)  # Predict 5 values directly


        # Elementos básicos
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, 
                input_ids=None,
                attention_mask: Optional[Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = False,
                output_hidden_states: Optional[bool] = False,
                return_dict: Optional[bool] = False) -> Tensor:
        # Check if input_ids are passed; you may not use it
        if input_ids is not None:
            print("Input IDs are not used in this model.")
            
        
        
        src_mask = self._generate_square_subsequent_mask()
        src_mask = src_mask.to(self.device)
        batch_size = x.size(0)
        
        # Embedding
        if self.conv1d_emb:
            x = F.pad(x, (0, 0, self.conv1d_padding, 0), "constant", -1)
            x = self.input_embedding(x.transpose(1, 2))
            x = x.transpose(1, 2)
        else:
            x = self.input_embedding(x)
        
        # Positional Encoding
        x = self.position_encoder(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x, mask=src_mask)
        
        # Flatten para la capa de regresión
        x = x.reshape(batch_size, -1)
        
        # Regresión
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Salida
        x = self.linear4(x)     
        
        return x
        
    # Function to create source mask
    def _generate_square_subsequent_mask(self):
        return torch.triu(torch.full((self.seq_len, self.seq_len), float('-inf'), device=self.device), diagonal=1)


