import torch, math
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Positional Encoding
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

# Model with Transformer Encoder
class ForecastingModel(nn.Module):
    def __init__(self, 
                 seq_len=100,  # Ajuste para tus secuencias de 100 pasos
                 embed_size=16,
                 nhead=4,
                 dim_feedforward=2048,
                 dropout=0.1,
                 conv1d_emb=True,
                 conv1d_kernel_size=3,
                 device="cuda"):
        super(ForecastingModel, self).__init__()

        # Set Class-level Parameters
        self.device = device
        self.conv1d_emb = conv1d_emb
        self.conv1d_kernel_size = conv1d_kernel_size
        self.seq_len = seq_len
        self.embed_size = embed_size

        # Input Embedding Component
        if conv1d_emb:
            if conv1d_kernel_size % 2 == 0:
                raise Exception("conv1d_kernel_size must be an odd number to preserve dimensions.")
            self.conv1d_padding = conv1d_kernel_size - 1
            self.input_embedding = nn.Conv1d(5, embed_size, kernel_size=conv1d_kernel_size)  # Ajuste para 5 características de entrada
        else: 
            self.input_embedding = nn.Linear(5, embed_size)  # 5 características de entrada

        # Positional Encoder Component
        self.position_encoder = PositionalEncoding(d_model=embed_size, dropout=dropout, max_len=seq_len)
        
        # Transformer Encoder Layer Component
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)  # Definir el número de capas

        # Regression Component
        self.linear1 = nn.Linear(seq_len * embed_size, int(dim_feedforward))
        self.linear2 = nn.Linear(int(dim_feedforward), int(dim_feedforward / 2))
        self.linear3 = nn.Linear(int(dim_feedforward / 2), int(dim_feedforward / 4))
        self.linear4 = nn.Linear(int(dim_feedforward / 4), int(dim_feedforward / 16))
        self.linear5 = nn.Linear(int(dim_feedforward / 16), 1)

        # Basic Components
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # Model Forward Pass
    def forward(self, x):
        src_mask = self._generate_square_subsequent_mask()
        src_mask = src_mask.to(self.device)

        # Conv1D Embedding
        if self.conv1d_emb:
            x = F.pad(x, (0, 0, self.conv1d_padding, 0), "constant", -1)
            x = self.input_embedding(x.transpose(1, 2))
            x = x.transpose(1, 2)
        else: 
            x = self.input_embedding(x)
            
        # Positional Encoding
        x = self.position_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x, src_mask=src_mask)

        # Flatten and pass through regression layers
        x = x.reshape((-1, self.seq_len * self.embed_size))
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear5(x)
        return x
    
    # Function to create source mask
    def _generate_square_subsequent_mask(self):
        return torch.triu(torch.full((self.seq_len, self.seq_len), float('-inf'), device=self.device), diagonal=1)

# Instantiate and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ForecastingModel(device=device).to(device)

# Now you can use your dataloaders and training loop as before
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Example training function
def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            
            # Forward pass
            output = model(x_batch)
            
            # Compute loss
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

# You can now create your DataLoader and train the model



class ForecastingModel(torch.nn.Module):
    def __init__(self, 
                 seq_len=95,     # Adjusted input sequence length to 95
                 pred_len=5,      # Number of future steps to predict
                 embed_size = 16,
                 nhead = 4,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                 conv1d_emb = True,
                 conv1d_kernel_size = 3,
                 device = "cuda"):
        super(ForecastingModel, self).__init__()

        # Set Class-level Parameters
        self.device = device
        self.conv1d_emb = conv1d_emb
        self.conv1d_kernel_size = conv1d_kernel_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed_size = embed_size

        # Input Embedding Component
        if conv1d_emb:
            if conv1d_kernel_size % 2 == 0:
                raise Exception("conv1d_kernel_size must be an odd number to preserve dimensions.")
            self.conv1d_padding = conv1d_kernel_size - 1
            self.input_embedding  = nn.Conv1d(1, embed_size, kernel_size=conv1d_kernel_size)
        else: 
            self.input_embedding  = nn.Linear(1, embed_size)

        # Positional Encoder Component
        self.position_encoder = PositionalEncoding(d_model=embed_size, 
                                                   dropout=dropout,
                                                   max_len=seq_len)

        # Transformer Encoder Layer Component
        self.transformer_encoder = TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Linear layers for regression
        self.linear1 = nn.Linear(seq_len * embed_size, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, int(dim_feedforward / 2))
        self.linear3 = nn.Linear(int(dim_feedforward / 2), int(dim_feedforward / 4))
        self.linear4 = nn.Linear(int(dim_feedforward / 4), self.pred_len)  # Predict 5 values directly

        # Basic Components
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # Model Forward Pass
    def forward(self, x):
        batch_size = x.size(0)
        
        # Generate a mask for the transformer
        src_mask = self._generate_square_subsequent_mask(self.seq_len).to(self.device)
        
        # Apply input embedding
        if self.conv1d_emb: 
            x = F.pad(x, (0, 0, self.conv1d_padding, 0), "constant", -1)
            x = self.input_embedding(x.transpose(1, 2))  # Conv1d expects (batch, channels, seq_len)
            x = x.transpose(1, 2)  # Return to (batch, seq_len, embed_size)
        else: 
            x = self.input_embedding(x)  # Linear layer expects (batch, seq_len, embed_size)

        # Add positional encoding
        x = self.position_encoder(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x, src_mask=src_mask)
        
        # Flatten the sequence and embed dimensions
        x = x.reshape(batch_size, -1)  # (batch_size, seq_len * embed_size)

        # Pass through the linear layers
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Final output layer to predict 5 future values
        x = self.linear4(x)  # Output shape: (batch_size, 5)

        return x
    
    # Function to create upper-triangular source mask
    def _generate_square_subsequent_mask(self, size):
        return torch.triu(torch.full((size, size), float('-inf')), diagonal=1)
