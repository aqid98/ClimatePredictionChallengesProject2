import torch.nn as nn
import torch


class LSTM_base(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(LSTM_base, self).__init__()
        self.output_size = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = input_dim, 
                            hidden_size = hidden_dim, 
                            num_layers = n_layers, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch, depth, seq_len, feat_dim = x.shape
        
        x = x.reshape(batch * depth, seq_len, feat_dim)
        lstm_out, hidden = self.lstm(x)
        out = self.fc(lstm_out)
        out = out.reshape(batch, depth, seq_len, 1).squeeze()
        return out
    
    
class LSTM_depth_cnn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(LSTM_depth_cnn, self).__init__()
        self.output_size = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = hidden_dim, 
                            hidden_size = hidden_dim, 
                            num_layers = n_layers, 
                            batch_first=True)
        
        self.depth_cnn = nn.Conv1d(in_channels = input_dim, out_channels = hidden_dim, padding = 1, kernel_size = 3)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch, depth, seq_len, feat_dim = x.shape
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch * seq_len, depth, feat_dim)
        
        depth_out = self.depth_cnn(x.permute(0, 2, 1)).transpose(1, 2)
        depth_out = depth_out.reshape(batch, seq_len, depth, self.hidden_dim)
        depth_out = depth_out.permute(0, 2, 1, 3)
        
        depth_out = depth_out.reshape(batch * depth, seq_len, self.hidden_dim)
        lstm_out, hidden = self.lstm(depth_out)
        out = self.fc(lstm_out)
        out = out.reshape(batch, depth, seq_len, 1).squeeze()
        return out
    
class LSTM_depth_rnn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(LSTM_depth_rnn, self).__init__()
        self.output_size = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = hidden_dim * 2, 
                            hidden_size = hidden_dim, 
                            num_layers = n_layers, 
                            batch_first=True)
        
        self.depth_lstm = nn.LSTM(input_size = input_dim,
                                  hidden_size = hidden_dim,
                                  num_layers = n_layers,
                                  bidirectional = True,
                                  batch_first = True)
        
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch, depth, seq_len, feat_dim = x.shape
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch * seq_len, depth, feat_dim)
        
        depth_out, hid_depth = self.depth_lstm(x)
        depth_out = depth_out.reshape(batch, seq_len, depth, self.hidden_dim * 2)
        depth_out = depth_out.permute(0, 2, 1, 3)
        
        depth_out = depth_out.reshape(batch * depth, seq_len, self.hidden_dim * 2)
        lstm_out, hidden = self.lstm(depth_out)
        out = self.fc(lstm_out)
        out = out.reshape(batch, depth, seq_len, 1).squeeze()
        return out
    
class LSTM_attention(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim, 
                 n_layers, 
                 num_head = 1):
        super(LSTM_attention, self).__init__()
        self.output_size = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = input_dim, 
                            hidden_size = hidden_dim, 
                            num_layers = n_layers, 
                            batch_first=True)
        
        self.attention = nn.MultiheadAttention(embed_dim = hidden_dim, 
                                               num_heads = num_head, 
                                               batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch, depth, seq_len, feat_dim = x.shape
        
        x = x.reshape(batch * depth, seq_len, feat_dim)
        lstm_out, hidden = self.lstm(x)
        
        mask = torch.triu(torch.ones([seq_len, seq_len], dtype=torch.bool), diagonal=1)
        #mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        attn_output, attn_weights = self.attention(query = lstm_out, 
                                      key = lstm_out, 
                                      value = lstm_out,
                                      attn_mask = mask)
        
        out = self.fc(attn_output)
        out = out.reshape(batch, depth, seq_len, 1).squeeze()
        return out
    
    
    
class LSTM_encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim, 
                 n_layers, 
                 num_head = 1):
        super(LSTM_encoder, self).__init__()
        self.output_size = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = input_dim, 
                            hidden_size = hidden_dim, 
                            num_layers = n_layers, 
                            batch_first=True)
        
        self.encoder = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead = num_head, 
                                                  dim_feedforward=hidden_dim * 2, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch, depth, seq_len, feat_dim = x.shape
        
        x = x.reshape(batch * depth, seq_len, feat_dim)
        lstm_out, hidden = self.lstm(x)
        
        mask = torch.triu(torch.ones([seq_len, seq_len], dtype=torch.bool), diagonal=1).to(x.device)
        attn_output = self.encoder(lstm_out,
                                      src_mask = mask)
        
        out = self.fc(attn_output)
        out = out.reshape(batch, depth, seq_len, 1).squeeze()
        return out