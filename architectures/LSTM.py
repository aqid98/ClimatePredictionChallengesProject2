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
    
    
class LSTM_depth_rnn_spatial(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(LSTM_depth_rnn_spatial, self).__init__()
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
        
        
        self.fc = nn.LazyLinear(output_dim)
        
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
    
    
class LSTM_depth_rnn_temporal(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(LSTM_depth_rnn_temporal, self).__init__()
        self.output_size = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = input_dim, 
                            hidden_size = hidden_dim, 
                            num_layers = n_layers, 
                            batch_first=True)
        
        self.depth_lstm = nn.LSTM(input_size = hidden_dim,
                                  hidden_size = hidden_dim,
                                  num_layers = n_layers,
                                  bidirectional = True,
                                  batch_first = True)
        
        
        self.fc = nn.LazyLinear(output_dim)
        
    def forward(self, x):
        batch, depth, seq_len, feat_dim = x.shape
        
        x = x.reshape(batch * depth, seq_len, feat_dim)
        lstm_out, hidden = self.lstm(x)
        lstm_out = lstm_out.reshape(batch, depth, seq_len, self.hidden_dim)
        lstm_out = lstm_out.permute(0, 2, 1, 3)
        
        
        lstm_out = lstm_out.reshape(batch * seq_len, depth, self.hidden_dim)
        depth_out, hid_depth = self.depth_lstm(lstm_out)
        
        out = self.fc(depth_out)
        out = out.reshape(batch, seq_len, depth, 1)
        out = out.permute(0,2,1,3).squeeze()
        
        
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
        mask = torch.triu(torch.ones([seq_len, seq_len], dtype=torch.bool), 
                          diagonal=1).to(x.device)
        #mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        attn_output, attn_weights = self.attention(query = lstm_out, 
                                      key = lstm_out, 
                                      value = lstm_out,
                                      attn_mask = mask)
        out = self.fc(attn_output)
        out = out.reshape(batch, depth, seq_len, 1).squeeze()
        return out
class LSTM_depth_rnn_spatial_attn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(LSTM_depth_rnn_spatial_attn, self).__init__()
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
        
        self.attention = nn.MultiheadAttention(embed_dim = hidden_dim, 
                                               num_heads = 1, 
                                               batch_first = True)
        self.fc = nn.LazyLinear(output_dim)
        
    def forward(self, x):
        batch, depth, seq_len, feat_dim = x.shape
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch * seq_len, depth, feat_dim)
        
        depth_out, hid_depth = self.depth_lstm(x)
        depth_out = depth_out.reshape(batch, seq_len, depth, self.hidden_dim * 2)
        depth_out = depth_out.permute(0, 2, 1, 3)
        
        depth_out = depth_out.reshape(batch * depth, seq_len, self.hidden_dim * 2)
        lstm_out, hidden = self.lstm(depth_out)
        mask = torch.triu(torch.ones([seq_len, seq_len], dtype=torch.bool), 
                          diagonal=1).to(x.device)
        attn_output, attn_weights = self.attention(query = lstm_out, 
                                      key = lstm_out, 
                                      value = lstm_out,
                                      attn_mask = mask)
        
        out = self.fc(attn_output)
        out = out.reshape(batch, depth, seq_len, 1).squeeze()
        return out
