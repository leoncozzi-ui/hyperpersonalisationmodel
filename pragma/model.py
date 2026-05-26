import math
import torch
import torch.nn as nn
from typing import Optional, List

class ContinuousTimeEmbedder(nn.Module):
    """Continuous time embedder using sinusoidal projections of log-seconds elapsed."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Generate division term for sinusoidal frequencies
        self.register_buffer(
            "div_term", 
            torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        )
        
    def forward(self, times: torch.Tensor) -> torch.Tensor:
        # times shape: [batch_size, seq_len]
        if times.dim() == 2:
            times = times.unsqueeze(-1) # [batch_size, seq_len, 1]
            
        batch_size, seq_len, _ = times.shape
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=times.device)
        
        # Compute sine and cosine projections
        pe[..., 0::2] = torch.sin(times * self.div_term)
        pe[..., 1::2] = torch.cos(times * self.div_term)
        return pe

class CalendarEmbedder(nn.Module):
    """Embeds cyclical hour-of-day, day-of-week, and day-of-month using periodic projections and an MLP."""
    def __init__(self, d_model: int):
        super().__init__()
        # Input features: 6 elements (sin/cos for hour, weekday, day)
        self.mlp = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, calendar_features: torch.Tensor) -> torch.Tensor:
        # calendar_features shape: [batch_size, num_events, 3] (hour, weekday, day)
        # Convert to sine/cosine radians
        h, w, d = calendar_features[..., 0], calendar_features[..., 1], calendar_features[..., 2]
        
        h_rad = h * (2 * math.pi / 24.0)
        w_rad = w * (2 * math.pi / 7.0)
        d_rad = (d - 1) * (2 * math.pi / 31.0) # day of month is 1-indexed
        
        periodic = torch.stack([
            torch.sin(h_rad), torch.cos(h_rad),
            torch.sin(w_rad), torch.cos(w_rad),
            torch.sin(d_rad), torch.cos(d_rad)
        ], dim=-1) # [batch_size, num_events, 6]
        
        return self.mlp(periodic) # [batch_size, num_events, d_model]

class PragmaModel(nn.Module):
    def __init__(
        self,
        key_vocab_size: int,
        value_vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_encoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        num_classes: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        
        # Unified Embedding Tables
        self.key_embed = nn.Embedding(key_vocab_size, d_model)
        self.value_embed = nn.Embedding(value_vocab_size, d_model)
        self.word_pos_embed = nn.Embedding(32, d_model) # Up to 32 words in free-text fields
        
        # Special classification tokens
        self.usr_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.evt_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Encoders
        self.time_embedder = ContinuousTimeEmbedder(d_model)
        self.calendar_embedder = CalendarEmbedder(d_model)
        
        # Profile Encoder
        profile_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True, norm_first=True
        )
        self.profile_encoder = nn.TransformerEncoder(profile_layer, num_layers=num_encoder_layers)
        
        # Event Encoder
        event_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True, norm_first=True
        )
        self.event_encoder = nn.TransformerEncoder(event_layer, num_layers=num_encoder_layers)
        
        # History Encoder
        history_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True, norm_first=True
        )
        self.history_encoder = nn.TransformerEncoder(history_layer, num_layers=num_encoder_layers)
        
        # Downstream next best action classification head
        self.nba_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(
        self,
        profile_keys: torch.Tensor,        # [batch_size, profile_len]
        profile_values: torch.Tensor,      # [batch_size, profile_len]
        profile_times: torch.Tensor,       # [batch_size, profile_len]
        event_keys: torch.Tensor,          # [batch_size, num_events, max_event_tokens]
        event_values: torch.Tensor,        # [batch_size, num_events, max_event_tokens]
        event_times: torch.Tensor,         # [batch_size, num_events]
        event_calendar: torch.Tensor       # [batch_size, num_events, 3]
    ) -> torch.Tensor:
        batch_size = profile_keys.shape[0]
        
        # --- 1. PROFILE STATE ENCODING ---
        # Embed keys and values
        prof_k_emb = self.key_embed(profile_keys)     # [batch_size, profile_len, d_model]
        prof_v_emb = self.value_embed(profile_values) # [batch_size, profile_len, d_model]
        
        # Sum embeddings
        prof_emb = prof_k_emb + prof_v_emb
        
        # Add continuous time positional embeddings for profile lifelong milestones
        prof_t_emb = self.time_embedder(profile_times) # [batch_size, profile_len, d_model]
        prof_emb = prof_emb + prof_t_emb
        
        # Prepend learnable [USR] token
        usr_tokens = self.usr_token.expand(batch_size, 1, -1) # [batch_size, 1, d_model]
        prof_seq = torch.cat([usr_tokens, prof_emb], dim=1)    # [batch_size, 1 + profile_len, d_model]
        
        # Process with Profile State Encoder
        prof_encoded = self.profile_encoder(prof_seq) # [batch_size, 1 + profile_len, d_model]
        
        # Extract the [USR] output representation (za)
        za = prof_encoded[:, 0, :].unsqueeze(1) # [batch_size, 1, d_model]
        
        # --- 2. EVENT ENCODING ---
        num_events = event_keys.shape[1]
        max_tokens = event_keys.shape[2]
        
        # Reshape to process all events in parallel across batch
        # Shape: [batch_size * num_events, max_tokens]
        evt_keys_flat = event_keys.view(batch_size * num_events, max_tokens)
        evt_vals_flat = event_values.view(batch_size * num_events, max_tokens)
        
        evt_k_emb = self.key_embed(evt_keys_flat) # [B * N, T, d_model]
        evt_v_emb = self.value_embed(evt_vals_flat) # [B * N, T, d_model]
        
        # Create positional encodings for multi-valued word tokens inside fields
        positions = torch.arange(max_tokens, device=event_keys.device).unsqueeze(0) # [1, T]
        pos_emb = self.word_pos_embed(positions) # [1, T, d_model]
        
        # Sum them up
        evt_emb = evt_k_emb + evt_v_emb + pos_emb
        
        # Prepend learnable [EVT] token to each event
        evt_tokens = self.evt_token.expand(batch_size * num_events, 1, -1) # [B * N, 1, d_model]
        evt_seq = torch.cat([evt_tokens, evt_emb], dim=1) # [B * N, 1 + T, d_model]
        
        # Encode each event independently
        evt_encoded = self.event_encoder(evt_seq) # [B * N, 1 + T, d_model]
        
        # Extract [EVT] representation (z'e)
        z_prime_e_flat = evt_encoded[:, 0, :] # [B * N, d_model]
        z_prime_e = z_prime_e_flat.view(batch_size, num_events, -1) # [batch_size, num_events, d_model]
        
        # Embed cyclical calendar features and add to z'e
        zt = self.calendar_embedder(event_calendar) # [batch_size, num_events, d_model]
        ze = z_prime_e + zt # [batch_size, num_events, d_model]
        
        # --- 3. FUSION HISTORY ENCODING ---
        # Concatenate [USR] token embedding za and [EVT] embeddings ze
        # Sequence shape: [batch_size, 1 + num_events, d_model]
        history_seq = torch.cat([za, ze], dim=1)
        
        # Add temporal coordinates (time distance to evaluation point)
        # For profile position za, temporal coordinate is 0.0
        zeros = torch.zeros(batch_size, 1, device=event_times.device)
        history_times = torch.cat([zeros, event_times], dim=1) # [batch_size, 1 + num_events]
        
        hist_t_emb = self.time_embedder(history_times) # [batch_size, 1 + num_events, d_model]
        history_seq = history_seq + hist_t_emb
        
        # Process with History Encoder
        history_encoded = self.history_encoder(history_seq) # [batch_size, 1 + num_events, d_model]
        
        # Extract ultimate user-level summary representation from index 0 (the [USR] token output)
        zh_0 = history_encoded[:, 0, :] # [batch_size, d_model]
        
        # --- 4. NEXT BEST ACTION PREDICTION ---
        logits = self.nba_head(zh_0) # [batch_size, num_classes]
        return logits
