import torch
import torch.nn as nn

from menrot.models.transformer import AutoregTransformer, ViT3d
from menrot.utils.config import load_model_from_config

__all__ = [
    "VisionSymbolicModel"
]

class VisionSymbolicModel(nn.Module):
    def __init__(self, config_path, model_config=None, device=None):
        super().__init__()
        self.device = device
        if model_config is not None:
            self.model_config = model_config
            self.encoder = ViT3d(**self.model_config)
        else:
            self.encoder, self.model_config = load_model_from_config(model_class=ViT3d, config_path=config_path)
        
        self.decoder = AutoregTransformer(
            embed_dim=self.model_config['embed_dim'], 
            heads=self.model_config['heads'], 
            num_layers=3, vocab_size=6, seq_len=9
        )

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)        
      
    def forward(self, x, tgt=None, return_cls=False):
        B = x.size(0)
        memory = self.encoder(x)
        sos_token = torch.full((B, 1), self.decoder.sos_token_id, dtype=torch.long, device=x.device)
        if tgt is not None:
            """
            Training function with teacher forcing
            memory: [B, T, D] output of the ViT
            target:    [B, seq_len] — ground truth tokens (int) (without <SOS>/<EOS> token)
            -> so target we will had sos_token to target and remove the last elem so lenght is 9
            """
            T = tgt.size(1) 
            tgt_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device),
                diagonal=1,
            )   # [T, T]
            
            input_tokens = torch.cat([sos_token, tgt[:, :-1]], dim=1)
            logits = self.decoder(
                input_tokens=input_tokens,      # [B, T] (we shift target by truncatding the last elem)
                memory=memory,                  # [B, D]
                tgt_mask=tgt_mask               # [T, T]
            ) # without <SOS> token [B, seq_len, vocab_size]
            
            if return_cls:
                return logits, memory[:,0,:]
            else:
                return logits 
        else:
            """
            Autoregressive inference mode
            memory: [B, T, D] output from ViT
            """
            generated = sos_token
            for _ in range(self.decoder.seq_len):
                t = generated.size(1)
                tgt_mask = torch.triu(
                    torch.full((t, t), float("-inf"), device=self.device),
                    diagonal=1,
                )
                
                # Forward pass through decoder
                logits = self.decoder(generated, memory=memory, tgt_mask=tgt_mask)  # [B, T, vocab]
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)             # [B, 1]

                # Append predicted token
                generated = torch.cat([generated, next_token], dim=1)  # [B, T+1]
            return generated[:, 1:], logits  # without <SOS> token
