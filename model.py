import torch
from torch import nn

class MultimodalEncoder(nn.Module):
    def __init__(self, plm_config, plm_size, pvm_size, labels_to_ids):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(plm_size+pvm_size, plm_config.hidden_size),
                                       nn.Dropout(plm_config.hidden_dropout_prob))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=plm_config.hidden_size,
                                                        nhead=plm_config.num_attention_heads,
                                                        dim_feedforward=plm_config.intermediate_size,
                                                        dropout=plm_config.hidden_dropout_prob,
                                                        activation=plm_config.hidden_act,
                                                        layer_norm_eps=plm_config.layer_norm_eps,
                                                        batch_first=True,
                                                        norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=plm_config.num_hidden_layers)
        self.cls_layer = nn.Linear(plm_config.hidden_size, len(labels_to_ids))
    def forward(self, plm_logit, pvm_logit):
        input_logit = torch.cat((plm_logit, pvm_logit), dim=-1)
        logit_embedded = self.embedding(input_logit)
        encoder_logit = self.transformer_encoder(logit_embedded)
        output_logit = self.cls_layer(encoder_logit)
        return output_logit