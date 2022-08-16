# coding=utf-8
import torch
import torch.nn as nn
# from .zen import ZenModel




class CLSModel(nn.Module):
    def __init__(self, encoder_class, encoder_path, num_labels):
        super(CLSModel, self).__init__()

        self.encoder = encoder_class.from_pretrained(encoder_path)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_hidden_states=True,
        input_ngram_ids=None,
        ngram_position_matrix=None,
        ngram_token_type_ids=None,
        ngram_attention_mask=None
    ):

        outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                   output_hidden_states=output_hidden_states)

        # batch, seq, hidden
        last_hidden_states, first_hidden_states = outputs[0], outputs[2][0]
        # batch, hidden
        avg_hidden_states = torch.mean((last_hidden_states + first_hidden_states), dim=1)
        avg_hidden_states = self.dropout(avg_hidden_states)
        logits = self.classifier(avg_hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits

        return logits