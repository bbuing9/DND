import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_backbone(name, output_attentions=False):
    if name == 'bert':
        from transformers import BertForMaskedLM, BertTokenizer
        backbone = BertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=output_attentions)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.name = 'bert-base-uncased'
    elif name == 'roberta':
        from transformers import RobertaForMaskedLM, RobertaTokenizer
        backbone = RobertaForMaskedLM.from_pretrained('roberta-base', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer.name = 'roberta-base'
    elif name == 'roberta_large':
        from transformers import RobertaForMaskedLM, RobertaTokenizer
        backbone = RobertaForMaskedLM.from_pretrained('roberta-large', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        tokenizer.name = 'roberta-large'
    elif name == 'albert':
        from transformers import AlbertModel, AlbertTokenizer
        backbone = AlbertModel.from_pretrained('albert-base-v2', output_attentions=output_attentions)
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        tokenizer.name = 'albert-base-v2'
    else:
        raise ValueError('No matching backbone network')

    return backbone, tokenizer

class Classifier(nn.Module):
    def __init__(self, backbone_name, backbone, n_classes, train_type='None'):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.dropout = nn.Dropout(0.1)
        self.n_classes = n_classes
        self.train_type = train_type

        if 'large' in backbone_name:
            n_dim = 1024
        else:
            n_dim = 768

        ##### Classifier for down-stream task
        self.net_cls = nn.Linear(n_dim, n_classes)

        ##### Classifier for measuring a similarity between sentences
        self.net_sent = nn.Sequential(nn.Linear(3 * n_dim, n_dim), nn.ReLU(), nn.Linear(n_dim, 2))

    def forward(self, x, inputs_embed=None, get_embeds=False):
        if self.backbone_name in ['bert', 'albert']:
            attention_mask = (x > 0).float()
        else:
            attention_mask = (x != 1).float()

        if get_embeds:
            if inputs_embed is None:
                cls_outputs, sentence_outputs = self.backbone(x, attention_mask)[1:]
            else:
                cls_outputs, sentence_outputs = self.backbone(None, attention_mask, inputs_embeds=inputs_embed)[1:]

            out_cls = self.dropout(cls_outputs)
            out_cls = self.net_cls(out_cls)

            return out_cls, sentence_outputs
        else:
            if inputs_embed is not None:
                out_cls_orig = self.backbone(None, attention_mask, inputs_embeds=inputs_embed)[1]
            else:
                out_cls_orig = self.backbone(x, attention_mask, inputs_embeds=inputs_embed)[1]

            out_cls = self.dropout(out_cls_orig)
            out_cls = self.net_cls(out_cls)

            return out_cls



