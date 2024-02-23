import re
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class EchoParser(nn.Module):
    def __init__(self, tokenizer, templates, max_length=None):
        super(EchoParser, self).__init__()
        self.tokenizer = tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, add_bos_token=False, add_eos_token=False)
        if self.tokenizer.padding_side != 'right':
            self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '<unk>'
        self.templates = templates
        self.template_pieces = {k: self._parse_template(template) for k, template in templates.items()}
        self.max_length = max_length

    def _parse_template(self, template):
        matches = [m for m in re.finditer(r'\{(.|\n|\r|\t)+?\}', template)]
        template_pieces = []
        for i, m in enumerate(matches):
            if i == 0:
                template_pieces.append(template[:m.start()])
            else:
                template_pieces.append(template[matches[i - 1].end():m.start()])
            template_pieces.append(m.group())
        template_pieces.append(template[matches[-1].end():])
        template_pieces = [t for t in template_pieces if t]
        tokenized_pieces = []
        for template_piece in template_pieces:
            if template_piece.startswith('{') and template_piece.endswith('}'):
                tokenized_pieces.append(template_piece[1:-1])
            else:
                tokenized_pieces.append(self.tokenizer(template_piece)['input_ids'])
        return tokenized_pieces

    def _tokenize_piece(self, x, template_piece):
        if isinstance(template_piece, str):
            if template_piece.startswith('!'):
                template_piece = template_piece[1:]
                embed_mask_value = 0
            else:
                embed_mask_value = 1
            for k, v in x.items():
                template_piece = template_piece.replace(f'%%{k}%%', v)
            tokenized_piece = self.tokenizer(template_piece)['input_ids']
            tokenized_piece = tokenized_piece[:self.max_length] if self.max_length is not None else tokenized_piece
            attention_mask = torch.ones(len(tokenized_piece), dtype=torch.long)
            embed_mask = torch.full((len(tokenized_piece),), embed_mask_value, dtype=torch.long)
            return {
                'input_ids': torch.tensor(tokenized_piece, dtype=torch.long),
                'attention_mask': attention_mask,
                'embed_mask': embed_mask,
            }
        else:
            template_piece = template_piece[:self.max_length] if self.max_length is not None else template_piece
            attention_mask = torch.ones(len(template_piece), dtype=torch.long)
            embed_mask = torch.zeros(len(template_piece), dtype=torch.long)
            return {
                'input_ids': torch.tensor(template_piece, dtype=torch.long),
                'attention_mask': attention_mask,
                'embed_mask': embed_mask,
            }

    def _tokenize_from_pieces(self, x, template_pieces):
        token_pieces = [self._tokenize_piece(x, template_piece) for template_piece in template_pieces]
        tokens = {
            k: torch.cat([z[k] for z in token_pieces]) for k in token_pieces[0]
        }
        return tokens

    def tokenize(self, xs):
        # reminder: xs should ideally be a dict of str -> tuple(type, x)
        if isinstance(xs, tuple) and isinstance(xs[1], str):
            xs = [(xs[0], {'x': xs[1]})]
        elif isinstance(xs, tuple) and isinstance(xs[1], dict):
            xs = [xs]
        elif isinstance(xs[0], tuple) and isinstance(xs[0][1], str):
            xs = [(x[0], {'x': x[1]}) for x in xs]
        tokenized = [self._tokenize_from_pieces(x[1], self.template_pieces[x[0]]) for x in xs]
        max_tokenized_length = max([len(x['input_ids']) for x in tokenized])
        for x in tokenized:
            if len(x['input_ids']) < max_tokenized_length:
                x['input_ids'] = torch.cat([x['input_ids'], torch.tensor([self.tokenizer.pad_token_id] * (max_tokenized_length - len(x['input_ids'])), dtype=torch.long)])
                x['attention_mask'] = torch.cat([x['attention_mask'], torch.zeros(max_tokenized_length - len(x['attention_mask']), dtype=torch.long)])
                x['embed_mask'] = torch.cat([x['embed_mask'], torch.zeros(max_tokenized_length - len(x['embed_mask']), dtype=torch.long)])
        tokenized = {
            k: torch.stack([z[k] for z in tokenized]) for k in tokenized[0]
        }
        return tokenized

    def __call__(self, xs):
        tokens = self.tokenize(xs)
        return tokens

    def get_tokenizer(self):
        return self.tokenizer


class EchoPooling(nn.Module):
    def __init__(self, strategy='mean'):
        super(EchoPooling, self).__init__()
        self.strategy = strategy
    
    def forward(self, xs):
        token_embeddings = xs['token_embeddings']
        embed_mask = xs['embed_mask'].to(token_embeddings.device)        
        if self.strategy == 'mean':
            pooled = torch.sum(token_embeddings * embed_mask.unsqueeze(-1), dim=1) / torch.sum(embed_mask, dim=1).unsqueeze(-1)
            pooled.masked_fill_(torch.isnan(pooled), 0)
        elif self.strategy == 'last':
            def _extract_last_nonzero(m):
                nonzeros = (m == 1).nonzero(as_tuple=True)[0]
                return torch.max(nonzeros) if nonzeros.size(0) > 0 else 0
            last_indices = torch.tensor([_extract_last_nonzero(m) for m in embed_mask])
            i = torch.arange(token_embeddings.shape[0]).reshape(token_embeddings.shape[0], 1, 1)
            j = last_indices.reshape(last_indices.shape[0], 1, 1)
            k = torch.arange(token_embeddings.shape[2])
            pooled = token_embeddings[i, j, k][:, 0, :]
            pooled.masked_fill_(torch.isnan(pooled), 0)
        else:
            raise ValueError(f'Unknown pooling strategy: {self.strategy}')
        xs.update({
            'sentence_embedding': pooled,
        })
        return xs


class EchoEmbeddingsMistral(nn.Module):
    def __init__(self, model, parser=None, pooling=None):
        super(EchoEmbeddingsMistral, self).__init__()
        self.model = model
        self.parser = parser
        self.pooling = pooling

    def forward(self, xs):
        inputs = {
            'input_ids': xs['input_ids'].to(self.model.device),
            'attention_mask': xs['attention_mask'].to(self.model.device),
        }
        outputs = self.model(**inputs).last_hidden_state
        xs.update({
            'token_embeddings': outputs,
        })
        return xs

    @staticmethod
    def from_pretrained(base_model_path, **kwargs):
        base_model = AutoModel.from_pretrained(base_model_path, **kwargs)
        return EchoEmbeddingsMistral(base_model)
