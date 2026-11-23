import torch
from torch import nn

from ldm.data.personalized import per_img_token_list
from ldm.data.dataset_paths import parse_name_anomaly_file
from transformers import CLIPTokenizer
from functools import partial
from ldm.models.vit import VisionTransformer as VIT
from ldm.models.psp_encoder.encoders import psp_encoders
import os
from torchvision.utils import save_image
DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000


def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(
        tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]


def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(
        token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token


def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]


class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            mvtec_path=None,
            return_position=False,
            **kwargs
    ):
        super().__init__()
        self.return_position=return_position
        self.spatial_encoder = False
        self.training_stage = "single"
        self.string_to_token_dict = {}

        self.string_to_param_dict = nn.ParameterDict()
        self.bg_token_embeddings = nn.ParameterDict()
        self.bg_token_k = 0
        self.bg_token_frozen = False

        self.initial_embeddings = nn.ParameterDict()  # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(embedder, 'tokenizer'):  # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            token_dim = 768
        else:  # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280
        self._get_token_for_string = get_token_for_string
        self._get_embedding_for_tkn = get_embedding_for_tkn
        self.token_dim = token_dim
        print(placeholder_strings)  # 是"*"
        print(initializer_words)  # 是一个单词
        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)
        name_anomaly_file = 'name-anomaly.txt'
        sample_anomaly_pairs = [
            f'{sample}+{anomaly}'
            for sample, anomaly in parse_name_anomaly_file(name_anomaly_file)
        ]
        for name in sample_anomaly_pairs:
            for idx, placeholder_string in enumerate(placeholder_strings):

                token = get_token_for_string(placeholder_string)

                if initializer_words and idx < len(initializer_words):
                    init_word_token = get_token_for_string(initializer_words[idx])

                    with torch.no_grad():
                        init_word_embedding = get_embedding_for_tkn(init_word_token.cpu())

                    token_params = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1),
                                                      requires_grad=True)
                    self.initial_embeddings[placeholder_string] = torch.nn.Parameter(
                        init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=False)
                else:
                    token_params = torch.nn.Parameter(
                        torch.rand(size=(num_vectors_per_token, token_dim), requires_grad=True))
                self.string_to_token_dict[placeholder_string] = token
                self.string_to_param_dict[name] = token_params
        self.string_to_param_dict=self.string_to_param_dict.cuda()

    def forward(
            self,
            tokenized_text,
            embedded_text,
            cond_img=None,
            name=None,
            **kwargs
    ):
        img=cond_img
        b, n, device = *tokenized_text.shape, tokenized_text.device
        if name is None:
            # Unconditional branch: return pure text embeddings without
            # per-class TI / bg / PSP injection.
            return embedded_text, None
        # if img is not None:
        #     from torchvision.utils import save_image
        #     print(img.shape,img.min(),img.max())
        #     save_image(img,'tmp.jpg',nrow=4)
        for placeholder_string, placeholder_token in self.string_to_token_dict.items():  # 只有一次，self.string_to_token_dict.items()只有*
            placeholder_embedding=[]
            for i in name:
                placeholder_embedding.append(self.string_to_param_dict[i])
            placeholder_embedding=torch.stack(placeholder_embedding,dim=0)
            if self.max_vectors_per_token == 1:  # If there's only one vector per token, we can do a simple replacement
                placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
                if self.spatial_encoder and img is not None:
                    embedded_text[placeholder_idx] = self.spatial_encoder_model(img)[:, 0]
                else:
                    embedded_text[placeholder_idx] = placeholder_embedding
            else:  # otherwise, need to insert and keep track of changing indices
                if self.progressive_words:
                    self.progressive_counter += 1
                    max_step_tokens = 1 + self.progressive_counter // PROGRESSIVE_SCALE
                else:  # 执行这个
                    max_step_tokens = self.max_vectors_per_token
                # Stage 0: condition on bg tokens only (no class TI, no PSP).
                if str(getattr(self, "training_stage", "")).upper() == "0" \
                        and self.bg_token_k > 0 and len(self.bg_token_embeddings) > 0:
                    placeholder_embedding = self.get_bg_token(
                        name,
                        device=placeholder_embedding.device,
                        dtype=placeholder_embedding.dtype)
                    num_vectors_for_token = placeholder_embedding.shape[1]
                    placeholder_rows, placeholder_cols = torch.where(
                        tokenized_text == placeholder_token.to(device))
                    if placeholder_rows.nelement() == 0:
                        continue
                    sorted_cols, sort_idx = torch.sort(
                        placeholder_cols, descending=True)
                    sorted_rows = placeholder_rows[sort_idx]
                    position = torch.zeros(tokenized_text.size(0), 2)
                    for idx in range(len(sorted_rows)):
                        row = sorted_rows[idx]
                        col = sorted_cols[idx]
                        new_token_row = torch.cat(
                            [tokenized_text[row][:col],
                             placeholder_token.repeat(
                                 num_vectors_for_token).to(device),
                             tokenized_text[row][col + 1:]], dim=0)[:n]
                        new_embed_row = torch.cat(
                            [embedded_text[row][:col],
                             placeholder_embedding[row, :num_vectors_for_token],
                             embedded_text[row][col + 1:]], dim=0)[:n]
                        embedded_text[row] = new_embed_row
                        tokenized_text[row] = new_token_row
                        position[row][0] = col
                        position[row][1] = col + num_vectors_for_token
                    continue
                # Stage-A/external stages: prepend the frozen background
                # token block to [class_TI | PSP].
                if self.bg_token_k > 0 and len(self.bg_token_embeddings) > 0:
                    bg_tok = self.get_bg_token(
                        name,
                        device=placeholder_embedding.device,
                        dtype=placeholder_embedding.dtype)
                    if bg_tok is not None:
                        placeholder_embedding = torch.cat(
                            [bg_tok, placeholder_embedding], dim=1)
                if self.spatial_encoder and img is not None:
                    placeholder_embedding2 = self.spatial_encoder_model(img)
                    placeholder_embedding= torch.cat([placeholder_embedding,placeholder_embedding2],dim=1)
                    num_vectors_for_token = placeholder_embedding.shape[1]
                    # num_vectors_for_token = min(placeholder_embedding.shape[1], max_step_tokens)
                    #print(num_vectors_for_token)
                    placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))
                    # rows对应batchsize：0~batchsize-1;col:对应*在哪个位置
                    if placeholder_rows.nelement() == 0:
                        continue

                    sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                    sorted_rows = placeholder_rows[sort_idx]
                    position=torch.zeros(tokenized_text.size(0),2)
                    for idx in range(len(sorted_rows)):
                        row = sorted_rows[idx]
                        col = sorted_cols[idx]
                        # print(embedded_text[row][:col].shape,num_vectors_for_token, placeholder_embedding[row,:num_vectors_for_token].shape)
                        new_token_row = torch.cat(
                            [tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device),
                             tokenized_text[row][col + 1:]], dim=0)[:n]  # 把*插到77维的text中间
                        new_embed_row = torch.cat(
                            [embedded_text[row][:col], placeholder_embedding[row, :num_vectors_for_token],
                             embedded_text[row][col + 1:]], dim=0)[:n]
                        embedded_text[row] = new_embed_row
                        tokenized_text[row] = new_token_row
                        position[row][0]=col
                        position[row][1]=col+num_vectors_for_token
                else:
                    num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)

                    placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))
                    # rows对应batchsize：0~batchsize-1;col:对应*在哪个位置
                    if placeholder_rows.nelement() == 0:
                        continue

                    sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                    sorted_rows = placeholder_rows[sort_idx]

                    for idx in range(len(sorted_rows)):
                        row = sorted_rows[idx]
                        col = sorted_cols[idx]
                        new_token_row = torch.cat(
                            [tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device),
                             tokenized_text[row][col + 1:]], axis=0)[:n]  # 把*插到77维的text中间
                        new_embed_row = torch.cat(
                            [embedded_text[row][:col], placeholder_embedding[:num_vectors_for_token],
                             embedded_text[row][col + 1:]], axis=0)[:n]

                        embedded_text[row] = new_embed_row
                        tokenized_text[row] = new_token_row
        return embedded_text,position

    def prepare_bg_token_embeddings(self, k_bg=4, init_word="background",
                                    class_names=None):
        """Initialise one K-token background bank per class."""
        k_bg = int(k_bg)
        if k_bg <= 0:
            raise ValueError(f"bg_token_k must be > 0, got {k_bg}")
        token = self._get_token_for_string(init_word)
        # The embedding bank and the token-embedding module can be on different
        # devices during load (main.py prepares bg tokens before Lightning moves
        # the full model to CUDA). Resolve each device separately and then move
        # the initialized token to the bank device.
        try:
            _bank_dev = next(self.string_to_param_dict.parameters()).device
        except StopIteration:
            _bank_dev = torch.device("cpu")
        _embed_dev = torch.device("cpu")
        try:
            if hasattr(self._get_embedding_for_tkn, "weight"):
                _embed_dev = self._get_embedding_for_tkn.weight.device
            else:
                _embed_dev = next(self._get_embedding_for_tkn.parameters()).device
        except (StopIteration, AttributeError):
            pass
        with torch.no_grad():
            init = self._get_embedding_for_tkn(token.to(_embed_dev)).to(_bank_dev)
        names = class_names or list(self.string_to_param_dict.keys())
        for name in names:
            if name in self.bg_token_embeddings:
                continue
            weight = init.unsqueeze(0).repeat(k_bg, 1).clone().detach()
            self.bg_token_embeddings[name] = nn.Parameter(
                weight, requires_grad=True)
        self.bg_token_k = k_bg
        print(f"[BgTokenTI] prepared: K_bg={k_bg}, "
              f"classes={len(self.bg_token_embeddings)}, init_word={init_word!r}")

    def freeze_bg_token_embeddings(self, freeze=True):
        self.bg_token_frozen = bool(freeze)
        for p in self.bg_token_embeddings.parameters():
            p.requires_grad_(not self.bg_token_frozen)
        print(f"[BgTokenTI] frozen={self.bg_token_frozen}")

    def get_bg_token(self, name, device=None, dtype=None):
        """Return [B, K_bg, D] background tokens for class names."""
        if self.bg_token_k <= 0 or len(self.bg_token_embeddings) == 0:
            return None
        if name is None:
            name = list(self.bg_token_embeddings.keys())[:1]
        if isinstance(name, str):
            name = [name]
        bank = []
        any_param = next(iter(self.bg_token_embeddings.values()))
        device = device if device is not None else any_param.device
        dtype = dtype if dtype is not None else any_param.dtype
        for nm in name:
            tok = self.bg_token_embeddings.get(nm, None)
            if tok is None:
                raise KeyError(
                    f"[BgTokenTI] no bg_token for class {nm!r}; available="
                    f"{list(self.bg_token_embeddings.keys())[:8]}")
            bank.append(tok.to(device=device, dtype=dtype))
        return torch.stack(bank, dim=0)

    def prepare_spatial_encoder(self,text_num=4):
        self.spatial_encoder = True
        self.spatial_encoder_model = psp_encoders.GradualStyleEncoder(text_num=text_num).cuda()

    def save(self, ckpt_path):
        save_dict = {
            "string_to_token": self.string_to_token_dict,
            "string_to_param": self.string_to_param_dict,
        }
        if len(self.bg_token_embeddings) > 0:
            save_dict["bg_token_embeddings"] = {
                k: v.detach().cpu() for k, v in self.bg_token_embeddings.items()
            }
            save_dict["bg_token_k"] = int(self.bg_token_k)
        torch.save(save_dict, ckpt_path)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        self.string_to_token_dict = ckpt["string_to_token"]
        tmp = ckpt['string_to_param']
        self.string_to_param_dict = tmp.cuda()
        if 'bg_token_embeddings' in ckpt:
            k_bg = int(ckpt.get('bg_token_k', 0))
            if k_bg > 0 and len(self.bg_token_embeddings) == 0:
                self.prepare_bg_token_embeddings(
                    k_bg=k_bg,
                    class_names=list(ckpt['bg_token_embeddings'].keys()))
            loaded = 0
            for k, v in ckpt['bg_token_embeddings'].items():
                if k in self.bg_token_embeddings:
                    cur = self.bg_token_embeddings[k]
                    cur.data.copy_(v.to(device=cur.device, dtype=cur.dtype))
                    loaded += 1
            print(f'[BgTokenTI] loaded {loaded}/{len(ckpt["bg_token_embeddings"])} '
                  f'from {ckpt_path}')

    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0)  # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)  # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        params = list(self.string_to_param_dict.parameters())
        if self.bg_token_k > 0 and not self.bg_token_frozen:
            params += list(self.bg_token_embeddings.parameters())
        return params

    def embedding_to_coarse_loss(self):

        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss