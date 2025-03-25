import json
from typing import List
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import transformers
from transformers import CLIPTokenizerFast
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
import torch.distributed as dist
import ipdb
import yaml
import numpy as np
import horovod.torch as hvd

from src.modeling.VidCLIP import VidCLIP
from src.modeling.CLIP import CLIPModel as CLIP
from src.modeling.co_attention_module import Co_attention_block
# from src.modeling.med import BertConfig, BertForMaskedLM, BertEncoder
from src.modeling.Qformer import BertConfig, BertEncoder, BertLMHeadModelForQformer
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter




class FK(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.encoder = VidCLIP(self.cfg)

        if cfg.loss_config.loss_name == 'NCELearnableTempLoss_two_matrix':
            self.alpha = nn.Parameter(torch.tensor(0.5))

        if hasattr(cfg, 'fusioner_config') and cfg.fusioner_config.final_alpha_add:
            self.final_add_alpha = nn.Parameter(torch.tensor(0.1))

        if hasattr(cfg, 'fusioner_config') and (
                cfg.fusioner_config.fuse_text_first or cfg.fusioner_config.text_embeds_type == 'word-level'):
            if cfg.fusioner_config.fuse_text_first:
                assert cfg.fusioner_config.text_embeds_type == 'eot-level'
            # (B, M, D) --> (B, M, 1)
            self.AFA = nn.Sequential(nn.Linear(cfg.fusioner_config.embed_dim, cfg.fusioner_config.embed_dim, bias=True),
                                     nn.SiLU(),
                                     nn.Linear(cfg.fusioner_config.embed_dim, 1, bias=True),
                                     nn.Softmax(dim=-1))

        if hasattr(cfg, 'pseudo_final_alpha_add'):
            self.final_pseudo_final_alpha_add = nn.Parameter(torch.tensor(0.1))

        # self.aux_feature_scale = nn.Parameter(torch.tensor(1.0))
        if hasattr(self.cfg, 'generator_config'):
            self.generator = FeatureReconstructor(self.cfg, self.cfg.generator_config)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip_config)

        if hasattr(self.cfg, 'decoder_config'):
            if self.cfg.decoder_config.decoder_type == 'TransformerDecoder':
                self.caption_prenorm = nn.LayerNorm(
                    self.cfg.decoder_config.hidden_dim
                )
                self.caption_decoder = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(
                        d_model=self.cfg.decoder_config.hidden_dim,
                        nhead=self.cfg.decoder_config.num_heads,
                        dim_feedforward=self.cfg.decoder_config.dim_feedforward,
                        dropout=self.cfg.decoder_config.dropout
                    ),
                    num_layers=cfg.decoder_config.num_layers,
                    norm=nn.LayerNorm(cfg.decoder_config.hidden_dim)
                )

                self.caption_head = nn.Linear(
                    self.cfg.decoder_config.hidden_dim,
                    self.encoder.clipmodel.text_model.config.vocab_size
                )

                self.caption_head.weight.data.copy_(
                    self.encoder.clipmodel.text_model.embeddings.token_embedding.weight.data
                )
            elif self.cfg.decoder_config.decoder_type == 'BLIP':
                blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
                self.caption_prenorm = nn.LayerNorm(
                    self.cfg.decoder_config.hidden_dim
                )
                self.caption_pre_fc = nn.Linear(512, 768, bias=True)
                self.caption_decoder = blip_model.text_decoder.bert.encoder
                del blip_model
                self.caption_post_fc = nn.Linear(768, 512, bias=True)
                self.caption_head = nn.Linear(
                    self.cfg.hidden_dim,
                    self.encoder.clipmodel.text_model.config.vocab_size
                )

                self.caption_head.weight.data.copy_(
                    self.encoder.clipmodel.text_model.embeddings.token_embedding.weight.data
                )

            else:
                raise NotImplementedError

        if hasattr(self.cfg, 'fusioner_config'):
            self.fusioner_type = cfg.fusioner_config.fusioner_type
            if self.fusioner_type == 'mean_pooling':
                self.fusioner = MeanPooler()
            elif self.fusioner_type == 'self_attention':
                self.fusioner = AttentionFusioner(cfg.fusioner_config)
            elif self.fusioner_type == 'XPoolCrossAttention':
                self.fusioner = XPoolCrossAttentionPooler(cfg.fusioner_config)
            elif self.fusioner_type == 'wide_cross_attention':
                self.fusioner = WideCrossAttentionPooler(cfg.fusioner_config)
            elif self.fusioner_type == 'wo_residual_wide_cross_attention':
                self.fusioner = WoResidualWideCrossAttentionPooler(cfg.fusioner_config)
            elif self.fusioner_type == 'ts2-XPoolCrossAttention':
                self.fusioner = TS2XPoolCrossAttentionPooler(cfg.fusioner_config)
            elif self.fusioner_type == 'CoAttention':
                self.fusioner = CoAttentionPooler(cfg.fusioner_config)
            else:
                raise NotImplementedError

        # self.fusioner_vision_proj = nn.Linear(cfg.generator_config.Qformer_hidden_dim, cfg.fusioner_config.embed_dim, bias=True)

    def get_clip_embeds(self, video, text_input_ids, text_input_mask, image=None, caption_ids=None, caption_masks=None):
        inputs = {"input_ids": text_input_ids,
                  "attention_mask": text_input_mask,
                  "pixel_values": video,
                  "output_attentions": True,
                  "output_hidden_states": True,
                  "return_loss": False}
        outputs = self.encoder.clipmodel.forward_wo_norm(**inputs)
        # embeds are global features: [cls] and [sep] tokens
        text_feats = outputs["text_embeds"]
        video_feats = outputs["image_embeds"]

        # 0 is all feats(last_hidden_state)
        text_all_feats = outputs["text_model_output"][0]
        video_all_feats = outputs["vision_model_output"][0]
        if hasattr(self.cfg, 'generation_loss_config') and hasattr(self.cfg.generation_loss_config, 'num_layers'):
            num_layers = self.cfg.generation_loss_config.num_layers
            # do not select last hidden state
            text_hidden_states = outputs["text_model_output"]["hidden_states"][-num_layers:-1]
            # (bs, num_layers, len_seq, hidden_dim)
            text_hidden_states = torch.stack(text_hidden_states, dim=1)
            text_hidden_states = torch.cat([text_hidden_states, text_all_feats.unsqueeze(1)], dim=1)
        else:
            text_hidden_states = text_all_feats

        vision_attentions = outputs['vision_model_output']['attentions']
        return text_feats, video_feats, text_all_feats, video_all_feats, text_hidden_states, vision_attentions

    def forward_captioner(self, video_features, tokens):
        if self.cfg.decoder_config.decoder_type == 'TransformerDecoder':
            # B x N x D -> N x B x D
            video_features = video_features.transpose(0, 1)
            with torch.no_grad():
                token_embeds = self.encoder.clipmodel.text_model.embeddings(tokens)
            # B x N x D -> N x B x D
            token_embeds = token_embeds.transpose(0, 1)

            token_embeds = self.caption_prenorm(token_embeds)

            attn_mask = torch.empty(tokens.size(1), tokens.size(1))
            attn_mask.fill_(float("-inf"))
            attn_mask.triu_(1)
            attn_mask = attn_mask.to(token_embeds.device)

            pred_embeds = self.caption_decoder(
                token_embeds, video_features, tgt_mask=attn_mask)

            # N x B x vocab_size
            pred_logits = self.caption_head(pred_embeds)
            return pred_logits.transpose(0, 1)
        # elif self.cfg.decoder_config.decoder_type == 'BLIP':
        #     with torch.no_grad():
        #         token_embeds = self.clip.text_model.embeddings(tokens)
        #
        #     token_embeds = self.caption_pre_fc(self.caption_prenorm(token_embeds))

        else:
            raise NotImplementedError

    def generate_caption(self, video_features, temperature=0.6, top_p=0.9):
        B, N, D = video_features.size()
        video_features = video_features.transpose(0, 1)
        bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        pad_token = bos_token_id
        # pad_token = self.encoder.clipmodel.text_model.config.pad_token_id
        max_len = self.cfg.max_txt_len

        # init tensor
        # B x N
        tokens = torch.full((B, max_len), pad_token, dtype=torch.long, device=video_features.device)
        tokens[:, 0] = bos_token_id

        eos_reached = torch.tensor([False] * B, device="cuda")

        # generation
        for cur_pos in range(max_len - 1):
            # B x N x vocab_size
            logits = self.forward_captioner(video_features, tokens)
            if temperature > 0:
                # B x 1
                probs = torch.softmax(logits[:, cur_pos] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                # shape in B x 1
                next_token = logits[:, cur_pos].argmax(dim=-1)

            next_token = next_token.squeeze(-1)
            tokens[:, cur_pos + 1] = next_token

            eos_reached |= next_token == eos_token_id
            if all(eos_reached):
                break

        # # post processing
        # out_tokens = []
        # for i, toks in enumerate(tokens.tolist()):
        #     # cut to eos tok if any
        #     if eos_token_id in toks:
        #         eos_idx = toks.index(eos_token_id)
        #         toks = toks[:eos_idx]
        #     out_tokens.append(toks)

        # B x N x 1
        return tokens

    def select_video_tokens(self, cfg, select_type, video_feats, attention_scores=None):
        """
        video_feats:
        [bs, 1+num_global_prompts+num_frames*(num_local_prompts+num_patches), hidden_dim]
        we fix num_local_prompts == 1
        """
        if select_type == 'patch':
            return video_feats
        elif select_type == 'cls+added_cls':
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num

            cls_prompt_tokens = video_feats[:, 0, :].unsqueeze(1)
            global_prompt_tokens = video_feats[:, 1:num_global_prompts + 1, :]

            selected_video_feats = torch.cat([cls_prompt_tokens, global_prompt_tokens], dim=1)

            return selected_video_feats
        elif select_type == 'added_cls+frame_cls':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size
            global_prompt_tokens = video_feats[:, 1:num_global_prompts + 1, :]
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(video_feats.shape[0], num_frames, -1,
                                                                                   video_feats.shape[-1])
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])
            # [bs, num_global_prompts+num_frames*num_local_prompts, hidden_dim]
            selected_video_feats = torch.cat([global_prompt_tokens, local_prompt_tokens], dim=1)

            return selected_video_feats
        elif select_type == 'cls+added_cls+frame_cls':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size
            cls_prompt_tokens = video_feats[:, 0, :]
            cls_prompt_tokens = cls_prompt_tokens.unsqueeze(1)
            global_prompt_tokens = video_feats[:, 1:num_global_prompts + 1, :]
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(video_feats.shape[0], num_frames, -1,
                                                                                   video_feats.shape[-1])
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])
            # [bs, num_global_prompts+num_frames*num_local_prompts, hidden_dim]
            selected_video_feats = torch.cat([cls_prompt_tokens, global_prompt_tokens, local_prompt_tokens], dim=1)

            return selected_video_feats
        elif select_type == 'cls+frame_cls':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size
            cls_prompt_tokens = video_feats[:, 0, :]
            cls_prompt_tokens = cls_prompt_tokens.unsqueeze(1)
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(video_feats.shape[0], num_frames, -1,
                                                                                   video_feats.shape[-1])
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])
            # [bs, num_global_prompts+num_frames*num_local_prompts, hidden_dim]
            selected_video_feats = torch.cat([cls_prompt_tokens, local_prompt_tokens], dim=1)

            return selected_video_feats
        elif select_type == 'frame_cls':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(video_feats.shape[0], num_frames, -1,
                                                                                   video_feats.shape[-1])
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])
            # [bs, num_global_prompts+num_frames*num_local_prompts, hidden_dim]
            selected_video_feats = local_prompt_tokens

            return selected_video_feats
        elif select_type == 'cls+cls_attention_scores':
            assert attention_scores is not None, 'attention scores is None'
            # attention_scores shape in (bs, num_heads, num_queries, num_keys)
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            cls_prompt_tokens = video_feats[:, 0, :]
            cls_prompt_tokens = cls_prompt_tokens.unsqueeze(1)
            # reduce head dim
            if self.cfg.fusioner_config.select_attention_scores_type == 'max':
                attention_scores, _ = attention_scores.max(dim=1)
            elif self.cfg.fusioner_config.select_attention_scores_type == 'mean':
                attention_scores = attention_scores.mean(dim=1)

            # select cls tokens attention distribution at position 0
            attention_scores = attention_scores[:, 0, :]
            k = self.cfg.fusioner_config.num_selected_tokens

            # topk_indices shape in (bs, k)
            topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1, largest=True, sorted=True)
            batch_indices = torch.arange(attention_scores.size(0)).unsqueeze(1).expand(-1, k).to(
                attention_scores.device)
            # [bs, k, hidden_dim]
            selected_video_feats = video_feats[batch_indices, topk_indices, :]

            # [bs, k+1, hidden_dim]
            selected_video_feats = torch.cat([cls_prompt_tokens, selected_video_feats], dim=1)

            return selected_video_feats
        elif select_type == 'cls+added_cls+frame_cls+cls_attention_scores':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size
            cls_prompt_tokens = video_feats[:, 0, :]
            cls_prompt_tokens = cls_prompt_tokens.unsqueeze(1)
            global_prompt_tokens = video_feats[:, 1:num_global_prompts + 1, :]
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(video_feats.shape[0], num_frames, -1,
                                                                                   video_feats.shape[-1])
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])

            # attention_scores shape in (bs, num_heads, num_queries, num_keys)
            assert attention_scores is not None, 'attention scores is None'

            # reduce head dim
            if self.cfg.fusioner_config.select_attention_scores_type == 'max':
                attention_scores, _ = attention_scores.max(dim=1)
            elif self.cfg.fusioner_config.select_attention_scores_type == 'mean':
                attention_scores = attention_scores.mean(dim=1)

            # select cls tokens attention distribution at position 0
            attention_scores = attention_scores[:, 0, :]
            k = self.cfg.fusioner_config.num_selected_tokens

            # topk_indices shape in (bs, k)
            topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1, largest=True, sorted=True)
            batch_indices = torch.arange(attention_scores.size(0)).unsqueeze(1).expand(-1, k).to(
                attention_scores.device)
            # [bs, k, hidden_dim]
            attention_tokens = video_feats[batch_indices, topk_indices, :]

            selected_video_feats = torch.cat(
                [cls_prompt_tokens, global_prompt_tokens, local_prompt_tokens, attention_tokens], dim=1)
            return selected_video_feats
        elif select_type == 'cls+added_cls+frame_cls+cls_attention_scores_no_overlap':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size

            cls_prompt_tokens = video_feats[:, 0, :].unsqueeze(1)
            global_prompt_tokens = video_feats[:, 1:num_global_prompts + 1, :]
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(
                video_feats.shape[0], num_frames, -1, video_feats.shape[-1]
            )
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])

            # attention_scores: shape = (bs, num_heads, num_queries, num_keys)
            assert attention_scores is not None, 'attention scores is None'

            if self.cfg.fusioner_config.select_attention_scores_type == 'max':
                attention_scores, _ = attention_scores.max(dim=1)  # [bs, num_queries, num_keys]
            elif self.cfg.fusioner_config.select_attention_scores_type == 'mean':
                attention_scores = attention_scores.mean(dim=1)  # [bs, num_queries, num_keys]

            # select cls tokens attention distribution at position 0
            attention_scores = attention_scores[:, 0, :]
            k = self.cfg.fusioner_config.num_selected_tokens
            bs, total_token_num = attention_scores.shape  # total_token_num == video_feats.shape[1]

            # num_local_prompts+num_patches
            num_tokens_per_frame = video_frame_feats.shape[2]
            # first part skipped indexes: cls_prompt_tokens + global_prompt_tokens
            skip_indices_list = list(range(num_global_prompts + 1))

            # second part skipped indexes: local_prompt_tokens
            for i in range(num_frames):
                # TODO: make an assumption that num_local_prompts == 1
                local_prompt_index = num_global_prompts + 1 + i * num_tokens_per_frame
                skip_indices_list.append(local_prompt_index)

            skip_indices = torch.tensor(skip_indices_list, device=attention_scores.device)

            # set attention_scores to -inf
            attention_scores.scatter_(
                dim=1,
                index=skip_indices.unsqueeze(0).expand(bs, -1),
                value=float('-inf')
            )

            # topk_indices.shape = (bs, k)
            topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1, largest=True, sorted=True)
            batch_indices = torch.arange(bs, device=attention_scores.device).unsqueeze(1).expand(-1, k)

            # [bs, k, hidden_dim]
            attention_tokens = video_feats[batch_indices, topk_indices, :]

            selected_video_feats = torch.cat([
                cls_prompt_tokens,  # [bs, 1, hidden_dim]
                global_prompt_tokens,  # [bs, num_global_prompts, hidden_dim]
                local_prompt_tokens,  # [bs, num_frames, hidden_dim] (已flatten或未flatten)
                attention_tokens  # [bs, k, hidden_dim]
            ], dim=1)

            return selected_video_feats, attention_scores
        elif select_type == 'added_cls+frame_cls+cls_attention_scores_no_overlap':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size

            global_prompt_tokens = video_feats[:, 1:num_global_prompts + 1, :]
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(
                video_feats.shape[0], num_frames, -1, video_feats.shape[-1]
            )
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])

            # attention_scores: shape = (bs, num_heads, num_queries, num_keys)
            assert attention_scores is not None, 'attention scores is None'

            if self.cfg.fusioner_config.select_attention_scores_type == 'max':
                attention_scores, _ = attention_scores.max(dim=1)  # [bs, num_queries, num_keys]
            elif self.cfg.fusioner_config.select_attention_scores_type == 'mean':
                attention_scores = attention_scores.mean(dim=1)  # [bs, num_queries, num_keys]

            # select cls tokens attention distribution at position 0
            attention_scores = attention_scores[:, 0, :]
            k = self.cfg.fusioner_config.num_selected_tokens
            bs, total_token_num = attention_scores.shape  # total_token_num == video_feats.shape[1]

            # num_local_prompts+num_patches
            num_tokens_per_frame = video_frame_feats.shape[2]
            # first part skipped indexes: cls_prompt_tokens + global_prompt_tokens
            skip_indices_list = list(range(num_global_prompts + 1))  

            # second part skipped indexes: local_prompt_tokens
            for i in range(num_frames):
                # TODO: make an assumption that num_local_prompts == 1
                local_prompt_index = num_global_prompts + 1 + i * num_tokens_per_frame
                skip_indices_list.append(local_prompt_index)

            skip_indices = torch.tensor(skip_indices_list, device=attention_scores.device)

            # set attention_scores to -inf
            attention_scores.scatter_(
                dim=1,
                index=skip_indices.unsqueeze(0).expand(bs, -1),
                value=float('-inf')
            )

            # topk_indices.shape = (bs, k)
            topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1, largest=True, sorted=True)
            batch_indices = torch.arange(bs, device=attention_scores.device).unsqueeze(1).expand(-1, k)

            # [bs, k, hidden_dim]
            attention_tokens = video_feats[batch_indices, topk_indices, :]

            selected_video_feats = torch.cat([
                global_prompt_tokens,  # [bs, num_global_prompts, hidden_dim]
                local_prompt_tokens,  # [bs, num_frames, hidden_dim] (已flatten或未flatten)
                attention_tokens  # [bs, k, hidden_dim]
            ], dim=1)

            return selected_video_feats
        else:
            raise NotImplementedError

    def select_text_tokens(self, cfg, select_type, text_all_feats, text_input_mask, video_feats):
        # text_all_feats: [bs, num_words, hidden_dim]
        # text_input_mask: [bs, num_words]
        # video_feats: [bs, hidden_dim]
        if select_type == 'words2video-topk':
            k = cfg.generator_config.topk_text_tokens
            video_embeds = video_feats / video_feats.norm(dim=-1, keepdim=True)
            text_all_embeds = text_all_feats / text_all_feats.norm(dim=-1, keepdim=True)
            # [bs, num_words]
            sim_matrix = torch.bmm(text_all_embeds, video_embeds.unsqueeze(-1)).squeeze(-1)
            # mask out [sep] token
            text_input_mask = flip_last_one_token(text_input_mask)
            text_input_mask = text_input_mask.bool()
            sim_matrix[~text_input_mask] = -float('inf')

            scores, indices = torch.topk(sim_matrix, k, dim=-1)
            selected_text_feats = text_all_feats[torch.arange(text_all_feats.shape[0]).unsqueeze(-1), indices, :]
        else:
            raise NotImplementedError
        return selected_text_feats


    def forward(self, video, text_input_ids, text_input_mask, **kwargs):

        text_outputs = self.get_text_features(text_input_ids, text_input_mask)  # dict
        video_outputs = self.get_video_features(video)  # dict
        outputs = text_outputs
        outputs.update(video_outputs)
        text_all_feats = outputs['text_all_feats']
        video_feats = outputs['video_feats']

        # get topk most informative text tokens
        if hasattr(self.cfg, 'generator_config'):
            selected_text_feats = self.select_text_tokens(self.cfg, self.cfg.generator_config.select_text_tokens_type,
                                                          text_all_feats, text_input_mask, video_feats)

        else:
            selected_text_feats = text_all_feats

        outputs['selected_text_feats'] = selected_text_feats
        # TODO: list all key value in outputs explicitly
        return outputs


    def get_text_features(self, text_input_ids, text_input_mask, **kwargs):
        inputs = {"input_ids": text_input_ids,
                  "attention_mask": text_input_mask,
                  "output_attentions": True,
                  "output_hidden_states": True,
                  "return_dict": True}

        ############
        # basic part
        ############
        text_outputs = self.encoder.clipmodel.get_text_outputs(**inputs)
        # 0 is all feats(last_hidden_state)
        text_all_feats = text_outputs['last_hidden_state']
        # 1 is textual global feats
        text_feats = text_outputs['pooler_output']
        # # this is a tuple of hidden states, len == 13, the first one is input_embeds before layer 0
        text_hidden_states = text_outputs['hidden_states']

        if hasattr(self.cfg, 'generator_config'):
            text_eos_layers_output = text_hidden_states[-self.cfg.generation_loss_config.num_layers:]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # [bs, hidden_dim]
            text_eos_layers_output = tuple(x[torch.arange(x.shape[0]), text_input_ids.argmax(dim=-1)] for x in text_eos_layers_output)
            # [bs, num_layers, hidden_dim]
            text_eos_layers_feats = torch.stack(text_eos_layers_output, dim=1)
            text_eos_layers_feats = text_eos_layers_feats.reshape(-1, text_eos_layers_feats.shape[-1])
        else:
            text_eos_layers_feats = text_feats

        # fc of clipmodel, text_feats and video_feats are also go through fc inside of self.encoder.clipmodel.forward_wo_norm call
        # only video_feats need to go through a post_layernorm
        text_feats = self.encoder.clipmodel.text_projection(text_feats)
        text_all_feats = self.encoder.clipmodel.text_projection(text_all_feats)

        ############
        # other outputs
        ############
        text_outputs = {
            'text_feats': text_feats,
            'text_all_feats': text_all_feats,
            'text_eos_layers_feats': text_eos_layers_feats,
            # 'selected_text_feats': selected_text_feats,
        }

        return text_outputs

    def get_video_features(self, video, **kwargs):
        inputs = {"pixel_values": video,
                  "output_attentions": True,
                  "output_hidden_states": True,
                  "return_dict": True}

        ############
        # basic part
        ############
        video_outputs = self.encoder.clipmodel.get_vision_outputs(**inputs)
        # 0 is all feats(last_hidden_state)
        video_all_feats = video_outputs['last_hidden_state']
        # 1 is visual global feats
        video_feats = video_outputs['pooler_output']
        vision_attentions = video_outputs['attentions']


        # fc of clipmodel, text_feats and video_feats are also go through fc inside of self.encoder.clipmodel.forward_wo_norm call
        # only video_feats need to go through a post_layernorm
        video_feats = self.encoder.clipmodel.visual_projection(video_feats)
        video_all_feats = self.encoder.clipmodel.vision_model.post_layernorm(video_all_feats)
        video_all_feats = self.encoder.clipmodel.visual_projection(video_all_feats)

        ############
        # other outputs
        ############
        if hasattr(self.cfg, 'generator_config'):
            # select the attention score of the last layer
            attention_scores = vision_attentions[-1]
            generator_selected_video_feats, informativeness = self.select_video_tokens(self.cfg,
                                                                      self.cfg.generator_config.input_video_tokens_type,
                                                                      video_all_feats,
                                                                      attention_scores=attention_scores)
            generator_outputs = self.generator(generator_selected_video_feats)
            generated_global_text_feats = generator_outputs['generated_global_text_feats']
            generated_text_feats = generator_outputs['generated_word_text_feats']

            # this is a tuple of hidden states, len == 13, the first one is input_embeds before layer 0
            generator_eos_layers_output = generator_outputs['hidden_states']
            generator_eos_layers_output = generator_eos_layers_output[-self.cfg.generation_loss_config.num_layers:]
            # take features from the eot embedding (eot_token is the last one in each sequence)
            generator_eos_layers_output = tuple(x[torch.arange(x.shape[0]), -1] for x in generator_eos_layers_output)
            # [bs, num_layers, hidden_dim]
            generator_eos_layers_feats = torch.stack(generator_eos_layers_output, dim=1)
            generator_eos_layers_feats = generator_eos_layers_feats.reshape(-1, generator_eos_layers_feats.shape[-1])


            # generated_text_feats, generator_hidden_states = self.generator(generator_selected_video_feats)
            # generated_global_text_feats = self.generator.get_overall_text_repr(generated_text_feats)
        else:
            generator_hidden_states = video_feats
            generator_selected_video_feats = video_feats.unsqueeze(1)
            generated_text_feats = video_feats
            generated_global_text_feats = video_feats
            generator_eos_layers_feats = video_feats

        if hasattr(self.cfg, 'fusioner_config'):
            # select the attention score of the last layer
            attention_scores = vision_attentions[-1]
            fusioner_selected_video_feats = self.select_video_tokens(self.cfg,
                                                                     self.cfg.fusioner_config.input_video_tokens_type,
                                                                     video_all_feats, attention_scores=attention_scores)

            if self.cfg.fusioner_config.text_embeds_type == 'word-level':
                if self.cfg.fusioner_config.fuse_text_first:
                    weights = self.AFA(generated_text_feats)
                    generated_global_text_feats = torch.sum(weights * generated_text_feats, dim=1)
                    fused_video_feats, place_holder = self.fusioner(generated_global_text_feats,
                                                                fusioner_selected_video_feats, video_mask=None)
                else:
                    _, num_words, _ = generated_text_feats.shape
                    fused_video_feats_list = []
                    for i in range(num_words):
                        generated_word_feats = generated_text_feats[:, i, :]
                        fused_video_feats, _ = self.fusioner(generated_word_feats, fusioner_selected_video_feats,
                                                             video_mask=None)
                        fused_video_feats_list.append(fused_video_feats)
                    # fused_video_feats = torch.mean(torch.stack(fused_video_feats_list, dim=1), dim=1)
                    # B x num_words x D
                    fused_video_feats = torch.stack(fused_video_feats_list, dim=1)
                    # B x num_words x 1
                    weights = self.AFA(fused_video_feats)
                    weighted_sum_video_feats = torch.sum(weights * fused_video_feats, dim=1)
                    fused_video_feats = weighted_sum_video_feats
            # TODO: the logic of co-attn branch is not correct
            elif self.cfg.fusioner_config.text_embeds_type == 'co-attn':
                fused_video_feats, _ = self.fusioner(generated_text_feats, fusioner_selected_video_feats,
                                                     video_mask=None)
            else:

                fused_video_feats, place_holder = self.fusioner(generated_global_text_feats, fusioner_selected_video_feats,
                                                   video_mask=None)

        else:
            # fusioner_selected_video_feats = self.select_video_tokens(self.cfg,
            #                                                          self.cfg.fusioner_select_video_tokens_type,
            #                                                          video_all_feats)
            fusioner_selected_video_feats = video_feats.unsqueeze(1)
            fused_video_feats = generated_global_text_feats

        if hasattr(self.cfg, 'fusioner_config') and self.cfg.fusioner_config.final_residual_add:
            final_video_feats = video_feats + fused_video_feats
        elif hasattr(self.cfg, 'fusioner_config') and self.cfg.fusioner_config.final_alpha_add:
            # using this to check if fusioner is frozen further to decide which to return
            if all(param.requires_grad == False for param in self.fusioner.parameters()):
                final_video_feats = video_feats
            else:
                final_video_feats = (1 - self.final_add_alpha) * video_feats + self.final_add_alpha * fused_video_feats
        elif hasattr(self.cfg, 'pseudo_final_alpha_add') and self.final_pseudo_final_alpha_add:
            final_video_feats = ((1 - self.final_pseudo_final_alpha_add) * generated_global_text_feats +
                                 self.final_pseudo_final_alpha_add * video_feats)
        else:
            final_video_feats = fused_video_feats

        video_outputs = {
            'video_feats': video_feats,
            'video_all_feats': video_all_feats,
            'vision_attentions': vision_attentions,
            'final_video_feats': final_video_feats,
            'fused_video_feats': fused_video_feats,
            'generated_text_feats': generated_text_feats,
            'generated_global_text_feats': generated_global_text_feats,
            'generator_selected_video_feats': generator_selected_video_feats,
            'generator_eos_layers_feats': generator_eos_layers_feats,
            'fusioner_selected_video_feats': fusioner_selected_video_feats,
            'attention_scores': informativeness,
        }

        return video_outputs


class FeatureReconstructor(nn.Module):
    def __init__(self, cfg, fr_config):
        super().__init__()
        self.cfg = cfg
        self.fr_config = fr_config
        self.fr_model_type = self.fr_config.fr_model_type
        if self.fr_model_type == 'CLIP-text-encoder':
            self.num_query_token = self.fr_config.num_query_token

            if self.num_query_token == 2 and getattr(self.cfg.generator_config, 'topk_text_tokens', 0) == 1:
                self.clip_reconstructor, self.bos_tokens, self.query_tokens, self.word_tokens = FeatureReconstructor.init_CLIP_reconstructor(
                    self.cfg)
            else:
                self.clip_reconstructor, self.bos_tokens, self.query_tokens = FeatureReconstructor.init_CLIP_reconstructor(
                    self.cfg)

        elif self.fr_model_type == 'Bert-Qformer':
            self.num_query_token = self.fr_config.num_query_token
            self.vision_hidden_dim = self.fr_config.vision_hidden_dim
            self.model_hidden_dim = self.fr_config.model_hidden_dim
            self.Qformer_hidden_dim = self.fr_config.Qformer_hidden_dim
            self.huggingface_model_name = self.fr_config.huggingface_model_name
            self.huggingface_model_config = self.fr_config.huggingface_model_config
            self.overall_text_repr_type = self.fr_config.overall_text_repr_type
            self.input_video_tokens_type = self.fr_config.input_video_tokens_type
            # hard code for checking
            # TODO: find a better way to check
            encoder_config = BertConfig.from_dict(self.huggingface_model_config)
            assert self.Qformer_hidden_dim == encoder_config.hidden_size
            self.Qformer, self.query_tokens = FeatureReconstructor.init_Qformer(self.num_query_token,
                                                                                self.vision_hidden_dim,
                                                                                bert_config=self.huggingface_model_config,
                                                                                huggingface_model_name=self.huggingface_model_name)
            self.Qformer_proj = nn.Linear(self.Qformer_hidden_dim, self.model_hidden_dim, bias=True)
        else:
            raise NotImplementedError

    @classmethod
    # we only need a standard transformer decoder, so fix cross_attention_freq to 1
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=1, from_pretrained=True,
                     bert_config=None, huggingface_model_name=None):
        # encoder_config = BertConfig.from_pretrained(bert_config)
        encoder_config = BertConfig.from_dict(bert_config)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        if from_pretrained:
            Qformer, msg = BertLMHeadModelForQformer.from_pretrained(
                pretrained_model_name_or_path=huggingface_model_name, config=encoder_config, output_loading_info=True
            )
            if hvd.rank() == 0:
                print('When initialize Qformer form {}, got msg: \n {}'.format(huggingface_model_name, msg))
            query_tokens = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size)
            )
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        else:
            raise NotImplementedError
        return Qformer, query_tokens

    @classmethod
    def init_CLIP_reconstructor(cls, cfg):
        clip_reconstructor = CLIPFeatureReconstructor(cfg)

        with torch.no_grad():
            tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip_config)
            bos_token_id = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.bos_token), dtype=int, device='cpu')
            eos_token_id = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.eos_token), dtype=int, device='cpu')
            bos_embeds = clip_reconstructor.embeddings.token_embedding(bos_token_id)
            eos_embeds = clip_reconstructor.embeddings.token_embedding(eos_token_id)

            # make embeds trainable
            if cfg.generator_config.freeze_bos_embeds:
                bos_embeds = nn.Parameter(bos_embeds.clone().detach(), requires_grad=False)
            else:
                bos_embeds = nn.Parameter(bos_embeds.clone().detach(), requires_grad=True)
            if cfg.generator_config.freeze_eos_embeds:
                eos_embeds = nn.Parameter(eos_embeds.clone().detach(), requires_grad=False)
            else:
                eos_embeds = nn.Parameter(eos_embeds.clone().detach(), requires_grad=True)

            if cfg.generator_config.num_query_token == 2:
                word_embeds = clip_reconstructor.embeddings.token_embedding(eos_token_id)
                word_embeds = nn.Parameter(word_embeds.clone().detach(), requires_grad=True)
                del clip_reconstructor.embeddings.token_embedding
                return clip_reconstructor, bos_embeds, eos_embeds, word_embeds

        del clip_reconstructor.embeddings.token_embedding
        return clip_reconstructor, bos_embeds, eos_embeds

    def get_overall_text_repr(self, x):
        if self.num_query_token > 1:
            if self.overall_text_repr_type == 'pick_eot':
                # pick the last token
                x = x[:, -1, :]
                x = x.squeeze(1)
            else:
                raise NotImplementedError
        else:
            x = x.squeeze(1)
        return x

    def forward(self, video_embeds):
        if self.fr_model_type == 'Bert-Qformer':
            # expand batch_size
            query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)

            video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(
                video_embeds.device
            )

            # position embeds are added inside qformer
            query_outputs = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=video_embeds,
                encoder_attention_mask=video_atts,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

            x = self.Qformer_proj(query_outputs.last_hidden_state)
            hidden_states = query_outputs.hidden_states
            if hasattr(self.cfg.generation_loss_config, 'num_layers'):
                num_layers = self.cfg.generation_loss_config.num_layers
                # (bs, num_layers, len_seq, hidden_dim)
                hidden_states = torch.stack(hidden_states[-num_layers:-1], dim=1)
                hidden_states = torch.cat([hidden_states, x.unsqueeze(1)], dim=1)
            else:
                hidden_states = x

            return x, hidden_states
        elif self.fr_model_type == 'CLIP-text-encoder':
            # [bs, hidden_dim]
            bos_tokens = self.bos_tokens.expand(video_embeds.shape[0], -1).to(video_embeds.device)
            query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1).to(video_embeds.device)
            # [bs, hidden_dim]
            if self.num_query_token == 2 and getattr(self.cfg.generator_config, 'topk_text_tokens', 0) == 1:
                word_tokens = self.query_tokens.expand(video_embeds.shape[0], -1).to(video_embeds.device)
                query_outputs = self.clip_reconstructor.generate_features(video_embeds, bos_embeds=bos_tokens,
                                                                          eos_embeds=query_tokens, word_embeds=word_tokens)
                word_output = query_outputs.word_pooler_output



            else:
                query_outputs = self.clip_reconstructor.generate_features(video_embeds, bos_embeds=bos_tokens,
                                                                      eos_embeds=query_tokens)
                word_output = query_outputs.pooler_output

            hidden_states = query_outputs.hidden_states
            x = query_outputs.pooler_output

            outputs = {
                'generated_global_text_feats': x,
                'generated_word_text_feats': word_output,
                'hidden_states': hidden_states,
            }

            return outputs
        else:
            raise NotImplementedError


class CLIPFeatureReconstructor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_query_token = self.cfg.generator_config.num_query_token
        clipconfig = CLIPConfig.from_pretrained(cfg.clip_config)
        clipmodel = CLIP.from_pretrained(cfg.clip_weights, config=clipconfig)
        self.embeddings = clipmodel.text_model.embeddings
        self.encoder = clipmodel.text_model.encoder
        self.text_projection = clipmodel.text_projection
        self.final_layer_norm = clipmodel.text_model.final_layer_norm

        if getattr(self.cfg.generator_config, 'video_mapper_type') == 'Linear':
            self.video_mapper = nn.Linear(clipmodel.text_model.config.hidden_size,
                                          clipmodel.text_model.config.hidden_size, bias=True)
        elif getattr(self.cfg.generator_config, 'video_mapper_type') == 'MLP':
            self.video_mapper = nn.Sequential(
                nn.Linear(clipmodel.text_model.config.hidden_size, clipmodel.text_model.config.hidden_size, bias=True),
                nn.GELU(),
                nn.Linear(clipmodel.text_model.config.hidden_size, clipmodel.text_model.config.hidden_size, bias=True)
            )
        else:
            raise NotImplementedError

        del clipmodel

    def generate_features(self, video_feats, bos_embeds, eos_embeds, word_embeds=None):
        video_feats = self.video_mapper(video_feats)

        if hasattr(self.cfg, 'generator_config') and getattr(self.cfg.generator_config, 'use_bos_embeds', 0):
            if self.num_query_token == 2 and getattr(self.cfg.generator_config, 'topk_text_tokens', 0) == 1:
                inputs_embeds = torch.cat((bos_embeds.unsqueeze(1), video_feats, word_embeds.unsqueeze(1), eos_embeds.unsqueeze(1)), dim=1)
            else:
                inputs_embeds = torch.cat((bos_embeds.unsqueeze(1), video_feats, eos_embeds.unsqueeze(1)), dim=1)
        else:
            if self.num_query_token == 2 and getattr(self.cfg.generator_config, 'topk_text_tokens', 0) == 1:
                inputs_embeds = torch.cat((video_feats, word_embeds.unsqueeze(1), eos_embeds.unsqueeze(1)), dim=1)
            else:
                inputs_embeds = torch.cat((video_feats, eos_embeds.unsqueeze(1)), dim=1)

        bsz, seq_len, hidden_dim = inputs_embeds.shape
        position_ids = self.embeddings.position_ids[:, :seq_len]
        position_embeddings = self.embeddings.position_embedding(position_ids)
        hidden_states = inputs_embeds + position_embeddings

        attention_mask = None
        output_attentions = True
        output_hidden_states = True
        return_dict = True
        if_fp16 = hidden_states.dtype == torch.float16
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, fp16=if_fp16).to(hidden_states.device)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        last_hidden_state = self.text_projection(last_hidden_state)

        # pick the last one
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), -1, :]
        # pick the second last one, which is the word embeds
        word_pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), -2, :].unsqueeze(1)

        return BaseModelOutputWith2Pooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            word_pooler_output=word_pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, fp16=False):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        if fp16:
            mask = mask.half()
        return mask

    def freeze_encoder(self):
        freeze_list = [self.embeddings, self.encoder, self.text_projection, self.final_layer_norm]
        for m in freeze_list:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if hvd.rank() == 0:
            for name, param in self.named_parameters():
                if param.requires_grad:
                     LOGGER.info('{} in generator requires grad'.format(name))



class MeanPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def mean_pooling_visual(self, visual_output, video_mask):
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out, None

    def forward(self, generated_feat, visual_output, video_mask):
        return self.mean_pooling_visual(visual_output, video_mask)


class AttentionFusioner(nn.Module):
    def __init__(self,
                 fusioner_config='./attention_fusioner.json',
                 set_position_embeddings: bool = False,
                 input_dim=None,
                 ):
        super().__init__()

        encoder_config = BertConfig.from_dict(fusioner_config.self_attention_config.huggingface_model_config)
        self.encoder = BertEncoder(encoder_config)

        self.hidden_dim = encoder_config.hidden_size
        scale = self.hidden_dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(self.hidden_dim))
        self.input_dim = fusioner_config.embed_dim
        self.output_dim = encoder_config.hidden_size
        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=True)
        self.LayerNorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)
        if set_position_embeddings:
            raise NotImplementedError

    def forward(self, generated_text_feat, visual_feat, video_mask):
        # visual_feat shape is (batch_size, len_seq, hidden_dim)
        # generated_text_feat shape is (batch_size, len_seq, hidden_dim)
        visual_feat = self.fc(self.LayerNorm(visual_feat))
        # data = torch.cat((self.class_embedding + torch.zeros(visual_feat.shape[0], 1, visual_feat.shape[-1]).to(
        #     self.class_embedding.device),
        #                   visual_feat, generated_text_feat), dim=1)
        data = torch.cat((self.class_embedding + torch.zeros(visual_feat.shape[0], 1, visual_feat.shape[-1]).to(
            self.class_embedding.device), visual_feat), dim=1)
        # data = visual_feat

        # encoder_output = self.encoder(visual_feat, mode=None)
        encoder_output = self.encoder(data, mode=None)
        # get cls token
        x = encoder_output.last_hidden_state
        x = x[:, 0, :]
        # x = torch.mean(x, dim=1)
        # x = encoder_output.last_hidden_state[torch.arange(x.shape[0]), torch.zeros(x.shape[0])]

        return x, encoder_output.last_hidden_state


class WideMutliHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.embed_dim, self.proj_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.proj_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.proj_dim)
        self.out_proj = nn.Linear(self.proj_dim, self.embed_dim)
        self.dropout = nn.Dropout(0.1)



    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x embed_dim
        """
        num_texts, embed_dim = text_embeds.shape
        num_vids, num_frames, embed_dim = video_embeds.shape
        # num_vids == num_texts

        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        q = q.unsqueeze(-1)

        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0, 2, 1, 3)

        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)

        # num_vids x num_heads x num_frames x 1
        attention_logits = k @ q
        attention_logits = attention_logits.squeeze(-1) / math.sqrt(self.head_dim)
        # num_vids x num_heads x num_frames
        attention_weights = F.softmax(attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0, 2, 3, 1)
        # num_vids x num_heads x num_frames x 1
        attention_weights = attention_weights.unsqueeze(-1)
        # num_vids x num_heads x head_dim x 1
        attention = v @ attention_weights
        attention = attention.squeeze(-1).reshape(num_vids, self.proj_dim)

        # num_vids x embed_dim
        o = self.out_proj(attention)
        return o


class ResidualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.attn = WideMutliHeadedAttention(embed_dim, num_heads, head_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim))
        ]))
        self.ln_text = nn.LayerNorm(embed_dim)
        self.ln_video = nn.LayerNorm(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x embed_dim
        """
        text_embeds = self.ln_text(text_embeds)
        video_embeds = self.ln_video(video_embeds)
        x = text_embeds + self.attn(text_embeds, video_embeds)
        x = x + self.mlp(self.ln(x))

        return x


class WoResidualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.attn = WideMutliHeadedAttention(embed_dim, num_heads, head_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim))
        ]))
        self.ln_text = nn.LayerNorm(embed_dim)
        self.ln_video = nn.LayerNorm(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x embed_dim
        """
        text_embeds = self.ln_text(text_embeds)
        video_embeds = self.ln_video(video_embeds)
        x = self.attn(text_embeds, video_embeds)
        # x = text_embeds + self.attn(text_embeds, video_embeds)
        x = x + self.mlp(self.ln(x))

        return x


class WideCrossAttentionPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = self.config['num_layers']
        self.embed_dim = self.config['embed_dim']
        self.num_heads = self.config['num_mha_heads']
        self.head_dim = self.config['head_dim']
        self.dropout = self.config['transformer_dropout']
        self.text_embeds_type = self.config['text_embeds_type']

        self.ln = nn.LayerNorm(self.embed_dim)
        self.final_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.blocks = nn.ModuleList([ResidualAttentionBlock(self.embed_dim, self.num_heads, self.head_dim)
                                     for _ in range(self.num_layers)])

        if self.config.identical_init:
            self._identical_init()
        else:
            self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def _identical_init(self):
        for name, param in self.named_parameters():
            if 'attn' in name:
                if 'out_proj' in name:
                    if 'weight' in name:
                        weight = param.data
                        weight.zero_()

                        for i in range(self.head_dim):
                            for k in range(self.num_heads):
                                weight[i, i + k * self.head_dim] = 1.0 / self.num_heads

                    elif 'bias' in name:
                        param.data.fill_(0.0)
                elif 'proj' in name:
                    if 'weight' in name:
                        weight = torch.cat([torch.eye(512) for _ in range(self.num_heads)], dim=0)
                        param.data.copy_(weight)
                    elif 'bias' in name:
                        param.data.fill_(0.0)
            elif 'final_linear' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds, video_mask):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x embed_dim
        """
        x = text_embeds
        for block in self.blocks:
            x = block(x, video_embeds)

        x = self.ln(x + text_embeds)
        x = self.final_linear(x)
        return x, None


class WoResidualWideCrossAttentionPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = self.config['num_layers']
        self.embed_dim = self.config['embed_dim']
        self.num_heads = self.config['num_mha_heads']
        self.head_dim = self.config['head_dim']
        self.dropout = self.config['transformer_dropout']
        self.text_embeds_type = self.config['text_embeds_type']

        self.ln = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.final_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.blocks = nn.ModuleList([WoResidualAttentionBlock(self.embed_dim, self.num_heads, self.head_dim)
                                     for _ in range(self.num_layers)])

        if self.config.zero_init:
            self._init_zero_map_parameters()
        elif self.config.identical_init:
            self._identical_init()
        else:
            self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name or 'fc' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def _init_zero_map_parameters(self):
        for name, param in self.named_parameters():
            if 'final_mlp.weight' in name:
                nn.init.constant_(param, 0.0)
            elif 'final_mlp.bias' in name:
                param.data.fill_(0.0)

    def _identical_init(self):
        for name, param in self.named_parameters():
            if 'attn' in name:
                if 'out_proj' in name:
                    if 'weight' in name:
                        weight = param.data
                        weight.zero_()

                        for i in range(self.head_dim):
                            for k in range(self.num_heads):
                                weight[i, i + k * self.head_dim] = 1.0 / self.num_heads

                    elif 'bias' in name:
                        param.data.fill_(0.0)
                elif 'proj' in name:
                    if 'weight' in name:
                        weight = torch.cat([torch.eye(512) for _ in range(self.num_heads)], dim=0)
                        param.data.copy_(weight)
                    elif 'bias' in name:
                        param.data.fill_(0.0)
            elif 'final_linear' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)



    def forward(self, text_embeds, video_embeds, video_mask):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x embed_dim
        """
        x = text_embeds
        for block in self.blocks:
            x = block(x, video_embeds)

        # x = self.dropout(x)
        x = self.ln(x)
        # x = self.ln(x + text_embeds)
        x = self.final_linear(x)
        return x, None


class XPoolMultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(XPoolMultiHeadedAttention, self).__init__()
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_mha_heads']
        self.attn_temp = config['attn_temp']
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1, 2, 0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0, 2, 1, 3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0, 2, 3, 1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_logits = attention_logits / self.attn_temp
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0, 3, 1, 2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
        return o


class XPoolCrossAttentionPooler(nn.Module):
    def __init__(self, config):
        super(XPoolCrossAttentionPooler, self).__init__()
        self.config = config
        self.embed_dim = self.config.embed_dim
        dropout = self.config.transformer_dropout

        self.cross_attn = XPoolMultiHeadedAttention(self.config)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        if self.config.zero_init:
            self._init_zero_map_parameters()
        else:
            self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def _init_zero_map_parameters(self):
        for name, param in self.named_parameters():
            if 'linear_proj.weight' or 'out_proj.weight' in name:
                nn.init.constant_(param, 0.0)
            elif 'linear_proj.bias' or 'out_proj.bias' in name:
                param.data.fill_(0.0)

    def forward(self, text_embeds, video_embeds, video_mask):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        # only get positive pairs
        diagonal_out = torch.diagonal(out, dim1=0, dim2=1).permute(1, 0)

        return diagonal_out, out


class TS2XPoolCrossAttentionPooler(nn.Module):
    def __init__(self, config):
        super(TS2XPoolCrossAttentionPooler, self).__init__()
        self.config = config
        self.embed_dim = self.config.embed_dim
        dropout = self.config.transformer_dropout

        self.pre_mlp = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim, bias=True), nn.ReLU(),
                                     nn.Linear(self.embed_dim, self.embed_dim, bias=True))
        self.post_mlp = nn.Sequential(nn.Linear(self.embed_dim * 2, self.embed_dim, bias=True), nn.ReLU(),
                                      nn.Linear(self.embed_dim, self.embed_dim, bias=True))
        self.cross_attn = XPoolMultiHeadedAttention(self.config)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        if self.config.zero_init:
            self._init_zero_map_parameters()
        else:
            self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def _init_zero_map_parameters(self):
        for name, param in self.named_parameters():
            if 'linear_proj.weight' or 'out_proj.weight' in name:
                nn.init.constant_(param, 0.0)
            elif 'linear_proj.bias' or 'out_proj.bias' in name:
                param.data.fill_(0.0)

    def forward(self, text_embeds, video_embeds, video_mask):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        video_embeds = self.pre_mlp(video_embeds)
        video_cls = video_embeds[:, 0, :]
        video_embeds = video_embeds[:, 1:, :]
        video_cls = video_cls.unsqueeze(1).expand(-1, video_embeds.shape[1], -1)
        video_embeds = torch.cat([video_cls, video_embeds], dim=-1)
        video_embeds = self.post_mlp(video_embeds)

        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        # only get positive pairs
        diagonal_out = torch.diagonal(out, dim1=0, dim2=1).permute(1, 0)

        return diagonal_out, out


class CoAttentionPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = self.config.embed_dim
        self.num_heads = self.config.num_mha_heads
        self.dropout = 0.1
        self.num_layers = self.config.num_layers
        self.co_attn_final_pooler_type = self.config.co_attn_final_pooler_type
        self.final_pooler_input = self.config.co_attn_final_pooler_input

        self.co_attn_blocks = nn.ModuleList([Co_attention_block(hidden_size=self.hidden_dim,
                                                                num_attention_heads=self.num_heads,
                                                                dropout_rate=self.dropout) for _ in
                                             range(self.num_layers)])

        if self.config.identical_init:
            self.init_identity_weights()


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token



def flip_last_one_token(text_input_mask):
    """
    flip the last "1" token to "0" in the text_input_mask, which is the [sep] token's mask.
    attention! the text_input_mask must has at least one "1" token in each sample.

    Args:
        text_input_mask: [BS, num_words]
    """
    # don't change the original mask
    text_input_mask = text_input_mask.clone()
    indices = torch.sum(text_input_mask, dim=-1, keepdim=False) - 1
    text_input_mask[torch.arange(text_input_mask.size(0)), indices] = 0
    return text_input_mask

@dataclass
class BaseModelOutputWith2Pooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    word_pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
