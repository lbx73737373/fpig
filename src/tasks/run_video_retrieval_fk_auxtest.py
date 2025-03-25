import os

print(os.getcwd())
import time
import random
import math
from collections import defaultdict
import numpy as np
from os.path import join, exists
from easydict import EasyDict as edict
import ipdb
import json
from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as tutils
from torch.utils.data.distributed import DistributedSampler
# import language_evaluation

from apex import amp
import torch.nn as nn
import horovod.torch as hvd
from transformers import CLIPTokenizerFast

from src.modeling.VidCLIP import VidCLIP

from src.datasets.dataset_video_retrieval import (
    HDVILAVideoRetrievalDataset, HDVILAVideoRetrievalWithAuxiliaryTextsDataset,
    ImageTextPretrainingDataset, VideoRetrievalCollator,
    SentenceDataset, OnlySentenceCollator,
    VideoDataset, OnlyVideoCollator)
from src.datasets.dataloader import InfiniteIterator, PrefetchLoader
from src.datasets.data_utils import ImageNorm, mk_input_group

from src.configs.config import shared_configs
from src.utils.sampler import SequentialDistributedSampler
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.basic_utils import (
    load_jsonl, load_json, save_json, get_rounded_percentage)
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import (ModelSaver,
                                 BestModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_mismatch)
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer
from src.optimization.loss import build_loss_func, build_generation_loss_func, build_captioning_loss_func
from src.utils.distributed import all_gather_list
from src.utils.metrics import cal_cossim, cal_cossim_torch, cal_cossim_multi, compute_metrics, compute_metrics_zero_division, compute_metrics_gpu, compute_metrics_multi, np_softmax, tensor_text_to_video_metrics, tensor_video_to_text_sim
from src.utils.distributed import interleaved_allgather

from src.modeling.modeling_fk import FK


def mk_video_ret_dataloader(dataset_name, vis_format, anno_path, auxiliary_anno_path, vis_dir, cfg, tokenizer, mode):
    """"""
    is_train = mode == "train"
    if auxiliary_anno_path:
        dataset = HDVILAVideoRetrievalWithAuxiliaryTextsDataset(
            cfg=cfg,
            vis_dir=vis_dir,
            anno_path=anno_path,
            auxiliary_anno_path=auxiliary_anno_path,
            vis_format=vis_format,
            mode=mode
        )
    else:
        dataset = HDVILAVideoRetrievalDataset(
            cfg=cfg,
            vis_dir=vis_dir,
            anno_path=anno_path,
            vis_format=vis_format,
            mode=mode
        )
    LOGGER.info(f"[{dataset_name}] is_train {is_train} "
                f"dataset size {len(dataset)}, ")

    batch_size = cfg.train_batch_size if is_train else cfg.test_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train)
    vret_collator = VideoRetrievalCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len, is_train=is_train)

    if auxiliary_anno_path:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=sampler,
                                num_workers=cfg.n_workers,
                                pin_memory=cfg.pin_mem,
                                collate_fn=vret_collator.collate_batch_with_auxiliary_text)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=sampler,
                                num_workers=cfg.n_workers,
                                pin_memory=cfg.pin_mem,
                                collate_fn=vret_collator.collate_batch)
    return dataloader

def mk_only_sentence_dataloader(dataset_name, vis_format, anno_path, vis_dir, cfg, tokenizer, mode, multi_sentence_per_video):
    is_train = mode == "train"
    dataset = SentenceDataset(
        cfg=cfg,
        vis_dir=vis_dir,
        anno_path=anno_path,
        vis_format=vis_format,
        mode=mode,
        multi_sentence_per_video=multi_sentence_per_video
    )
    LOGGER.info(f"[{dataset_name}] is_train {is_train} "
                f"dataset size {len(dataset)}, ")
    LOGGER.info(f"len of only sentence dataset is: {len(dataset)}")

    batch_size = cfg.train_batch_size if is_train else cfg.test_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train, drop_last=False)
    # sampler = SequentialDistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    ret_collator = OnlySentenceCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len, is_train=is_train)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=ret_collator.collate_batch)

    return dataloader


def mk_only_video_dataloader(dataset_name, vis_format, anno_path, vis_dir, cfg, tokenizer, mode):
    is_train = mode == "train"
    dataset = VideoDataset(
        cfg=cfg,
        vis_dir=vis_dir,
        anno_path=anno_path,
        vis_format=vis_format,
        mode=mode
    )
    LOGGER.info(f"[{dataset_name}] is_train {is_train} "
                f"dataset size {len(dataset)}, ")
    LOGGER.info(f"len of only video dataset is: {len(dataset)}")

    batch_size = cfg.train_batch_size if is_train else cfg.test_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train, drop_last=False)
    # sampler = SequentialDistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    ret_collator = OnlyVideoCollator(is_train=is_train)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=ret_collator.collate_batch)

    return dataloader


def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")

    db = cfg.train_datasets
    train_loader = mk_video_ret_dataloader(
        dataset_name=db.name, vis_format=db.vis_format,
        anno_path=db.txt, auxiliary_anno_path=db.auxiliary_txt, vis_dir=db.vis,
        cfg=cfg, tokenizer=tokenizer, mode="train"
    )

    val_loaders = {}
    for db in cfg.val_datasets:
        val_loaders[db.name] = {
            'sentence_loader': mk_only_sentence_dataloader(
                dataset_name=db.name, vis_format=db.vis_format,
                anno_path=db.txt, vis_dir=db.vis,
                cfg=cfg, tokenizer=tokenizer, mode="val", multi_sentence_per_video=db.multi_sentence_per_video
            ),
            'video_loader': mk_only_video_dataloader(
                dataset_name=db.name, vis_format=db.vis_format,
                anno_path=db.txt, vis_dir=db.vis,
                cfg=cfg, tokenizer=tokenizer, mode="val"
            )
        }

    inference_loaders = {}
    for db in cfg.inference_datasets:
        inference_loaders[db.name] = {
            'sentence_loader': mk_only_sentence_dataloader(
                dataset_name=db.name, vis_format=db.vis_format,
                anno_path=db.txt, vis_dir=db.vis,
                cfg=cfg, tokenizer=tokenizer, mode="test", multi_sentence_per_video=db.multi_sentence_per_video
            ),
            'video_loader': mk_only_video_dataloader(
                dataset_name=db.name, vis_format=db.vis_format,
                anno_path=db.txt, vis_dir=db.vis,
                cfg=cfg, tokenizer=tokenizer, mode="test"
            )
        }
    return train_loader, val_loaders, inference_loaders


def replace_pretrained_generator_weights_name(weights_path, mode='text_encoder'):
    # clip_reconstructor.encoder.layers.7.self_attn.k_proj.bias
    loaded_state_dict = torch.load(weights_path, map_location="cpu")
    if mode == 'text_encoder':
        for k, v in list(loaded_state_dict.items()):
            if k.startswith("encoder.clipmodel.text_model."):
                new_k = k.replace("encoder.clipmodel.text_model.", "clip_reconstructor.")
                loaded_state_dict[new_k] = v
                del loaded_state_dict[k]
            elif k.startswith('encoder.clipmodel.text_projection.'):
                new_k = k.replace("encoder.clipmodel.", "clip_reconstructor.")
                loaded_state_dict[new_k] = v
                del loaded_state_dict[k]
            elif k.startswith("encoder.clipmodel.vision_model."):
                del loaded_state_dict[k]
    else:
        for k, v in list(loaded_state_dict.items()):
            if k.startswith("generator."):
                new_k = k.replace("generator.", "")
                loaded_state_dict[new_k] = v
                del loaded_state_dict[k]

    return loaded_state_dict


def setup_fk_model(cfg, device=None):
    LOGGER.info("Setup model...")

    # model = VidCLIP(cfg)
    model = FK(cfg)

    if cfg.clip_vision_additional_config.keep_frame_cls and cfg.clip_vision_additional_config.type == "ViP":
        # TODO: to fix: hard coding for init frame cls
        model.encoder.clipmodel.vision_model.embeddings.cls_in_frames = nn.Parameter(
            model.encoder.clipmodel.vision_model.embeddings.class_embedding.detach().clone().expand_as(model.encoder.clipmodel.vision_model.embeddings.cls_in_frames))
        LOGGER.info("Init frame cls with class embedding.")

    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)

    if hasattr(cfg, "pretrained_generator_path"):
        LOGGER.info(f"Loading pretrained generator weights from {cfg.pretrained_generator_path}")
        # we only need to load the generator part
        renamed_state_dict = replace_pretrained_generator_weights_name(cfg.pretrained_generator_path, mode='general_generator')
        load_state_dict_with_mismatch(model.generator, renamed_state_dict)

    if hasattr(cfg, "text_encoder_as_pretrained_generator_path"):
        LOGGER.info("Loading text-encoder as pretrained generator weights")
        renamed_state_dict = replace_pretrained_generator_weights_name(cfg.text_encoder_as_pretrained_generator_path, mode='text_encoder')
        load_state_dict_with_mismatch(model.generator, renamed_state_dict)

    if hasattr(cfg, "freeze_generator") and cfg.freeze_generator:
        LOGGER.info('Freezing generator, only video_mapper and query_tokens are trainable')
        model.generator.clip_reconstructor.freeze_encoder()


    if hasattr(cfg, "freeze_text_model") and cfg.freeze_text_model:
        freeze_text_proj = hasattr(cfg, "freeze_text_proj") and cfg.freeze_text_proj
        LOGGER.info(f"Freeze CLIP text model and the status of freezing proj is: {freeze_text_proj}")
        model.encoder.freeze_text_encoder(freeze_text_proj)

    if hasattr(cfg, "overload_logit_scale"):
        model.overload_logit_scale(cfg.overload_logit_scale)

    model.to(device)

    LOGGER.info("Setup model done!")
    return model


@torch.no_grad()
def fk_validate(model, tokenizer, val_loaders, cfg):
    model.eval()

    st = time.time()
    # evaluator = language_evaluation.CocoEvaluator(coco_types=["BLEU", "METEOR", "ROUGE_L", "CIDEr"])
    # tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip_config)

    for loader_name, val_loader in val_loaders.items():
        LOGGER.info(f"Loop val_loader {loader_name}.")
        # valid_len = len(val_loader.dataset)
        text_input_mask_list = []
        text_embeds_list = []
        text_all_embeds_list = []
        original_vis_embeds_list = []
        video_all_embeds_list = []
        final_vis_embeds_list = []
        generated_global_text_embeds_list = []
        fused_vis_embeds_list = []
        generator_selected_vis_embeds_list = []
        fusioner_selected_vis_embeds_list = []
        fusioner_selected_vis_embeds_mean_list = []
        out_tokens_list = []
        input_tokens_list = []
        attention_scores_list = []
        sentence_loader = val_loader['sentence_loader']
        sentence_valid_len = len(sentence_loader.dataset)
        video_loader = val_loader['video_loader']
        video_valid_len = len(video_loader.dataset)

        # #################################################################
        ## below variables are used to multi-sentences retrieval
        # multi_sentence_: important tag for eval
        # cut_off_points: used to tag the label when calculate the metric
        # sentence_num: used to cut the sentence representation
        # video_num: used to cut the video representation
        # #################################################################
        multi_sentence_ = False
        cut_off_points_, sentence_num_, video_num_ = [], -1, -1
        if hasattr(sentence_loader.dataset, 'multi_sentence_per_video') \
                and sentence_loader.dataset.multi_sentence_per_video:
            multi_sentence_ = True
            cut_off_points_ = sentence_loader.dataset.cut_off_points
            sentence_num_ = len(sentence_loader.dataset)
            video_num_ = len(video_loader.dataset)
            cut_off_points_ = [itm - 1 for itm in cut_off_points_]

        if multi_sentence_:
            LOGGER.warning("Eval under the multi-sentence per video clip setting.")
            LOGGER.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

        LOGGER.info(f"Caching text features...")
        if hvd.rank() != 0:
            LOGGER.disabled = True
        for val_step, batch in enumerate(sentence_loader):
            # if "indices" in batch:
            #     LOGGER.info(
            #         f"[Rank {hvd.rank()}] batch={val_step}, dataset indices={batch['indices']}"
            #     )
            outputs = model.get_text_features(**batch)  # dict

            # print('feats vis_features', feats['vis_features'].shape)
            # text_input_mask = batch['text_input_mask']
            # text_input_mask = hvd.allgather(text_input_mask.contiguous())
            # text_feats = hvd.allgather(outputs['text_feats'].contiguous())
            # text_all_feats = hvd.allgather(outputs['text_all_feats'].contiguous())

            text_input_mask = batch['text_input_mask']
            text_input_mask = interleaved_allgather(text_input_mask.contiguous(), dim=0)
            text_feats = interleaved_allgather(outputs['text_feats'].contiguous(), dim=0)
            text_all_feats = interleaved_allgather(outputs['text_all_feats'].contiguous(), dim=0)

            text_embeds = text_feats / text_feats.norm(dim=-1, keepdim=True)
            text_all_embeds = text_all_feats / text_all_feats.norm(dim=-1, keepdim=True)
            # print('allgather vis_features', vis_feat.shape)

            if hvd.rank() == 0:
                text_input_mask_list.append(text_input_mask.cpu().numpy())
                text_embeds_list.append(text_embeds.cpu().numpy())
                text_all_embeds_list.append(text_all_embeds.cpu().numpy())

        hvd.barrier()

        LOGGER.info(f"Caching video features...")
        for val_step, batch in enumerate(video_loader):
            if 'CLIP4clip' in cfg.clip_vision_additional_config.type:
                # reshape Time dim to the batch dim
                B, T, C, H, W = batch['video'].shape
                batch['video'] = batch['video'].reshape(-1, C, H, W)
            outputs = model.get_video_features(**batch)  # dict
            # print('feats vis_features', feats['vis_features'].shape)
            # vis_feats = hvd.allgather(outputs['video_feats'].contiguous())
            # # video_all_feats = hvd.allgather(outputs['video_all_feats'].contiguous())
            # final_vis_feats = hvd.allgather(outputs['final_video_feats'].contiguous())
            # # generated_text_feats = hvd.allgather(outputs['generated_text_feats'])
            # generated_global_text_feats = hvd.allgather(outputs['generated_global_text_feats'].contiguous())
            # fused_vis_feats = hvd.allgather(outputs['fused_video_feats'].contiguous())
            # generator_selected_vis_feats = hvd.allgather(outputs['generator_selected_video_feats'].contiguous())
            # fusioner_selected_vis_feats = hvd.allgather(outputs['fusioner_selected_video_feats'].contiguous())

            attention_scores = interleaved_allgather(outputs['attention_scores'].contiguous(), dim=0)
            vis_feats = interleaved_allgather(outputs['video_feats'].contiguous(), dim=0)
            # video_all_feats = hvd.allgather(outputs['video_all_feats'].contiguous())
            final_vis_feats = interleaved_allgather(outputs['final_video_feats'].contiguous(), dim=0)
            # generated_text_feats = hvd.allgather(outputs['generated_text_feats'])
            generated_global_text_feats = interleaved_allgather(outputs['generated_global_text_feats'].contiguous(), dim=0)
            fused_vis_feats = interleaved_allgather(outputs['fused_video_feats'].contiguous(), dim=0)
            generator_selected_vis_feats = interleaved_allgather(outputs['generator_selected_video_feats'].contiguous(), dim=0)
            fusioner_selected_vis_feats = interleaved_allgather(outputs['fusioner_selected_video_feats'].contiguous(), dim=0)
            if hasattr(cfg, 'caption_loss_config'):
                out_tokens = interleaved_allgather(outputs['out_tokens'].contiguous(), dim=0)
                input_tokens = interleaved_allgather(batch['text_input_ids'].contiguous(), dim=0)
            generator_selected_vis_feats = generator_selected_vis_feats.mean(dim=1)
            fusioner_selected_vis_feats_mean = fusioner_selected_vis_feats.mean(dim=1)

            final_vis_embeds = final_vis_feats / final_vis_feats.norm(dim=-1, keepdim=True)
            vis_embeds = vis_feats / vis_feats.norm(dim=-1, keepdim=True)
            # video_all_embeds = video_all_feats / video_all_feats.norm(dim=-1, keepdim=True)
            generated_global_text_embeds = generated_global_text_feats / generated_global_text_feats.norm(dim=-1, keepdim=True)
            fused_vis_embeds = fused_vis_feats / fused_vis_feats.norm(dim=-1, keepdim=True)
            generator_selected_vis_embeds = generator_selected_vis_feats / generator_selected_vis_feats.norm(dim=-1, keepdim=True)
            fusioner_selected_vis_embeds = fusioner_selected_vis_feats / fusioner_selected_vis_feats.norm(dim=-1, keepdim=True)
            fusioner_selected_vis_embeds_mean = fusioner_selected_vis_feats_mean / fusioner_selected_vis_feats_mean.norm(dim=-1, keepdim=True)


            if hvd.rank() == 0:
                attention_scores_list.append(attention_scores.cpu().numpy())
                final_vis_embeds_list.append(final_vis_embeds.cpu().numpy())
                original_vis_embeds_list.append(vis_embeds.cpu().numpy())
                # video_all_embeds_list.append(video_all_embeds.cpu().numpy())
                generated_global_text_embeds_list.append(generated_global_text_embeds.cpu().numpy())
                fused_vis_embeds_list.append(fused_vis_embeds.cpu().numpy())
                generator_selected_vis_embeds_list.append(generator_selected_vis_embeds.cpu().numpy())
                fusioner_selected_vis_embeds_list.append(fusioner_selected_vis_embeds.cpu().numpy())
                fusioner_selected_vis_embeds_mean_list.append(fusioner_selected_vis_embeds_mean.cpu().numpy())

        hvd.barrier()


        if hvd.rank() == 0:
            attention_scores = np.vstack(attention_scores_list)
            print(f'shape of attention_scores is {attention_scores.shape}')
            text_input_mask = np.vstack(text_input_mask_list)
            text_embeds = np.vstack(text_embeds_list)
            print(f'shape of text_embeds is {text_embeds.shape}')
            text_all_embeds = np.vstack(text_all_embeds_list)
            vis_embeds = np.vstack(final_vis_embeds_list)
            print(f'shape of vis_embeds is {vis_embeds.shape}')
            # video_all_embeds = np.vstack(video_all_embeds_list)
            original_vis_embeds = np.vstack(original_vis_embeds_list)
            generated_global_text_embeds = np.vstack(generated_global_text_embeds_list)
            fused_vis_embeds = np.vstack(fused_vis_embeds_list)
            generator_selected_vis_embeds = np.vstack(generator_selected_vis_embeds_list)
            fusioner_selected_vis_embeds = np.vstack(fusioner_selected_vis_embeds_list)
            fusioner_selected_vis_embeds_mean = np.vstack(fusioner_selected_vis_embeds_mean_list)
            if hasattr(cfg, 'caption_loss_config'):
                out_tokens = np.vstack(out_tokens_list)
                input_tokens = np.vstack(input_tokens_list)

            attention_scores = att
            text_input_mask = text_input_mask[:sentence_valid_len]
            text_embeds = text_embeds[:sentence_valid_len]
            text_all_embeds = text_all_embeds[:sentence_valid_len]
            vis_embeds = vis_embeds[:video_valid_len]
            # video_all_embeds = video_all_embeds[:valid_len]
            original_vis_embeds = original_vis_embeds[:video_valid_len]
            generated_global_text_embeds = generated_global_text_embeds[:video_valid_len]
            fused_vis_embeds = fused_vis_embeds[:video_valid_len]
            generator_selected_vis_embeds = generator_selected_vis_embeds[:video_valid_len]
            fusioner_selected_vis_embeds = fusioner_selected_vis_embeds[:video_valid_len]
            fusioner_selected_vis_embeds_mean = fusioner_selected_vis_embeds_mean[:video_valid_len]
            if hasattr(cfg, 'caption_loss_config'):
                out_tokens = out_tokens[:valid_len]
                input_tokens = input_tokens[:valid_len]

                eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                out_tokens, pred_sents, cleaned_pred_sents = cut_and_map_tokens(out_tokens, eos_token_id, tokenizer)
                input_tokens, gt_sents, cleaned_gt_sents = cut_and_map_tokens(input_tokens, eos_token_id, tokenizer)

                # save generated captions
                data = [{"idx_{}".format(idx): {"pred": pred, "gt": gt}} for idx, (pred, gt) in enumerate(zip(cleaned_pred_sents, cleaned_gt_sents))]
                with open(os.path.join(cfg.output_dir, 'generated_captions.json'), "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

                # captioning_results = evaluator.run_evaluation(cleaned_pred_sents, cleaned_gt_sents)

           # save embeds
            embeds_dict = {
                'text_embeds': text_embeds,
                'original_vis_embeds': original_vis_embeds,
                'vis_embeds': vis_embeds
            }
            if hasattr(cfg, 'generator_config'):
                embeds_dict.update({
                    'generated_global_text_embeds': generated_global_text_embeds,
                    'fused_vis_embeds': fused_vis_embeds,
                    'generator_selected_vis_embeds': generator_selected_vis_embeds,
                    'fusioner_selected_vis_embeds': fusioner_selected_vis_embeds
                })

            # print('text_embeds', text_embeds.shape)
            # print('vis_embeds', vis_embeds.shape)

            if hasattr(cfg, 'cal_eval_cossim_on_gpu') and cfg.cal_eval_cossim_on_gpu:
                LOGGER.info("Calculate cossim on GPU.")
                sim_matrix = cal_cossim_torch(text_embeds, vis_embeds, device=f'cuda:{cfg.eval_on_gpu_device_id}')
                text2fused_vis_sim_matrix = cal_cossim_torch(text_embeds, fused_vis_embeds, device=f'cuda:{cfg.eval_on_gpu_device_id}')
                text2generated_text_sim_matrix = cal_cossim_torch(text_embeds, generated_global_text_embeds, device=f'cuda:{cfg.eval_on_gpu_device_id}')
                text2original_vis_sim_matrix = cal_cossim_torch(text_embeds, original_vis_embeds, device=f'cuda:{cfg.eval_on_gpu_device_id}')
                text2fusioner_selected_vis_sim_matrix = cal_cossim_torch(text_embeds, fusioner_selected_vis_embeds_mean, device=f'cuda:{cfg.eval_on_gpu_device_id}')
            else:
                sim_matrix = cal_cossim(text_embeds, vis_embeds)
                text2fused_vis_sim_matrix = cal_cossim(text_embeds, fused_vis_embeds)
                text2generated_text_sim_matrix = cal_cossim(text_embeds, generated_global_text_embeds)
                text2original_vis_sim_matrix = cal_cossim(text_embeds, original_vis_embeds)
                text2fusioner_selected_vis_sim_matrix = cal_cossim(text_embeds, fusioner_selected_vis_embeds_mean)

            if multi_sentence_:
                LOGGER.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
                # sim_matrix = reshape_similarity_matrix(sim_matrix, cut_off_points_)
                sim_matrix = reshape_similarity_matrix_pytorch(sim_matrix, cut_off_points_)
                # cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
                # max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
                # sim_matrix_new = []
                # for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
                #     sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                #                                           np.full((max_length - e_ + s_, sim_matrix.shape[1]),
                #                                                   -np.inf)), axis=0))
                # sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
                # sim_matrix = torch.tensor(sim_matrix, dtype=float)
                LOGGER.info("after reshape, sim matrix size: {} x {} x {}".
                            format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

                text2fused_vis_sim_matrix = reshape_similarity_matrix_pytorch(text2fused_vis_sim_matrix, cut_off_points_)
                # text2fused_vis_sim_matrix = reshape_similarity_matrix(text2fused_vis_sim_matrix, cut_off_points_)
                text2generated_text_sim_matrix = reshape_similarity_matrix_pytorch(text2generated_text_sim_matrix, cut_off_points_)
                # text2generated_text_sim_matrix = reshape_similarity_matrix(text2generated_text_sim_matrix, cut_off_points_)
                text2original_vis_sim_matrix = reshape_similarity_matrix_pytorch(text2original_vis_sim_matrix, cut_off_points_)
                # text2original_vis_sim_matrix = reshape_similarity_matrix(text2original_vis_sim_matrix, cut_off_points_)
                text2fusioner_selected_vis_sim_matrix = reshape_similarity_matrix_pytorch(text2fusioner_selected_vis_sim_matrix, cut_off_points_)
                # text2fusioner_selected_vis_sim_matrix = reshape_similarity_matrix(text2fusioner_selected_vis_sim_matrix, cut_off_points_)

                for type in ["simple", "DSL"]:
                    LOGGER.info(f"Evaluate under setting: {type}.")
                    val_log = {f'valid/{loader_name}_t2v_recall_1': 0,
                               f'valid/{loader_name}_t2v_recall_5': 0,
                               f'valid/{loader_name}_t2v_recall_10': 0,
                               f'valid/{loader_name}_t2v_recall_median': 0,
                               f'valid/{loader_name}_t2v_recall_mean': 0,
                               f'valid/{loader_name}_t2v_recall_SumR': 0,
                               f'valid/{loader_name}_v2t_recall_1': 0,
                               f'valid/{loader_name}_v2t_recall_5': 0,
                               f'valid/{loader_name}_v2t_recall_10': 0,
                               f'valid/{loader_name}_v2t_recall_median': 0,
                               f'valid/{loader_name}_v2t_recall_mean': 0,
                               f'valid/{loader_name}_v2t_recall_SumR': 0,
                               }

                    if type == "DSL":
                        # (max_len, N_t, N_v)
                        sim_matrix = sim_matrix.permute(1, 0, 2)
                        sim_matrix = sim_matrix * F.softmax(sim_matrix * 100, dim=1)
                        # (N_t, max_len, N_v)
                        sim_matrix = sim_matrix.permute(1, 0, 2)

                    # tv_metrics = tensor_text_to_video_metrics(sim_matrix)
                    # t2vr1, t2vr5, t2vr10, t2vmedr, t2vmeanr = tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], \
                    # tv_metrics['MedianR'], tv_metrics['MeanR']
                    if hasattr(cfg, 'cal_metrics_on_gpu') and cfg.cal_metrics_on_gpu:
                        t2vr1, t2vr5, t2vr10, t2vmedr, t2vmeanr = tensor_text_to_video_metrics(sim_matrix, return_dict=False)
                        v2tr1, v2tr5, v2tr10, v2tmedr, v2tmeanr = compute_metrics_gpu(tensor_video_to_text_sim(sim_matrix))

                        t2vr1_text2fused_vis, t2vr5_text2fused_vis, t2vr10_text2fused_vis, t2vmedr_text2fused_vis, t2vmeanr_text2fused_vis = tensor_text_to_video_metrics(
                            text2fused_vis_sim_matrix, return_dict=False)
                        v2tr1_text2fused_vis, v2tr5_text2fused_vis, v2tr10_text2fused_vis, v2tmedr_text2fused_vis, v2tmeanr_text2fused_vis = compute_metrics_gpu(
                            tensor_video_to_text_sim(text2fused_vis_sim_matrix))

                        t2vr1_text2fusioner_selected_vis, t2vr5_text2fusioner_selected_vis, t2vr10_text2fusioner_selected_vis, t2vmedr_text2fusioner_selected_vis, t2vmeanr_text2fusioner_selected_vis = tensor_text_to_video_metrics(
                            text2fusioner_selected_vis_sim_matrix, return_dict=False)
                        v2tr1_text2fusioner_selected_vis, v2tr5_text2fusioner_selected_vis, v2tr10_text2fusioner_selected_vis, v2tmedr_text2fusioner_selected_vis, v2tmeanr_text2fusioner_selected_vis = compute_metrics_gpu(
                            tensor_video_to_text_sim(text2fusioner_selected_vis_sim_matrix))

                        t2vr1_text2original_vis, t2vr5_text2original_vis, t2vr10_text2original_vis, t2vmedr_text2original_vis, t2vmeanr_text2original_vis = tensor_text_to_video_metrics(
                            text2original_vis_sim_matrix, return_dict=False)
                        v2tr1_text2original_vis, v2tr5_text2original_vis, v2tr10_text2original_vis, v2tmedr_text2original_vis, v2tmeanr_text2original_vis = compute_metrics_gpu(
                            tensor_video_to_text_sim(text2original_vis_sim_matrix))

                        t2vr1_text2generated_text, t2vr5_text2generated_text, t2vr10_text2generated_text, t2vmedr_text2generated_text, t2vmeanr_text2generated_text = tensor_text_to_video_metrics(
                            text2generated_text_sim_matrix, return_dict=False)
                        v2tr1_text2generated_text, v2tr5_text2generated_text, v2tr10_text2generated_text, v2tmedr_text2generated_text, v2tmeanr_text2generated_text = compute_metrics_gpu(
                            tensor_video_to_text_sim(text2generated_text_sim_matrix))

                    else:
                        sim_matrix = sim_matrix.cpu()
                        text2fused_vis_sim_matrix = text2fused_vis_sim_matrix.cpu()
                        text2generated_text_sim_matrix = text2generated_text_sim_matrix.cpu()
                        text2original_vis_sim_matrix = text2original_vis_sim_matrix.cpu()
                        text2fusioner_selected_vis_sim_matrix = text2fusioner_selected_vis_sim_matrix.cpu()

                        t2vr1, t2vr5, t2vr10, t2vmedr, t2vmeanr = tensor_text_to_video_metrics(sim_matrix, return_dict=False)
                        v2tr1, v2tr5, v2tr10, v2tmedr, v2tmeanr = compute_metrics(tensor_video_to_text_sim(sim_matrix))


                        t2vr1_text2fused_vis, t2vr5_text2fused_vis, t2vr10_text2fused_vis, t2vmedr_text2fused_vis, t2vmeanr_text2fused_vis = tensor_text_to_video_metrics(
                            text2fused_vis_sim_matrix, return_dict=False)
                        v2tr1_text2fused_vis, v2tr5_text2fused_vis, v2tr10_text2fused_vis, v2tmedr_text2fused_vis, v2tmeanr_text2fused_vis = compute_metrics(
                            tensor_video_to_text_sim(text2fused_vis_sim_matrix))

                        t2vr1_text2fusioner_selected_vis, t2vr5_text2fusioner_selected_vis, t2vr10_text2fusioner_selected_vis, t2vmedr_text2fusioner_selected_vis, t2vmeanr_text2fusioner_selected_vis = tensor_text_to_video_metrics(
                            text2fusioner_selected_vis_sim_matrix, return_dict=False)
                        v2tr1_text2fusioner_selected_vis, v2tr5_text2fusioner_selected_vis, v2tr10_text2fusioner_selected_vis, v2tmedr_text2fusioner_selected_vis, v2tmeanr_text2fusioner_selected_vis = compute_metrics(
                            tensor_video_to_text_sim(text2fusioner_selected_vis_sim_matrix))

                        t2vr1_text2original_vis, t2vr5_text2original_vis, t2vr10_text2original_vis, t2vmedr_text2original_vis, t2vmeanr_text2original_vis = tensor_text_to_video_metrics(
                            text2original_vis_sim_matrix, return_dict=False)
                        v2tr1_text2original_vis, v2tr5_text2original_vis, v2tr10_text2original_vis, v2tmedr_text2original_vis, v2tmeanr_text2original_vis = compute_metrics(
                            tensor_video_to_text_sim(text2original_vis_sim_matrix))

                        t2vr1_text2generated_text, t2vr5_text2generated_text, t2vr10_text2generated_text, t2vmedr_text2generated_text, t2vmeanr_text2generated_text = tensor_text_to_video_metrics(
                            text2generated_text_sim_matrix, return_dict=False)
                        v2tr1_text2generated_text, v2tr5_text2generated_text, v2tr10_text2generated_text, v2tmedr_text2generated_text, v2tmeanr_text2generated_text = compute_metrics(
                            tensor_video_to_text_sim(text2generated_text_sim_matrix))

                    if type == 'simple':
                        simple_setting_t2vr1 = t2vr1
                        if cfg.post_training_strategy == 'only_train_generator_at_1stage':
                            simple_setting_t2vr1 = t2vr1_text2generated_text

                    val_log.update({f'valid/{loader_name}_t2v_recall_1': t2vr1,
                                    f'valid/{loader_name}_t2v_recall_5': t2vr5,
                                    f'valid/{loader_name}_t2v_recall_10': t2vr10,
                                    f'valid/{loader_name}_t2v_recall_median': t2vmedr,
                                    f'valid/{loader_name}_t2v_recall_mean': t2vmeanr,
                                    f'valid/{loader_name}_t2v_recall_SumR': t2vr1 + t2vr5 + t2vr10,
                                    f'valid/{loader_name}_v2t_recall_1': v2tr1,
                                    f'valid/{loader_name}_v2t_recall_5': v2tr5,
                                    f'valid/{loader_name}_v2t_recall_10': v2tr10,
                                    f'valid/{loader_name}_v2t_recall_median': v2tmedr,
                                    f'valid/{loader_name}_v2t_recall_mean': v2tmeanr,
                                    f'valid/{loader_name}_v2t_recall_SumR': v2tr1 + v2tr5 + v2tr10,
                                    })

                    LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, "
                                f"validated on {vis_embeds.shape[0]} videos"
                                f"{loader_name} t2v recall@1: {val_log['valid/%s_t2v_recall_1' % (loader_name)] * 100:.4f} "
                                f"{loader_name} t2v recall@5: {val_log['valid/%s_t2v_recall_5' % (loader_name)] * 100:.4f} "
                                f"{loader_name} t2v recall@10: {val_log['valid/%s_t2v_recall_10' % (loader_name)] * 100:.4f} "
                                f"{loader_name} t2v SumR: {val_log['valid/%s_t2v_recall_SumR' % (loader_name)] * 100:.1f} "
                                f"{loader_name} t2v recall_med: {val_log['valid/%s_t2v_recall_median' % (loader_name)] :.1f} "
                                f"{loader_name} t2v recall_mean: {val_log['valid/%s_t2v_recall_mean' % (loader_name)] :.1f} "
                                f"{loader_name} v2t recall@1: {val_log['valid/%s_v2t_recall_1' % (loader_name)] * 100:.4f} "
                                f"{loader_name} v2t recall@5: {val_log['valid/%s_v2t_recall_5' % (loader_name)] * 100:.4f} "
                                f"{loader_name} v2t recall@10: {val_log['valid/%s_v2t_recall_10' % (loader_name)] * 100:.4f} "
                                f"{loader_name} v2t SumR: {val_log['valid/%s_v2t_recall_SumR' %(loader_name)] * 100:.1f} "
                                f"{loader_name} v2t recall_med: {val_log['valid/%s_v2t_recall_median' % (loader_name)] :.1f} "
                                f"{loader_name} v2t recall_mean: {val_log['valid/%s_v2t_recall_mean' % (loader_name)] :.1f} "
                                )
                    embeds_dict = None
                    onlyposs_word2video_res_dict = None


                LOGGER.info('-------------' * 3 + '\n')
                LOGGER.info("{} text2fused_video_feat recall@1: {:.4f}".format(loader_name, t2vr1_text2fused_vis * 100))
                LOGGER.info("{} fused_video_feat2text recall@1: {:.4f}".format(loader_name, v2tr1_text2fused_vis * 100))
                LOGGER.info("{} text2fusioner_selected_video_feat recall@1: {:.4f}".format(loader_name, t2vr1_text2fusioner_selected_vis * 100))
                LOGGER.info("{} fusioner_selected_video_feat2text recall@1: {:.4f}".format(loader_name, v2tr1_text2fusioner_selected_vis * 100))
                LOGGER.info("{} text2original_video_feats recall@1: {:.4f}".format(loader_name, t2vr1_text2original_vis* 100))
                LOGGER.info("{} original_video_feats2text recall@1: {:.4f}".format(loader_name, v2tr1_text2original_vis* 100))
                LOGGER.info("{} text2generated_text_feats recall@1: {:.4f}".format(loader_name, t2vr1_text2generated_text* 100))
                LOGGER.info("{} generated_text_feats2text recall@1: {:.4f}".format(loader_name, v2tr1_text2generated_text* 100))
                LOGGER.info('-------------' * 3 + '\n')

            else:

                sim_matrix = sim_matrix.cpu()
                text2fused_vis_sim_matrix = text2fused_vis_sim_matrix.cpu()
                text2generated_text_sim_matrix = text2generated_text_sim_matrix.cpu()
                text2original_vis_sim_matrix = text2original_vis_sim_matrix.cpu()
                text2fusioner_selected_vis_sim_matrix = text2fusioner_selected_vis_sim_matrix.cpu()

                # res_dict = cal_cossim_multi(text_feats=text_embeds, text_all_feats=text_all_embeds, text_input_mask=text_input_mask, video_feats=original_vis_embeds, video_all_feats=fusioner_selected_vis_embeds, mode='word-video')
                # text_all2video_raw_sim_matrix, text_all2video_sim_matrix = res_dict['sim_matrix'], res_dict['sim_matrix_maxpool']
                # res_dict = cal_cossim_multi(text_feats=text_embeds, text_all_feats=text_all_embeds, text_input_mask=text_input_mask, video_feats=original_vis_embeds, video_all_feats=fusioner_selected_vis_embeds, mode='word-frame')
                # word2frame_raw_sim_matrix, word2frame_sim_matrix = res_dict['sim_matrix'], res_dict['sim_matrix_maxpool']
                # word2patch_res_dict = cal_cossim_multi(text_feats=text_embeds, text_all_feats=text_all_embeds, text_input_mask=text_input_mask, video_feats=original_vis_embeds, video_all_feats=video_all_embeds, mode='word-patch')
                # word2patch_raw_sim_matrix, word2patch_sim_matrix, word2patch_max_index, word2patch_diag_sim_matrix = word2patch_res_dict['sim_matrix'], word2patch_res_dict['sim_matrix_maxpool'], word2patch_res_dict['max_index'], word2patch_res_dict['diag_sim_matrix']
                onlyposs_word2video_res_dict = cal_cossim_multi(text_feats=text_embeds, text_all_feats=text_all_embeds, text_input_mask=text_input_mask, video_feats=original_vis_embeds, video_all_feats=fusioner_selected_vis_embeds, mode='word-video_only_positive')
                onlypos_word2video_raw_sim_matrix, onlypos_word2video_sim_matrix, onlypos_word2video_max_index, onlypos_word2video_diag_sim_matrix = onlyposs_word2video_res_dict['sim_matrix'], onlyposs_word2video_res_dict['sim_matrix_maxpool'], onlyposs_word2video_res_dict['max_index'], onlyposs_word2video_res_dict['diag_sim_matrix']
                #
                # word2video_top3_res_dict = cal_cossim_multi(text_feats=text_embeds, text_all_feats=text_all_embeds, text_input_mask=text_input_mask, video_feats=original_vis_embeds, video_all_feats=fusioner_selected_vis_embeds, mode='word-video_top3')
                # word2video_top3_raw_sim_matrix, word2video_top3_sim_matrix, word2video_top3_max_index, word2video_top3_diag_sim_matrix = word2video_top3_res_dict['sim_matrix'], word2video_top3_res_dict['sim_matrix_maxpool'], word2video_top3_res_dict['max_index'], word2video_top3_res_dict['diag_sim_matrix']

                text2generator_selected_vis_sim_matrix = cal_cossim(text_embeds, generator_selected_vis_embeds)
                # text2fusioner_selected_vis_sim_matrix = cal_cossim(text_embeds, fusioner_selected_vis_embeds_mean)
                # text2original_vis_sim_matrix = cal_cossim(text_embeds, original_vis_embeds)
                # text2generated_text_sim_matrix = cal_cossim(text_embeds, generated_global_text_embeds)
                if cfg.loss_config.loss_name == 'NCELearnableTempLoss_two_matrix':
                    alpha = model.alpha.cpu().numpy()
                    sim_matrix = text2original_vis_sim_matrix + text2fused_vis_sim_matrix * alpha

                # cal metric of mse
                text_embeds = torch.tensor(text_embeds, dtype=float)
                generated_global_text_embeds = torch.tensor(generated_global_text_embeds, dtype=float)
                only_positive_pairs_mse_normed = cal_mse_in_text_to_pseudo(text_embeds, generated_global_text_embeds, normed=False)
                min_mse_normed, mean_mse_normed = cal_mse_in_text_to_text(text_embeds, normed=False)


                LOGGER.info('-------------' * 3)
                LOGGER.info("\t>>>  In Test Set, after norm, Mean of MSE loss is {}".format(only_positive_pairs_mse_normed))
                LOGGER.info("\t>>>  In Test Set, after norm, Mean of Mean MSE of text-to-text is {}".format(mean_mse_normed))
                LOGGER.info("\t>>>  In Test Set, after norm, Mean of Min MSE of text-to-text is {}".format(min_mse_normed))
                # if hasattr(cfg, 'caption_loss_config'):
                #     LOGGER.info("\t>>>  In Test Set, captioning metrics are {}".format(captioning_results))
                LOGGER.info('-------------' * 3)

                for type in ["simple", "DSL"]:
                    LOGGER.info(f"Evaluate under setting: {type}.")
                    val_log = {f'valid/{loader_name}_t2v_recall_1': 0,
                               f'valid/{loader_name}_t2v_recall_5': 0,
                               f'valid/{loader_name}_t2v_recall_10': 0,
                               f'valid/{loader_name}_t2v_recall_median': 0,
                               f'valid/{loader_name}_t2v_recall_mean': 0,
                               f'valid/{loader_name}_t2v_recall_SumR': 0,
                               f'valid/{loader_name}_v2t_recall_1': 0,
                               f'valid/{loader_name}_v2t_recall_5': 0,
                               f'valid/{loader_name}_v2t_recall_10': 0,
                               f'valid/{loader_name}_v2t_recall_median': 0,
                               f'valid/{loader_name}_v2t_recall_mean': 0,
                               f'valid/{loader_name}_v2t_recall_SumR': 0,
                               }

                    if type == "DSL":
                        sim_matrix = sim_matrix * np_softmax(sim_matrix * 100, axis=0)

                    v2tr1, v2tr5, v2tr10, v2tmedr, v2tmeanr = compute_metrics(sim_matrix.T)
                    t2vr1, t2vr5, t2vr10, t2vmedr, t2vmeanr = compute_metrics(sim_matrix)

                    v2tr1_text2fused_vis, v2tr5_text2fused_vis, v2tr10_text2fused_vis, v2tmedr_text2fused_vis, v2tmeanr_text2fused_vis = compute_metrics_zero_division(text2fused_vis_sim_matrix.T)
                    t2vr1_text2fused_vis, t2vr5_text2fused_vis, t2vr10_text2fused_vis, t2vmedr_text2fused_vis, t2vmeanr_text2fused_vis = compute_metrics_zero_division(text2fused_vis_sim_matrix)

                    v2tr1_text2generator_selected_vis, v2tr5_text2generator_selected_vis, v2tr10_text2generator_selected_vis, v2tmedr_text2generator_selected_vis, v2tmeanr_text2generator_selected_vis = compute_metrics_zero_division(text2generator_selected_vis_sim_matrix.T)
                    t2vr1_text2generator_selected_vis, t2vr5_text2generator_selected_vis, t2vr10_text2generator_selected_vis, t2vmedr_text2generator_selected_vis, t2vmeanr_text2generator_selected_vis = compute_metrics_zero_division(text2generator_selected_vis_sim_matrix)

                    v2tr1_text2fusioner_selected_vis, v2tr5_text2fusioner_selected_vis, v2tr10_text2fusioner_selected_vis, v2tmedr_text2fusioner_selected_vis, v2tmeanr_text2fusioner_selected_vis = compute_metrics_zero_division(text2fusioner_selected_vis_sim_matrix.T)
                    t2vr1_text2fusioner_selected_vis, t2vr5_text2fusioner_selected_vis, t2vr10_text2fusioner_selected_vis, t2vmedr_text2fusioner_selected_vis, t2vmeanr_text2fusioner_selected_vis = compute_metrics_zero_division(text2fusioner_selected_vis_sim_matrix)

                    v2tr1_text2original_vis, v2tr5_text2original_vis, v2tr10_text2original_vis, v2tmedr_text2original_vis, v2tmeanr_text2original_vis = compute_metrics_zero_division(text2original_vis_sim_matrix.T)
                    t2vr1_text2original_vis, t2vr5_text2original_vis, t2vr10_text2original_vis, t2vmedr_text2original_vis, t2vmeanr_text2original_vis = compute_metrics_zero_division(text2original_vis_sim_matrix)

                    t2vr1_text2generated_text, t2vr5_text2generated_text, t2vr10_text2generated_text, t2vmedr_text2generated_text, t2vmeanr_text2generated_text = compute_metrics_zero_division(text2generated_text_sim_matrix)
                    v2tr1_text2generated_text, v2tr5_text2generated_text, v2tr10_text2generated_text, v2tmedr_text2generated_text, v2tmeanr_text2generated_text = compute_metrics_zero_division(text2generated_text_sim_matrix.T)

                    # t2vr1_text_all2original_vis, t2vr5_text_all2original_vis, t2vr10_text_all2original_vis, t2vmedr_text_all2original_vis, t2vmeanr_text_all2original_vis = compute_metrics_zero_division(text_all2video_sim_matrix)
                    # v2tr1_text_all2original_vis, v2tr5_text_all2original_vis, v2tr10_text_all2original_vis, v2tmedr_text_all2original_vis, v2tmeanr_text_all2original_vis = compute_metrics_zero_division(text_all2video_sim_matrix.T)
                    #
                    # t2vr1_word2frame, t2vr5_word2frame, t2vr10_word2frame, t2vmedr_word2frame, t2vmeanr_word2frame = compute_metrics_zero_division(word2frame_sim_matrix)
                    # v2tr1_word2frame, v2tr5_word2frame, v2tr10_word2frame, v2tmedr_word2frame, v2tmeanr_word2frame = compute_metrics_zero_division(word2frame_sim_matrix.T)
                    #
                    t2vr1_onlypos_word2video, t2vr5_onlypos_word2video, t2vr10_onlypos_word2video, t2vmedr_onlypos_word2video, t2vmeanr_onlypos_word2video = compute_metrics_zero_division(onlypos_word2video_sim_matrix)
                    v2tr1_onlypos_word2video, v2tr5_onlypos_word2video, v2tr10_onlypos_word2video, v2tmedr_onlypos_word2video, v2tmeanr_onlypos_word2video = compute_metrics_zero_division(onlypos_word2video_sim_matrix.T)
                    #
                    # t2vr1_word2video_top3, t2vr5_word2video_top3, t2vr10_word2video_top3, t2vmedr_word2video_top3, t2vmeanr_word2video_top3 = compute_metrics_zero_division(word2video_top3_sim_matrix)
                    # v2tr1_word2video_top3, v2tr5_word2video_top3, v2tr10_word2video_top3, v2tmedr_word2video_top3, v2tmeanr_word2video_top3 = compute_metrics_zero_division(word2video_top3_sim_matrix.T)

                    # t2vr1_word2patch, t2vr5_word2patch, t2vr10_word2patch, t2vmedr_word2patch, t2vmeanr_word2patch = compute_metrics_zero_division(word2patch_sim_matrix)
                    # v2tr1_word2patch, v2tr5_word2patch, v2tr10_word2patch, v2tmedr_word2patch, v2tmeanr_word2patch = compute_metrics_zero_division(word2patch_sim_matrix.T)


                    if type == 'simple':
                        simple_setting_t2vr1 = t2vr1
                        if cfg.post_training_strategy == 'only_train_generator_at_1stage':
                            simple_setting_t2vr1 = t2vr1_text2generated_text

                    val_log.update({f'valid/{loader_name}_t2v_recall_1': t2vr1,
                                    f'valid/{loader_name}_t2v_recall_5': t2vr5,
                                    f'valid/{loader_name}_t2v_recall_10': t2vr10,
                                    f'valid/{loader_name}_t2v_recall_median': t2vmedr,
                                    f'valid/{loader_name}_t2v_recall_mean': t2vmeanr,
                                    f'valid/{loader_name}_t2v_recall_SumR': t2vr1 + t2vr5 + t2vr10,
                                    f'valid/{loader_name}_v2t_recall_1': v2tr1,
                                    f'valid/{loader_name}_v2t_recall_5': v2tr5,
                                    f'valid/{loader_name}_v2t_recall_10': v2tr10,
                                    f'valid/{loader_name}_v2t_recall_median': v2tmedr,
                                    f'valid/{loader_name}_v2t_recall_mean': v2tmeanr,
                                    f'valid/{loader_name}_v2t_recall_SumR': v2tr1 + v2tr5 + v2tr10,
                                    })

                    LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, "
                                f"validated on {vis_embeds.shape[0]} videos"
                                f"{loader_name} t2v recall@1: {val_log['valid/%s_t2v_recall_1' % (loader_name)] * 100:.4f} "
                                f"{loader_name} t2v recall@5: {val_log['valid/%s_t2v_recall_5' % (loader_name)] * 100:.4f} "
                                f"{loader_name} t2v recall@10: {val_log['valid/%s_t2v_recall_10' % (loader_name)] * 100:.4f} "
                                f"{loader_name} t2v SumR: {val_log['valid/%s_t2v_recall_SumR' % (loader_name)] * 100:.1f} "
                                f"{loader_name} t2v recall_med: {val_log['valid/%s_t2v_recall_median' % (loader_name)] :.1f} "
                                f"{loader_name} t2v recall_mean: {val_log['valid/%s_t2v_recall_mean' % (loader_name)] :.1f} "
                                f"{loader_name} v2t recall@1: {val_log['valid/%s_v2t_recall_1' % (loader_name)] * 100:.4f} "
                                f"{loader_name} v2t recall@5: {val_log['valid/%s_v2t_recall_5' % (loader_name)] * 100:.4f} "
                                f"{loader_name} v2t recall@10: {val_log['valid/%s_v2t_recall_10' % (loader_name)] * 100:.4f} "
                                f"{loader_name} v2t SumR: {val_log['valid/%s_v2t_recall_SumR' %(loader_name)] * 100:.1f} "
                                f"{loader_name} v2t recall_med: {val_log['valid/%s_v2t_recall_median' % (loader_name)] :.1f} "
                                f"{loader_name} v2t recall_mean: {val_log['valid/%s_v2t_recall_mean' % (loader_name)] :.1f} "
                                )

                LOGGER.info('-------------' * 3 + '\n')
                # LOGGER.info(f"{loader_name} text2fused_video_feat recall@1: {t2vr1_text2fused_vis * 100:.4f} ")
                # LOGGER.info(f"{loader_name} fused_video_feat2text recall@1: {v2tr1_text2fused_vis * 100:.4f} ")
                LOGGER.info("{} text2fused_video_feat recall@1: {:.4f}".format(loader_name, t2vr1_text2fused_vis * 100))
                LOGGER.info("{} fused_video_feat2text recall@1: {:.4f}".format(loader_name, v2tr1_text2fused_vis * 100))
                LOGGER.info("{} text2generator_selected_video_feat recall@1: {:.4f}".format(loader_name, t2vr1_text2generator_selected_vis * 100))
                LOGGER.info("{} generator_selected_video_feat2text recall@1: {:.4f}".format(loader_name, v2tr1_text2generator_selected_vis * 100))
                LOGGER.info("{} text2fusioner_selected_video_feat recall@1: {:.4f}".format(loader_name, t2vr1_text2fusioner_selected_vis * 100))
                LOGGER.info("{} fusioner_selected_video_feat2text recall@1: {:.4f}".format(loader_name, v2tr1_text2fusioner_selected_vis * 100))
                LOGGER.info("{} text2original_video_feats recall@1: {:.4f}".format(loader_name, t2vr1_text2original_vis* 100))
                LOGGER.info("{} original_video_feats2text recall@1: {:.4f}".format(loader_name, v2tr1_text2original_vis* 100))
                LOGGER.info("{} text2generated_text_feats recall@1: {:.4f}".format(loader_name, t2vr1_text2generated_text* 100))
                LOGGER.info("{} generated_text_feats2text recall@1: {:.4f}".format(loader_name, v2tr1_text2generated_text* 100))
                # LOGGER.info("{} text_all2original_video_feats recall@1: {:.4f}".format(loader_name, t2vr1_text_all2original_vis* 100))
                # LOGGER.info("{} original_video_feats2text_all recall@1: {:.4f}".format(loader_name, v2tr1_text_all2original_vis* 100))
                # LOGGER.info("{} word2frame recall@1: {:.4f}".format(loader_name, t2vr1_word2frame* 100))
                # LOGGER.info("{} frame2word recall@1: {:.4f}".format(loader_name, v2tr1_word2frame* 100))
                LOGGER.info("{} onlypos_word2video recall@1: {:.4f}".format(loader_name, t2vr1_onlypos_word2video* 100))
                LOGGER.info("{} onlypos_video2word recall@1: {:.4f}".format(loader_name, v2tr1_onlypos_word2video* 100))
                # LOGGER.info("{} word2video_top3 recall@1: {:.4f}".format(loader_name, t2vr1_word2video_top3* 100))
                # LOGGER.info("{} video2word_top3 recall@1: {:.4f}".format(loader_name, v2tr1_word2video_top3* 100))
                # LOGGER.info("{} word2patch recall@1: {:.4f}".format(loader_name, t2vr1_word2patch* 100))
                # LOGGER.info("{} patch2word recall@1: {:.4f}".format(loader_name, v2tr1_word2patch* 100))
                LOGGER.info('-------------' * 3 + '\n')
                TB_LOGGER.log_scalar_dict(val_log)

                LOGGER.info(' Saving attention_scores !!!!!!!!')
                torch.save(torch.from_numpy(attention_scores), join(cfg.output_dir, f'attention_scores_{loader_name}.pt'))
        else:
            val_log = None
            simple_setting_t2vr1 = None
            embeds_dict = None
            onlyposs_word2video_res_dict = None


    model.train()
    # return val_log, t2vr1
    return val_log, simple_setting_t2vr1, embeds_dict, onlyposs_word2video_res_dict

def start_fk_training_amp():
    cfg = shared_configs.get_pretraining_args()
    blob_mount(cfg)
    set_random_seed(cfg.seed)

    n_gpu = hvd.size()
    cfg.n_gpu = n_gpu
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True
    LOGGER.info(f"device: {device} n_gpu: {n_gpu}, "
                f"rank: {hvd.rank()}, 16-bits training: {cfg.fp16}")

    # prepare data
    tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip_config)
    train_loader, val_loaders, inference_loaders = setup_dataloaders(cfg, tokenizer)

    img_norm = None

    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loaders = {k: {'sentence_loader': PrefetchLoader(v['sentence_loader'], img_norm),
                       'video_loader': PrefetchLoader(v['video_loader'], img_norm)}
                   for k, v in val_loaders.items()}
    inference_loaders = {k: {'sentence_loader': PrefetchLoader(v['sentence_loader'], img_norm),
                             'video_loader': PrefetchLoader(v['video_loader'], img_norm)}
                         for k, v in inference_loaders.items()}


    if cfg.thread_not_repeated_sampling:
        LOGGER.info('-'*20)
        LOGGER.info('Using Thread not repeated sampling strategy')
        LOGGER.info('-'*20)

    # compute the number of steps and update cfg
    total_train_batch_size = int(
        n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)

    total_n_examples = len(train_loader.dataset) * cfg.max_n_example_per_group
    print('total_n_examples', total_n_examples)

    cfg.num_train_steps = int(math.ceil(
        1. * cfg.num_train_epochs * total_n_examples / total_train_batch_size))

    cfg.valid_steps = int(math.ceil(
        1. * cfg.num_train_steps / cfg.num_valid /
        cfg.min_valid_steps)) * cfg.min_valid_steps
    actual_num_valid = int(math.floor(
        1. * cfg.num_train_steps / cfg.valid_steps)) + 1

    n_steps_in_epoch = int(math.ceil(1. * total_n_examples / total_train_batch_size))

    if not cfg.do_eval:
        # prepare model
        model = setup_fk_model(cfg, device=device)
        model.train()

        optimizer = setup_e2e_optimizer(model, cfg)

        scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

        # Horovod: (optional) compression algorithm.compressin
        compression = hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters(),
            compression=compression)

        #  Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # restore
        restorer = TrainingRestorer(cfg, model, optimizer)
        global_step = restorer.global_step
        TB_LOGGER.global_step = global_step
        if hvd.rank() == 0:
            LOGGER.info("Saving training meta...")
            save_training_meta(cfg)
            LOGGER.info("Saving training done...")
            if cfg.if_tb_log:
                TB_LOGGER.create(join(cfg.output_dir, 'log'))
            # pbar = tqdm(total=cfg.num_train_steps)
            if cfg.if_model_saver:
                model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
                best_model_saver = BestModelSaver(join(cfg.output_dir, "ckpt"))
            else:
                model_saver = NoOp()
                restorer = NoOp()
                best_model_saver = NoOp()

            if cfg.if_log2file:
                add_log_to_file(join(cfg.output_dir, "log", "log.txt"))
        else:
            LOGGER.disabled = True
            # pbar = NoOp()
            model_saver = NoOp()
            restorer = NoOp()
            best_model_saver = NoOp()

        if global_step > 0:
            pass  # pbar.update(global_step)

        LOGGER.info(cfg)
        LOGGER.info("Starting training...")
        LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
        LOGGER.info(f"  Single-GPU Non-Accumulated batch size = {cfg.train_batch_size}")
        LOGGER.info(f"  max_n_example_per_group = {cfg.max_n_example_per_group}")
        LOGGER.info(f"  Accumulate steps = {cfg.gradient_accumulation_steps}")
        LOGGER.info(f"  Total batch size = #GPUs * Single-GPU batch size * "
                    f"max_n_example_per_group * Accumulate steps [Image] = {total_train_batch_size}")
        LOGGER.info(f"  Total #epochs = {cfg.num_train_epochs}")
        LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
        LOGGER.info(f"  Validate and Save every {cfg.valid_steps} steps, in total {actual_num_valid} times")
        LOGGER.info(f"  Only Validate every {cfg.only_valid_steps} steps")

        # quick hack for amp delay_unscale bug
        with optimizer.skip_synchronize():
            optimizer.zero_grad()
            if global_step == 0:
                optimizer.step()

        running_loss = RunningMeter('train_loss', smooth=0)

        LOGGER.info(f'Step zero: start inference')

        # LOGGER.info(f'Eval on test set')
        # fk_validate(model, tokenizer, inference_loaders, cfg)

        LOGGER.info('-'*20)

        LOGGER.info(f'Eval on validation set')
        fk_validate(model, tokenizer, val_loaders, cfg)


        loss_func = build_loss_func(cfg.loss_config)
        if hasattr(cfg, 'generation_loss_config'):

            # hard code for init clip tokenizer in loss
            setattr(cfg.generation_loss_config, 'clip_config', cfg.clip_config)

            word_generation_loss_fn = build_generation_loss_func(cfg.generation_loss_config)
            eot_generation_loss_fn = build_generation_loss_func(cfg.generation_loss_config)
        if hasattr(cfg, 'second_generation_loss_config'):
            second_generation_loss_fn = build_generation_loss_func(cfg.second_generation_loss_config)
        if hasattr(cfg, 'captioning_loss_config'):
            captioning_loss_func = build_captioning_loss_func(cfg.captioning_loss_config)

        freeze_steps = cfg.freeze_epochs * n_steps_in_epoch
        setattr(cfg, 'freeze_steps', freeze_steps)
        setup_model_grad(cfg, model, cur_steps=-1, requires_grad=False)

        for step, batch in enumerate(InfiniteIterator(train_loader)):
            setup_model_grad(cfg, model, cur_steps=step, requires_grad=True)

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                if 'CLIP4clip' in cfg.clip_vision_additional_config.type:
                    # reshape Time dim to the batch dim
                    B, T, C, H, W = batch['video'].shape
                    batch['video'] = batch['video'].reshape(-1, C, H, W)
                outputs = model(**batch)
                # text_outputs = model.get_text_features(**batch)  # dict
                # video_outputs = model.get_video_features(**batch)  # dict
                # outputs = text_outputs
                # outputs.update(video_outputs)
                text_input_ids = batch['text_input_ids']
                text_input_mask = batch['text_input_mask']
                if cfg.loss_config.if_gather:
                    text_feats = hvd.allgather(outputs['text_feats'].contiguous())
                    text_all_feats = hvd.allgather(outputs['text_all_feats'].contiguous())
                    selected_text_feats = hvd.allgather(outputs['selected_text_feats'].contiguous())
                    text_eos_layers_feats = hvd.allgather(outputs['text_eos_layers_feats'].contiguous())
                    generator_eos_layers_feats = hvd.allgather(outputs['generator_eos_layers_feats'].contiguous())
                    final_vis_feats = hvd.allgather(outputs['final_video_feats'].contiguous())
                    video_feats = hvd.allgather(outputs['video_feats'].contiguous())
                    generated_text_feats = hvd.allgather(outputs['generated_text_feats'].contiguous())
                    generated_global_text_feats = hvd.allgather(outputs['generated_global_text_feats'].contiguous())
                    fusioner_selected_video_feats = hvd.allgather(outputs['fusioner_selected_video_feats'].contiguous())
                    text_input_mask = hvd.allgather(text_input_mask.contiguous())
                    text_input_ids = hvd.allgather(text_input_ids.contiguous())

                    if hasattr(cfg, 'fusioner_config') and cfg.fusioner_config.gt_text2fuse:
                        _, gt_fused_vis_feats = model.fusioner(text_feats, fusioner_selected_video_feats, video_mask=None)

                    if hasattr(cfg, 'captioning_loss_config'):
                        pred_logits = hvd.allgather(outputs['pred_logits'])

                    if cfg.loss_config.loss_name in ["NCELearnableTempLoss", "NCELearnableTempDSLLoss", "NCELearnableTempLoss_mixed", "NCELearnableTempLoss_two_matrix", "NCELearnableTempLossOneside"]:
                        if hasattr(model, 'module'):
                            logit_scale = model.module.encoder.clipmodel.logit_scale
                        else:
                            logit_scale = model.encoder.clipmodel.logit_scale

                        if cfg.loss_config.fusioner_selected_video_feats_loss:
                            fusioner_selected_video_feats = torch.mean(fusioner_selected_video_feats, dim=1)
                            fusioner_selected_video_embeds = fusioner_selected_video_feats / fusioner_selected_video_feats.norm(dim=-1, keepdim=True)
                        text_embeds = text_feats / text_feats.norm(dim=-1, keepdim=True)
                        text_all_embeds = text_all_feats / text_all_feats.norm(dim=-1, keepdim=True)
                        selected_text_embeds = selected_text_feats / selected_text_feats.norm(dim=-1, keepdim=True)
                        final_vis_embeds = final_vis_feats / final_vis_feats.norm(dim=-1, keepdim=True)
                        video_embeds = video_feats / video_feats.norm(dim=-1, keepdim=True)

                        if hasattr(cfg, 'fusioner_config') and cfg.fusioner_config.gt_text2fuse:
                            gt_fused_vis_embeds = gt_fused_vis_feats / gt_fused_vis_feats.norm(dim=-1, keepdim=True)
                        generated_text_embeds = generated_text_feats / generated_text_feats.norm(dim=-1, keepdim=True)
                        generated_global_text_embeds = generated_global_text_feats / generated_global_text_feats.norm(dim=-1, keepdim=True)

                        # retrieval loss
                        if cfg.loss_config.loss_name == "NCELearnableTempLoss_mixed":
                            retrieval_loss = loss_func(final_vis_embeds, text_embeds, gt_fused_vis_embeds, logit_scale)
                        elif cfg.loss_config.loss_name == "NCELearnableTempLoss_two_matrix":
                            alpha = model.alpha
                            retrieval_loss = loss_func(video_embeds, text_embeds, generated_global_text_embeds, alpha, logit_scale)
                        elif cfg.post_training_strategy == "only_train_generator_at_1stage" and step < freeze_steps - 1:
                            retrieval_loss = torch.tensor(0).to(device)
                        else:
                            retrieval_loss = loss_func(final_vis_embeds, text_embeds, logit_scale)

                        if hasattr(cfg, 'generation_loss_config') and cfg.generation_loss_config.use_aux_retrieval_loss:
                            aux_retrieval_loss = loss_func(generated_global_text_embeds, text_embeds, logit_scale)
                        else:
                            aux_retrieval_loss = torch.tensor(0).to(device)

                        if cfg.loss_config.fusioner_selected_video_feats_loss:
                            fusioner_selected_video_feats_loss = loss_func(fusioner_selected_video_embeds, text_embeds, logit_scale)
                        else:
                            fusioner_selected_video_feats_loss = 0

                        # captioning loss
                        if hasattr(cfg, 'captioning_loss_config'):
                            captioning_loss = captioning_loss_func(pred_logits, text_input_ids, text_input_mask)
                        else:
                            captioning_loss = 0

                        # generation loss
                        if hasattr(cfg, 'generation_loss_config'):
                            # in this case: it is word-level supervision
                            if cfg.generator_config.num_query_token > 1:
                                if cfg.generation_loss_config.generation_loss_level == 'topk_words+eot':
                                    word_generation_loss = word_generation_loss_fn(generated_text_feats, selected_text_feats,
                                                                                   text_feat_mask=None,
                                                                                   text_input_ids=None)
                                elif cfg.generation_loss_config.generation_loss_level == 'word+eot':
                                    word_generation_loss = word_generation_loss_fn(generated_text_feats, text_all_feats,
                                                                                   text_feat_mask=text_input_mask,
                                                                                   text_input_ids=text_input_ids)
                                else:
                                    raise NotImplementedError

                                eot_generation_loss = eot_generation_loss_fn(generated_global_text_feats, text_feats, text_feat_mask=None, text_input_ids=None)
                                # scale in word_loss and eot_loss is the same
                                generation_loss = word_generation_loss_fn.generation_loss_scale * (word_generation_loss + eot_generation_loss)
                            else:
                            # in this case: it is eot-level supervision
                                if cfg.generation_loss_config.num_layers == -1:
                                    generation_loss = eot_generation_loss_fn(generated_global_text_feats, text_feats, text_feat_mask=None, text_input_ids=None)
                                    if hasattr(cfg, 'second_generation_loss_config'):
                                        second_generation_loss = second_generation_loss_fn(generated_global_text_feats, text_feats, text_feat_mask=None, text_input_ids=None)
                                        generation_loss = generation_loss + second_generation_loss
                                else:
                                    text_feats = torch.cat([text_eos_layers_feats, text_feats], dim=0)
                                    generated_text_feats = torch.cat([generator_eos_layers_feats, generated_global_text_embeds])
                                    generation_loss = eot_generation_loss_fn(generated_text_feats, text_feats, text_feat_mask=None, text_input_ids=None)
                        else:
                            generation_loss = 0

                        # total loss
                        loss = retrieval_loss + generation_loss + captioning_loss + fusioner_selected_video_feats_loss + aux_retrieval_loss
                    else:
                        raise NotImplementedError
                else:
                    loss = outputs['loss']

                if hasattr(model, 'module'):
                    torch.clamp_(model.module.encoder.clipmodel.logit_scale.data, 0, np.log(200))
                    logit_scale_ = model.module.encoder.clipmodel.logit_scale.data
                else:
                    torch.clamp_(model.encoder.clipmodel.logit_scale.data, 0, np.log(200))
                    logit_scale_ = model.encoder.clipmodel.logit_scale.data

            if step % 10 == 0:
                lr_ = optimizer.param_groups[0]['lr']

                if hasattr(cfg, 'captioning_loss_config'):
                    captioning_loss_scale = captioning_loss_func.captioning_loss_scale
                else:
                    captioning_loss_scale = 1
                if hasattr(cfg, 'generation_loss_config'):
                    generation_loss_scale = eot_generation_loss_fn.generation_loss_scale
                    type = eot_generation_loss_fn.generation_loss_name
                else:
                    generation_loss_scale = 1
                    type = 'None'

                LOGGER.info(
                    'Step {}: loss {} | retrieval_loss {} | unscaled_generation_loss(type: {}) {} | aux_retrieval_loss {} | fusioner_selected_video_feats_loss {} | unscaled_captioning_loss {} | lr {} | logit_scale {}'.format(
                        global_step, loss, retrieval_loss, type, generation_loss / generation_loss_scale , aux_retrieval_loss, fusioner_selected_video_feats_loss, captioning_loss / captioning_loss_scale, lr_,
                        logit_scale_))

            running_loss(loss.item())

            delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0

            if delay_unscale:
                with optimizer.skip_synchronize():
                    scaler.scale(loss).backward()
                    optimizer.synchronize()
            else:
                scaler.scale(loss).backward()
                optimizer.synchronize()


            # optimizer
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                global_step += 1
                TB_LOGGER.log_scalar_dict({'vtc_loss': running_loss.val})
                n_epoch = int(
                    cfg.gradient_accumulation_steps * global_step / n_steps_in_epoch
                )
                lr_this_step = get_lr_sched(
                    global_step, cfg.decay, cfg.learning_rate,
                    cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                    decay_epochs=cfg.step_decay_epochs, multi_step_epoch=n_epoch
                )
                for pg_n, param_group in enumerate(optimizer.param_groups):
                    if pg_n in [0, 1]:
                        param_group['lr'] = cfg.lr_mul * lr_this_step
                    elif pg_n in [2, 3]:
                        param_group['lr'] = lr_this_step

                TB_LOGGER.add_scalar("train/lr", lr_this_step, global_step)

                scaler.unscale_(optimizer)
                if cfg.grad_norm != -1:
                    grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_norm)
                    TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)
                TB_LOGGER.step()

                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                restorer.step()

                # checkpoint
                if global_step % cfg.valid_steps == 0:
                    LOGGER.info(f'Step {global_step}: start validation and Save')
                    _, t2vr1, embeds_dict, onlypos_word2video_dict = fk_validate(model, tokenizer, val_loaders, cfg)
                    model_saver.save(step=global_step, model=model)
                    if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
                        best_model_saver.save(step=global_step, model=model)
                        best_model_saver.bestr1 = t2vr1

                else:
                    if global_step % cfg.only_valid_steps == 0:
                        LOGGER.info(f'Step {global_step}: start inference')
                        _, t2vr1, embeds_dict, onlyposs_word2video_res_dict = fk_validate(model, tokenizer, val_loaders, cfg)
                        if getattr(cfg, 'save_model_steps', 0):
                            model_saver.save(step=global_step, model=model)
                        if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
                            best_model_saver.save(step=global_step, model=model)
                            best_model_saver.bestr1 = t2vr1
                            LOGGER.info('*' + '-' * 20 + '*')
                            LOGGER.info(f'Best R1 is {best_model_saver.bestr1 * 100} at step {global_step}')

                            if embeds_dict is not None:
                                LOGGER.info(f'Saving npy files')

                                text_embeds, original_vis_embeds, vis_embeds = embeds_dict['text_embeds'], embeds_dict[
                                    'original_vis_embeds'], embeds_dict['vis_embeds']
                                os.makedirs(os.path.join(cfg.output_dir + '/saved_embeds'), exist_ok=True)
                                np.save(os.path.join(cfg.output_dir + '/saved_embeds/text_embeds.npy'), text_embeds)
                                np.save(os.path.join(cfg.output_dir + '/saved_embeds/original_vis_embeds.npy'), original_vis_embeds)
                                np.save(os.path.join(cfg.output_dir + '/saved_embeds/vis_embeds.npy'), vis_embeds)

                                if hasattr(cfg, 'generator_config'):
                                    generated_global_text_embeds = embeds_dict['generated_global_text_embeds']
                                    fused_vis_embeds = embeds_dict['fused_vis_embeds']
                                    generator_selected_vis_embeds = embeds_dict['generator_selected_vis_embeds']
                                    fusioner_selected_vis_embeds = embeds_dict['fusioner_selected_vis_embeds']

                                    np.save(os.path.join(cfg.output_dir + '/saved_embeds/generated_global_text_embeds.npy'), generated_global_text_embeds)
                                    np.save(os.path.join(cfg.output_dir + '/saved_embeds/fused_vis_embeds.npy'), fused_vis_embeds)
                                    np.save(os.path.join(cfg.output_dir + '/saved_embeds/generator_selected_vis_embeds.npy'), generator_selected_vis_embeds)
                                    np.save(os.path.join(cfg.output_dir + '/saved_embeds/fusioner_selected_vis_embeds.npy'), fusioner_selected_vis_embeds)

                            if hasattr(cfg, 'fusioner'):
                                os.makedirs(os.path.join(cfg.output_dir + '/fusioner_attention'), exist_ok=True)
                                if getattr(cfg.fusioner, 'fusioner_type') == 'XPoolCrossAttention':
                                    model.fusioner.cross_attn


                                if hasattr(cfg.fusioner, 'attention'):
                                    for i, att in enumerate(cfg.fusioner.attention):
                                        np.save(os.path.join(cfg.output_dir + '/fusioner_attention/att_{}.npy'.format(i)), att)

                            # LOGGER.info(f'Saving onlypos_word2video_res_dict')
                            # os.makedirs(os.path.join(cfg.output_dir + '/onlypos_word2video_res_dict'), exist_ok=True)
                            #
                            # # to list to can be saved by yaml
                            # onlyposs_word2video_res_dict['sim_matrix'] = onlyposs_word2video_res_dict['sim_matrix'].tolist()
                            # onlyposs_word2video_res_dict['sim_matrix_maxpool'] = onlyposs_word2video_res_dict['sim_matrix_maxpool'].tolist()
                            # onlyposs_word2video_res_dict['max_index'] = onlyposs_word2video_res_dict['max_index'].tolist()
                            # if isinstance(onlyposs_word2video_res_dict['diag_sim_matrix'], np.ndarray):
                            #     onlyposs_word2video_res_dict['diag_sim_matrix'] = onlyposs_word2video_res_dict['diag_sim_matrix'].tolist()
                            #
                            # with open(cfg.output_dir + '/onlypos_word2video_res_dict/res_dict.yaml', 'w') as f:
                            #     yaml.dump(onlyposs_word2video_res_dict, f, allow_unicode=True)


            if global_step >= cfg.num_train_steps:
                break

        if global_step % cfg.valid_steps != 0:
            LOGGER.info(f'Step {global_step}: start validation')
            _, t2vr1, embeds_dict, onlyposs_word2video_res_dict = fk_validate(model, tokenizer, val_loaders, cfg)

            model_saver.save(step=global_step, model=model)
            if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
                best_model_saver.save(step=global_step, model=model)
                best_model_saver.bestr1 = t2vr1
                LOGGER.info(f'Best R1 is {best_model_saver.bestr1 * 100} at step {global_step}')

    if hvd.rank() == 0 and cfg.if_log2file:
        os.makedirs(os.path.join(cfg.output_dir, "log"), exist_ok=True)
        add_log_to_file(join(cfg.output_dir, "log", "debug.txt"))
    else:
        LOGGER.disabled = True

    LOGGER.info('---------------------------')
    LOGGER.info('Start inference on test set using the best model on validation set')
    LOGGER.info('---------------------------')

    model = FK(cfg)
    load_state_dict_with_mismatch(model, os.path.join(cfg.output_dir, 'ckpt/model_best.pt'))
    model.to(device)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    LOGGER.info('Eval on test dataset')
    _, t2vr1, embeds_dict, onlyposs_word2video_res_dict = fk_validate(model, tokenizer, inference_loaders, cfg)
    LOGGER.info('---------------------------')
    LOGGER.info('Eval on validation dataset')
    _, t2vr1, embeds_dict, onlyposs_word2video_res_dict = fk_validate(model, tokenizer, val_loaders, cfg)

def blob_mount(cfg):
    keys = ["e2e_weights_path",
            "output_dir"]
    for key in keys:
        if cfg[key] is not None:
            cfg[key] = os.path.join(cfg.blob_mount_dir, cfg[key])

    db = cfg.train_datasets
    db.txt = os.path.join(cfg.blob_mount_dir, db.txt)
    db.vis = os.path.join(cfg.blob_mount_dir, db.vis)

    for db in cfg.val_datasets:
        db.txt = os.path.join(cfg.blob_mount_dir, db.txt)
        db.vis = os.path.join(cfg.blob_mount_dir, db.vis)

    for db in cfg.inference_datasets:
        db.txt = os.path.join(cfg.blob_mount_dir, db.txt)
        db.vis = os.path.join(cfg.blob_mount_dir, db.vis)


def cal_mse_in_text_to_text(text_feats, normed=True):
    batch_size, hidden_dim = text_feats.shape

    if normed:
        normed_text_feat = text_feats / text_feats.norm(dim=-1, keepdim=True)
    else:
        normed_text_feat = text_feats

    mse_matrix = torch.sub(normed_text_feat.unsqueeze(1), normed_text_feat).pow(2)
    # [bs, bs]
    mse_matrix_mean_reduction = torch.sum(mse_matrix, dim=-1, keepdim=False) / hidden_dim

    # get only negative pairs
    mask = ~torch.eye(batch_size, dtype=torch.bool)
    # [bs, bs-1]
    mse_matrix_mean_reduction = mse_matrix_mean_reduction[mask].view(batch_size, batch_size - 1)

    min_mse_list, _ = torch.min(mse_matrix_mean_reduction, dim=-1, keepdim=False)
    min_mse = np.mean(min_mse_list.numpy())
    mean_mse_list = torch.mean(mse_matrix, dim=-1, keepdim=False)
    mean_mse = np.mean(mean_mse_list.numpy())
    return min_mse, mean_mse


def cal_mse_in_text_to_pseudo(text_feats, pseudo_text_feats, normed=True):
    if normed:
        normed_text_feat = text_feats / text_feats.norm(dim=-1, keepdim=True)
        normed_pseudo_text_feat = pseudo_text_feats / pseudo_text_feats.norm(dim=-1, keepdim=True)
    else:
        normed_text_feat = text_feats
        normed_pseudo_text_feat = pseudo_text_feats

    batch_size, hidden_dim = normed_text_feat.shape
    mse_list = torch.sub(normed_text_feat, normed_pseudo_text_feat).pow(2)
    mse_list_mean_reduction = torch.sum(mse_list, dim=-1, keepdim=False) / hidden_dim
    mse_list_mean_reduction = mse_list_mean_reduction.numpy()
    only_positve_mse = np.mean(mse_list_mean_reduction)
    return only_positve_mse

def cut_and_map_tokens(tokens, eos_token_id, tokenizer):
    # post processing
    out_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        # cut to eos tok if any
        if eos_token_id in toks:
            eos_idx = toks.index(eos_token_id)
            toks = toks[:eos_idx+1]
        out_tokens.append(toks)

    sents = tokenizer.batch_decode(out_tokens)
    cleaned_sents = tokenizer.batch_decode(out_tokens, skip_special_tokens=True)

    return out_tokens, sents, cleaned_sents

def reshape_similarity_matrix(sim_matrix, cut_off_points):
    """
    Reshapes the similarity matrix by grouping rows according to cut-off points and
    padding each group to the maximum group length with -inf.

    Args:
        sim_matrix (np.ndarray): Original similarity matrix of shape (N, M).
        cut_off_points (list): List of cut-off indices for grouping rows.

    Returns:
        torch.Tensor: Reshaped similarity matrix of shape (G, max_length, M), where:
                      - G is the number of groups defined by cut_off_points.
                      - max_length is the length of the longest group.
                      - M is the original number of columns in sim_matrix.
    """
    # Adjust cut-off points for indexing
    cut_off_points2len = [itm + 1 for itm in cut_off_points]

    # Determine the maximum group length
    max_length = max([e - s for s, e in zip([0] + cut_off_points2len[:-1], cut_off_points2len)])

    # Create a list to store reshaped groups
    sim_matrix_new = []

    # Reshape and pad each group
    for s, e in zip([0] + cut_off_points2len[:-1], cut_off_points2len):
        group = sim_matrix[s:e]
        padding = np.full((max_length - (e - s), sim_matrix.shape[1]), -np.inf)
        sim_matrix_new.append(np.concatenate((group, padding), axis=0))

    # Stack the reshaped groups into a single array
    sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
    sim_matrix = torch.tensor(sim_matrix, dtype=torch.float)


    return sim_matrix



def reshape_similarity_matrix_pytorch(sim_matrix, cut_off_points):
    """
    Reshapes the similarity matrix by grouping rows according to cut-off points and
    padding each group to the maximum group length with -inf.

    Args:
        sim_matrix (torch.Tensor): Original similarity matrix of shape (N, M).
        cut_off_points (list): List of cut-off indices for grouping rows.

    Returns:
        torch.Tensor: Reshaped similarity matrix of shape (G, max_length, M), where:
                      - G is the number of groups defined by cut_off_points.
                      - max_length is the length of the longest group.
                      - M is the original number of columns in sim_matrix.
    """
    # Adjust cut-off points for indexing
    cut_off_points2len = [itm + 1 for itm in cut_off_points]

    # Determine the maximum group length
    max_length = max([e - s for s, e in zip([0] + cut_off_points2len[:-1], cut_off_points2len)])

    # Create a list to store reshaped groups
    sim_matrix_new = []

    # Reshape and pad each group
    start_idx = 0
    for end_idx in cut_off_points2len:
        group = sim_matrix[start_idx:end_idx]
        padding = torch.full((max_length - group.shape[0], sim_matrix.shape[1]), float('-inf'), device=group.device)
        sim_matrix_new.append(torch.cat((group, padding), dim=0))
        start_idx = end_idx

    # Stack the reshaped groups into a single tensor
    sim_matrix_new = torch.stack(sim_matrix_new, dim=0)

    return sim_matrix_new


def setup_model_grad(cfg, model, cur_steps, requires_grad):
    if hasattr(cfg, 'post_training_strategy'):
        if cfg.post_training_strategy == 'freeze_clip_encoder_at_2stage':
            if cur_steps == cfg.freeze_clip_encoder_steps - 1 or cur_steps == -1:
                for n, param in model.encoder.named_parameters():
                    param.requires_grad = requires_grad

                if hvd.rank() == 0:
                    if requires_grad is False:
                        LOGGER.info('--- freeze clip encoder for {} steps ---'.format(cfg.freeze_clip_encoder_steps))
                    else:
                        LOGGER.info(f'--- clip encoder grads state is : {requires_grad} ---')
        elif cfg.post_training_strategy == 'freeze_fusioner_at_2stage':
            if cur_steps == cfg.freeze_clip_encoder_steps - 1 or cur_steps == -1:
                for n, param in model.fusioner.named_parameters():
                    param.requires_grad = requires_grad

                if hvd.rank() == 0:
                    if requires_grad is False:
                        LOGGER.info('--- freeze fusioner for {} steps ---'.format(cfg.freeze_fusioner_steps))
                    else:
                        LOGGER.info(f'--- fusioner grads state is : {requires_grad} ---')
        elif cfg.post_training_strategy == 'only_train_generator_at_1stage':
            if cur_steps == cfg.freeze_steps - 1 or cur_steps == -1:
                for n, param in model.encoder.named_parameters():
                    param.requires_grad = requires_grad

                for n, param in model.fusioner.named_parameters():
                    param.requires_grad = requires_grad

                if 'AFA' in [n for n, _ in model.named_parameters()]:
                    for n, param in model.AFA.named_parameters():
                        param.requires_grad = requires_grad

                if hvd.rank() == 0:
                    if requires_grad is False:
                        LOGGER.info('--- freeze clip, fusioner and AFA for {} steps ---'.format(cfg.freeze_steps))
                    else:
                        LOGGER.info(f'--- grads state is : {requires_grad} ---')
        elif cfg.post_training_strategy == 'only_train_generator_and_fusioner':
            if cur_steps == -1:
                if hvd.rank() == 0:
                    LOGGER.info('--- only train generator and fusioner for {} steps ---'.format(cfg.num_train_steps))

            for n, param in model.encoder.named_parameters():
                param.requires_grad = False
    else:
        pass





if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    start_fk_training_amp()


