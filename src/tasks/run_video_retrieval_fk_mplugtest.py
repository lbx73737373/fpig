import os

print(os.getcwd())
import time
import random
import math
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from easydict import EasyDict as edict
import ipdb

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as tutils
from torch.utils.data.distributed import DistributedSampler

from apex import amp
import torch.nn as nn
import horovod.torch as hvd
from transformers import CLIPTokenizerFast

from src.modeling.VidCLIP import VidCLIP

from src.datasets.dataset_video_retrieval import (
    HDVILAVideoRetrievalDataset, HDVILAVideoRetrievalWithAuxiliaryTextsDataset, VideoRetrievalCollator)
from src.datasets.dataloader import InfiniteIterator, PrefetchLoader
from src.datasets.data_utils import ImageNorm, mk_input_group

from src.configs.config import shared_configs
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
from src.optimization.loss import build_loss_func, build_generation_loss_func
from src.utils.distributed import all_gather_list
from src.utils.metrics import cal_cossim, compute_metrics, compute_metrics_multi, np_softmax

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
        val_loaders[db.name] = mk_video_ret_dataloader(
            dataset_name=db.name, vis_format=db.vis_format,
            anno_path=db.txt, auxiliary_anno_path=db.auxiliary_txt, vis_dir=db.vis,
            cfg=cfg, tokenizer=tokenizer, mode="val"
        )

    inference_loaders = {}
    for db in cfg.inference_datasets:
        inference_loaders[db.name] = mk_video_ret_dataloader(
            dataset_name=db.name, vis_format=db.vis_format,
            anno_path=db.txt, auxiliary_anno_path=db.auxiliary_txt, vis_dir=db.vis,
            cfg=cfg, tokenizer=tokenizer, mode="test"
        )
    return train_loader, val_loaders, inference_loaders


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")

    model = VidCLIP(cfg)

    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)

    if cfg.clip_vision_additional_config.keep_frame_cls:
        # hard coding for init frame cls
        model.clipmodel.vision_model.embeddings.cls_in_frames = nn.Parameter(
            model.clipmodel.vision_model.embeddings.class_embedding.detach().clone())

    if hasattr(cfg, "overload_logit_scale"):
        model.overload_logit_scale(cfg.overload_logit_scale)

    model.to(device)

    LOGGER.info("Setup model done!")
    return model

def setup_fk_model(cfg, device=None):
    LOGGER.info("Setup model...")

    # model = VidCLIP(cfg)
    model = FK(cfg)

    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)

    if cfg.clip_vision_additional_config.keep_frame_cls:
        # hard coding for init frame cls
        model.encoder.clipmodel.vision_model.embeddings.cls_in_frames = nn.Parameter(
            model.encoder.clipmodel.vision_model.embeddings.class_embedding.detach().clone())

    if hasattr(cfg, "overload_logit_scale"):
        model.overload_logit_scale(cfg.overload_logit_scale)

    model.to(device)

    LOGGER.info("Setup model done!")
    return model

@torch.no_grad()
def validate(model, val_loaders, cfg):
    model.eval()

    st = time.time()

    for loader_name, val_loader in val_loaders.items():
        LOGGER.info(f"Loop val_loader {loader_name}.")
        valid_len = len(val_loader.dataset)
        text_feats = []
        vis_feats = []
        for val_step, batch in enumerate(val_loader):
            feats = model(**batch)  # dict
            # print('feats vis_features', feats['vis_features'].shape)
            vis_feat = hvd.allgather(feats['vis_features'])
            text_feat = hvd.allgather(feats['text_features'])


            # print('allgather vis_features', vis_feat.shape)

            text_feats.append(text_feat.cpu().numpy())
            vis_feats.append(vis_feat.cpu().numpy())

        # # Gather across all processes
        # text_feats = all_gather_list(text_feats)
        # vis_feats = all_gather_list(vis_feats)

        text_feats = np.vstack(text_feats)
        vis_feats = np.vstack(vis_feats)

        text_feats = text_feats[:valid_len]
        vis_feats = vis_feats[:valid_len]

        sim_matrix = cal_cossim(text_feats, vis_feats)

        for type in ["simple", "DSL"]:
            LOGGER.info(f"Evaluate under setting: {type}.")
            val_log = {f'valid/{loader_name}_t2v_recall_1': 0,
                       f'valid/{loader_name}_t2v_recall_5': 0,
                       f'valid/{loader_name}_t2v_recall_10': 0,
                       f'valid/{loader_name}_t2v_recall_median': 0,
                       f'valid/{loader_name}_t2v_recall_mean': 0,
                       f'valid/{loader_name}_v2t_recall_1': 0,
                       f'valid/{loader_name}_v2t_recall_5': 0,
                       f'valid/{loader_name}_v2t_recall_10': 0,
                       f'valid/{loader_name}_v2t_recall_median': 0,
                       f'valid/{loader_name}_v2t_recall_mean': 0}

            if type == "DSL":
                sim_matrix = sim_matrix * np_softmax(sim_matrix * 100, axis=0)

            v2tr1, v2tr5, v2tr10, v2tmedr, v2tmeanr = compute_metrics(sim_matrix.T)
            t2vr1, t2vr5, t2vr10, t2vmedr, t2vmeanr = compute_metrics(sim_matrix)

            val_log.update({f'valid/{loader_name}_t2v_recall_1': t2vr1,
                            f'valid/{loader_name}_t2v_recall_5': t2vr5,
                            f'valid/{loader_name}_t2v_recall_10': t2vr10,
                            f'valid/{loader_name}_t2v_recall_median': t2vmedr,
                            f'valid/{loader_name}_t2v_recall_mean': t2vmeanr,
                            f'valid/{loader_name}_v2t_recall_1': v2tr1,
                            f'valid/{loader_name}_v2t_recall_5': v2tr5,
                            f'valid/{loader_name}_v2t_recall_10': v2tr10,
                            f'valid/{loader_name}_v2t_recall_median': v2tmedr,
                            f'valid/{loader_name}_v2t_recall_mean': v2tmeanr
                            })

            LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, "
                        f"validated on {vis_feats.shape[0]} videos"
                        f"{loader_name} t2v recall@1: {val_log['valid/%s_t2v_recall_1' % (loader_name)] * 100:.4f} "
                        f"{loader_name} t2v recall@5: {val_log['valid/%s_t2v_recall_5' % (loader_name)] * 100:.4f} "
                        f"{loader_name} t2v recall@10: {val_log['valid/%s_t2v_recall_10' % (loader_name)] * 100:.4f} "
                        f"{loader_name} t2v recall_med: {val_log['valid/%s_t2v_recall_median' % (loader_name)] :.1f} "
                        f"{loader_name} t2v recall_mean: {val_log['valid/%s_t2v_recall_mean' % (loader_name)] :.1f} "
                        f"{loader_name} v2t recall@1: {val_log['valid/%s_v2t_recall_1' % (loader_name)] * 100:.4f} "
                        f"{loader_name} v2t recall@5: {val_log['valid/%s_v2t_recall_5' % (loader_name)] * 100:.4f} "
                        f"{loader_name} v2t recall@10: {val_log['valid/%s_v2t_recall_10' % (loader_name)] * 100:.4f} "
                        f"{loader_name} v2t recall_med: {val_log['valid/%s_v2t_recall_median' % (loader_name)] :.1f} "
                        f"{loader_name} v2t recall_mean: {val_log['valid/%s_v2t_recall_mean' % (loader_name)] :.1f} "
                        )
        TB_LOGGER.log_scalar_dict(val_log)
    model.train()
    return val_log, t2vr1


@torch.no_grad()
def fk_validate(model, val_loaders, cfg):
    model.eval()

    st = time.time()

    for loader_name, val_loader in val_loaders.items():
        LOGGER.info(f"Loop val_loader {loader_name}.")
        valid_len = len(val_loader.dataset)
        text_embeds_list = []
        final_vis_embeds_list = []
        generated_global_text_embeds_list = []
        fused_vis_embeds_list = []
        selected_vis_embeds_list = []
        for val_step, batch in enumerate(val_loader):
            outputs = model(**batch)  # dict
            # print('feats vis_features', feats['vis_features'].shape)
            vis_feats = hvd.allgather(outputs['video_feats'])
            text_feats = hvd.allgather(outputs['text_feats'])
            final_vis_feats = hvd.allgather(outputs['final_video_feats'])
            # generated_text_feats = hvd.allgather(outputs['generated_text_feats'])
            generated_global_text_feats = hvd.allgather(outputs['generated_global_text_feats'])
            fused_vis_feats = hvd.allgather(outputs['fused_video_feats'])
            selected_vis_feats = hvd.allgather(outputs['selected_video_feats'])
            # selected_vis_feats = selected_vis_feats.mean(dim=1)

            final_vis_embeds = final_vis_feats / final_vis_feats.norm(dim=-1, keepdim=True)
            text_embeds = text_feats / text_feats.norm(dim=-1, keepdim=True)
            generated_global_text_embeds = generated_global_text_feats / generated_global_text_feats.norm(dim=-1, keepdim=True)
            fused_vis_embeds = fused_vis_feats / fused_vis_feats.norm(dim=-1, keepdim=True)
            selected_vis_embeds = selected_vis_feats / selected_vis_feats.norm(dim=-1, keepdim=True)

            # print('allgather vis_features', vis_feat.shape)

            text_embeds_list.append(text_embeds.cpu().numpy())
            final_vis_embeds_list.append(final_vis_embeds.cpu().numpy())
            generated_global_text_embeds_list.append(generated_global_text_embeds.cpu().numpy())
            fused_vis_embeds_list.append(fused_vis_embeds.cpu().numpy())
            selected_vis_embeds_list.append(selected_vis_embeds.cpu().numpy())


        text_embeds = np.vstack(text_embeds_list)
        vis_embeds = np.vstack(final_vis_embeds_list)
        generated_global_text_embeds = np.vstack(generated_global_text_embeds_list)
        fused_vis_embeds = np.vstack(fused_vis_embeds_list)
        selected_vis_embeds = np.vstack(selected_vis_embeds_list)

        text_embeds = text_embeds[:valid_len]
        vis_embeds = vis_embeds[:valid_len]
        generated_global_text_embeds = generated_global_text_embeds[:valid_len]
        fused_vis_embeds = fused_vis_embeds[:valid_len]
        selected_vis_embeds = selected_vis_embeds[:valid_len]

        sim_matrix = cal_cossim(text_embeds, vis_embeds)
        text2fused_vis_sim_matrix = cal_cossim(text_embeds, fused_vis_embeds)
        text2selected_vis_sim_matrix = cal_cossim(text_embeds, selected_vis_embeds)

        # cal metric of mse
        text_embeds = torch.tensor(text_embeds, dtype=float)
        generated_global_text_embeds = torch.tensor(generated_global_text_embeds, dtype=float)
        only_positive_pairs_mse_normed = cal_mse_in_text_to_pseudo(text_embeds, generated_global_text_embeds, normed=False)
        min_mse_normed, mean_mse_normed = cal_mse_in_text_to_text(text_embeds, normed=False)


        LOGGER.info('-------------' * 3)
        LOGGER.info("\t>>>  In Test Set, after norm, Mean of MSE loss is {}".format(only_positive_pairs_mse_normed))
        LOGGER.info("\t>>>  In Test Set, after norm, Mean of Mean MSE of text-to-text is {}".format(mean_mse_normed))
        LOGGER.info("\t>>>  In Test Set, after norm, Mean of Min MSE of text-to-text is {}".format(min_mse_normed))
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

            v2tr1_text2fused_vis, v2tr5_text2fused_vis, v2tr10_text2fused_vis, v2tmedr_text2fused_vis, v2tmeanr_text2fused_vis = compute_metrics(text2fused_vis_sim_matrix.T)
            t2vr1_text2fused_vis, t2vr5_text2fused_vis, t2vr10_text2fused_vis, t2vmedr_text2fused_vis, t2vmeanr_text2fused_vis = compute_metrics(text2fused_vis_sim_matrix)

            v2tr1_text2selected_vis, v2tr5_text2selected_vis, v2tr10_text2selected_vis, v2tmedr_text2selected_vis, v2tmeanr_text2selected_vis = compute_metrics(text2selected_vis_sim_matrix.T)
            t2vr1_text2selected_vis, t2vr5_text2selected_vis, t2vr10_text2selected_vis, t2vmedr_text2selected_vis, t2vmeanr_text2selected_vis = compute_metrics(text2selected_vis_sim_matrix)


            if type == 'simple':
                simple_setting_t2vr1 = t2vr1

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
                        f"{loader_name} t2v recall_med: {val_log['valid/%s_t2v_recall_median' % (loader_name)] :.1f} "
                        f"{loader_name} t2v recall_mean: {val_log['valid/%s_t2v_recall_mean' % (loader_name)] :.1f} "
                        f"{loader_name} v2t recall@1: {val_log['valid/%s_v2t_recall_1' % (loader_name)] * 100:.4f} "
                        f"{loader_name} v2t recall@5: {val_log['valid/%s_v2t_recall_5' % (loader_name)] * 100:.4f} "
                        f"{loader_name} v2t recall@10: {val_log['valid/%s_v2t_recall_10' % (loader_name)] * 100:.4f} "
                        f"{loader_name} v2t recall_med: {val_log['valid/%s_v2t_recall_median' % (loader_name)] :.1f} "
                        f"{loader_name} v2t recall_mean: {val_log['valid/%s_v2t_recall_mean' % (loader_name)] :.1f} "
                        )

        LOGGER.info('-------------' * 3 + '\n')
        # LOGGER.info(f"{loader_name} text2fused_video_feat recall@1: {t2vr1_text2fused_vis * 100:.4f} ")
        # LOGGER.info(f"{loader_name} fused_video_feat2text recall@1: {v2tr1_text2fused_vis * 100:.4f} ")
        LOGGER.info("{} text2fused_video_feat recall@1: {:.4f}".format(loader_name, t2vr1_text2fused_vis * 100))
        LOGGER.info("{} fused_video_feat2text recall@1: {:.4f}".format(loader_name, v2tr1_text2fused_vis * 100))
        LOGGER.info("{} text2selected_video_feat recall@1: {:.4f}".format(loader_name, t2vr1_text2selected_vis * 100))
        LOGGER.info("{} selected_video_feat2text recall@1: {:.4f}".format(loader_name, v2tr1_text2selected_vis * 100))
        LOGGER.info('-------------' * 3 + '\n')
        TB_LOGGER.log_scalar_dict(val_log)
    model.train()
    # return val_log, t2vr1
    return val_log, simple_setting_t2vr1

# def start_training():
#     cfg = shared_configs.get_pretraining_args()
#     blob_mount(cfg)
#     set_random_seed(cfg.seed)
#
#     n_gpu = hvd.size()
#     cfg.n_gpu = n_gpu
#     device = torch.device("cuda", hvd.local_rank())
#     torch.cuda.set_device(hvd.local_rank())
#     if hvd.rank() != 0:
#         LOGGER.disabled = True
#     LOGGER.info(f"device: {device} n_gpu: {n_gpu}, "
#                 f"rank: {hvd.rank()}, 16-bits training: {cfg.fp16}")
#
#     if hvd.rank() != 0:
#         LOGGER.disabled = True
#
#     model = setup_model(cfg, device=device)
#     model.train()
#
#     optimizer = setup_e2e_optimizer(model, cfg)
#
#     # Horovod: (optional) compression algorithm.compressin
#     compression = hvd.Compression.none
#     optimizer = hvd.DistributedOptimizer(
#         optimizer, named_parameters=model.named_parameters(),
#         compression=compression)
#
#     #  Horovod: broadcast parameters & optimizer state.
#     hvd.broadcast_parameters(model.state_dict(), root_rank=0)
#     hvd.broadcast_optimizer_state(optimizer, root_rank=0)
#
#     model, optimizer = amp.initialize(
#         model, optimizer, enabled=cfg.fp16, opt_level=cfg.amp_level,
#         keep_batchnorm_fp32=True if cfg.amp_level=='O2' else None)
#
#     # prepare data
#     tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip_config)
#     train_loader, val_loaders, inference_loaders = setup_dataloaders(cfg, tokenizer)
#
#     img_norm = None
#     train_loader = PrefetchLoader(train_loader, img_norm)
#     val_loaders = {k: PrefetchLoader(v, img_norm)
#                 for k, v in val_loaders.items()}
#     inference_loaders = {k: PrefetchLoader(v, img_norm)
#                 for k, v in inference_loaders.items()}
#
#     # compute the number of steps and update cfg
#     total_train_batch_size = int(
#         n_gpu * cfg.train_batch_size *
#         cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)
#
#     total_n_examples = len(train_loader.dataset) * cfg.max_n_example_per_group
#     print('total_n_examples', total_n_examples)
#
#     cfg.num_train_steps = int(math.ceil(
#         1. * cfg.num_train_epochs * total_n_examples / total_train_batch_size))
#
#     cfg.valid_steps = int(math.ceil(
#         1. * cfg.num_train_steps / cfg.num_valid /
#         cfg.min_valid_steps)) * cfg.min_valid_steps
#     actual_num_valid = int(math.floor(
#         1. * cfg.num_train_steps / cfg.valid_steps)) + 1
#
#     n_steps_in_epoch = int(math.ceil(1. * total_n_examples / total_train_batch_size))
#
#     # restore
#     restorer = TrainingRestorer(cfg, model, optimizer)
#     global_step = restorer.global_step
#     TB_LOGGER.global_step = global_step
#     if hvd.rank() == 0:
#         LOGGER.info("Saving training meta...")
#         save_training_meta(cfg)
#         LOGGER.info("Saving training done...")
#         if cfg.if_tb_log:
#             TB_LOGGER.create(join(cfg.output_dir, 'log'))
#         # pbar = tqdm(total=cfg.num_train_steps)
#         if cfg.if_model_saver:
#             model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
#             best_model_saver = BestModelSaver(join(cfg.output_dir, "ckpt"))
#         else:
#             model_saver = NoOp()
#             restorer = NoOp()
#             best_model_saver = NoOp()
#
#         if cfg.if_log2file:
#             add_log_to_file(join(cfg.output_dir, "log", "log.txt"))
#     else:
#         LOGGER.disabled = True
#         # pbar = NoOp()
#         model_saver = NoOp()
#         restorer = NoOp()
#         best_model_saver = NoOp()
#
#     if global_step > 0:
#         pass # pbar.update(global_step)
#
#     LOGGER.info(cfg)
#     LOGGER.info("Starting training...")
#     LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
#     LOGGER.info(f"  Single-GPU Non-Accumulated batch size = {cfg.train_batch_size}")
#     LOGGER.info(f"  max_n_example_per_group = {cfg.max_n_example_per_group}")
#     LOGGER.info(f"  Accumulate steps = {cfg.gradient_accumulation_steps}")
#     LOGGER.info(f"  Total batch size = #GPUs * Single-GPU batch size * "
#                 f"max_n_example_per_group * Accumulate steps [Image] = {total_train_batch_size}")
#     LOGGER.info(f"  Total #epochs = {cfg.num_train_epochs}")
#     LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
#     LOGGER.info(f"  Validate and Save every {cfg.valid_steps} steps, in total {actual_num_valid} times")
#     LOGGER.info(f"  Only Validate every {cfg.only_valid_steps} steps")
#
#     # quick hack for amp delay_unscale bug
#     with optimizer.skip_synchronize():
#         optimizer.zero_grad()
#         if global_step == 0:
#             optimizer.step()
#
#     running_loss = RunningMeter('train_loss', smooth=0)
#
#     LOGGER.info(f'Step zero: start inference')
#     validate(model, inference_loaders, cfg)
#
#     loss_func = build_loss_func(cfg.loss_config)
#
#     for step, batch in enumerate(InfiniteIterator(train_loader)):
#         outputs = model(**batch)
#         if cfg.loss_config.if_gather:
#             vis_feat = hvd.allgather(outputs['vis_features'])
#             text_feat = hvd.allgather(outputs['text_features'])
#             if cfg.loss_config.loss_name in ["NCELearnableTempLoss", "NCELearnableTempDSLLoss"]:
#                 if hasattr(model, 'module'):
#                     logit_scale = model.module.clipmodel.logit_scale
#                 else:
#                     logit_scale = model.clipmodel.logit_scale
#                 loss = loss_func(vis_feat, text_feat, logit_scale)
#             else:
#                 loss = loss_func(vis_feat, text_feat)
#         else:
#             loss = outputs['loss']
#
#         if hasattr(model, 'module'):
#             torch.clamp_(model.module.clipmodel.logit_scale.data, 0, np.log(200))
#             logit_scale_ = model.module.clipmodel.logit_scale.data
#         else:
#             torch.clamp_(model.clipmodel.logit_scale.data, 0, np.log(200))
#             logit_scale_ = model.clipmodel.logit_scale.data
#
#         if step % 10 == 0:
#             lr_ = optimizer.param_groups[0]['lr']
#             LOGGER.info('Step {}: loss {} lr {} logit_scale {}'.format(global_step, loss, lr_, logit_scale_))
#
#         running_loss(loss.item())
#
#         delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0
#         with amp.scale_loss(
#                 loss, optimizer, delay_unscale=delay_unscale
#                 ) as scaled_loss:
#             scaled_loss.backward()
#             # zero_none_grad(model)
#             optimizer.synchronize()
#
#         # optimizer
#         if (step + 1) % cfg.gradient_accumulation_steps == 0:
#             global_step += 1
#             TB_LOGGER.log_scalar_dict({'vtc_loss': running_loss.val})
#             n_epoch = int(1.* cfg.gradient_accumulation_steps *
#                           global_step / n_steps_in_epoch)
#             # learning rate scheduling transformer
#             lr_this_step = get_lr_sched(
#                 global_step, cfg.decay, cfg.learning_rate,
#                 cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
#                 decay_epochs=cfg.step_decay_epochs, multi_step_epoch=n_epoch)
#
#             for pg_n, param_group in enumerate(
#                     optimizer.param_groups):
#                 if pg_n in [0, 1]:
#                     param_group['lr'] = (
#                         cfg.lr_mul * lr_this_step)
#                 elif pg_n in [2, 3]:
#                     param_group['lr'] = lr_this_step
#
#             TB_LOGGER.add_scalar(
#                 "train/lr", lr_this_step,
#                 global_step)
#
#             # update model params
#             if cfg.grad_norm != -1:
#                 grad_norm = clip_grad_norm_(
#                     amp.master_params(optimizer), cfg.grad_norm)
#                 TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)
#             TB_LOGGER.step()
#
#             # Check if there is None grad
#             none_grads = [
#                 p[0] for p in model.named_parameters()
#                 if p[1].requires_grad and p[1].grad is None]
#
#             assert len(none_grads) == 0, f"{none_grads}"
#
#             with optimizer.skip_synchronize():
#                 optimizer.step()
#                 optimizer.zero_grad()
#             restorer.step()
#
#             # checkpoint
#             if global_step % cfg.valid_steps == 0:
#                 LOGGER.info(f'Step {global_step}: start validation and Save')
#                 _, t2vr1 = validate(model, inference_loaders, cfg)
#                 model_saver.save(step=global_step, model=model)
#                 if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
#                     best_model_saver.save(step=global_step, model=model)
#                     best_model_saver.bestr1 = t2vr1
#             else:
#                 if global_step % cfg.only_valid_steps == 0:
#                     LOGGER.info(f'Step {global_step}: start inference')
#                     _, t2vr1 = validate(model, inference_loaders, cfg)
#                     if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
#                         best_model_saver.save(step=global_step, model=model)
#                         best_model_saver.bestr1 = t2vr1
#                         LOGGER.info('*' + '-' * 20 + '*')
#                         LOGGER.info(f'Best R1 is {best_model_saver.bestr1} at step {global_step}')
#
#         if global_step >= cfg.num_train_steps:
#             break
#
#     if global_step % cfg.valid_steps != 0:
#         LOGGER.info(f'Step {global_step}: start validation')
#         _, t2vr1 = validate(model, inference_loaders, cfg)
#
#         model_saver.save(step=global_step, model=model)
#         if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
#             best_model_saver.save(step=global_step, model=model)
#             best_model_saver.bestr1 = t2vr1
#             LOGGER.info(f'Best R1 is {best_model_saver.bestr1} at step {global_step}')


def start_training():
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

    if hvd.rank() != 0:
        LOGGER.disabled = True

    model = setup_model(cfg, device=device)
    model.train()

    optimizer = setup_e2e_optimizer(model, cfg)

    # Horovod: (optional) compression algorithm.compressin
    compression = hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression)

    #  Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    model, optimizer = amp.initialize(
        model, optimizer, enabled=cfg.fp16, opt_level=cfg.amp_level,
        keep_batchnorm_fp32=True if cfg.amp_level == 'O2' else None)

    # prepare data
    tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip_config)
    train_loader, val_loaders, inference_loaders = setup_dataloaders(cfg, tokenizer)

    img_norm = None
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loaders = {k: PrefetchLoader(v, img_norm)
                   for k, v in val_loaders.items()}
    inference_loaders = {k: PrefetchLoader(v, img_norm)
                         for k, v in inference_loaders.items()}

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
    validate(model, inference_loaders, cfg)

    loss_func = build_loss_func(cfg.loss_config)

    for step, batch in enumerate(InfiniteIterator(train_loader)):
        outputs = model(**batch)
        if cfg.loss_config.if_gather:
            vis_embeds = hvd.allgather(outputs['vis_features'])
            text_embeds = hvd.allgather(outputs['text_features'])

            if cfg.loss_config.loss_name in ["NCELearnableTempLoss", "NCELearnableTempDSLLoss"]:
                if hasattr(model, 'module'):
                    logit_scale = model.module.clipmodel.logit_scale
                else:
                    logit_scale = model.clipmodel.logit_scale
                loss = loss_func(vis_embeds, text_embeds, logit_scale)
            else:
                loss = loss_func(vis_embeds, text_embeds)
        else:
            loss = outputs['loss']

        if hasattr(model, 'module'):
            torch.clamp_(model.module.clipmodel.logit_scale.data, 0, np.log(200))
            logit_scale_ = model.module.clipmodel.logit_scale.data
        else:
            torch.clamp_(model.clipmodel.logit_scale.data, 0, np.log(200))
            logit_scale_ = model.clipmodel.logit_scale.data

        if step % 10 == 0:
            lr_ = optimizer.param_groups[0]['lr']
            LOGGER.info('Step {}: loss {} lr {} logit_scale {}'.format(global_step, loss, lr_, logit_scale_))

        running_loss(loss.item())

        delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0
        with amp.scale_loss(
                loss, optimizer, delay_unscale=delay_unscale
        ) as scaled_loss:
            scaled_loss.backward()
            # zero_none_grad(model)
            optimizer.synchronize()

        # optimizer
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            global_step += 1
            TB_LOGGER.log_scalar_dict({'vtc_loss': running_loss.val})
            n_epoch = int(1. * cfg.gradient_accumulation_steps *
                          global_step / n_steps_in_epoch)
            # learning rate scheduling transformer
            lr_this_step = get_lr_sched(
                global_step, cfg.decay, cfg.learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.step_decay_epochs, multi_step_epoch=n_epoch)

            for pg_n, param_group in enumerate(
                    optimizer.param_groups):
                if pg_n in [0, 1]:
                    param_group['lr'] = (
                            cfg.lr_mul * lr_this_step)
                elif pg_n in [2, 3]:
                    param_group['lr'] = lr_this_step

            TB_LOGGER.add_scalar(
                "train/lr", lr_this_step,
                global_step)

            # update model params
            if cfg.grad_norm != -1:
                grad_norm = clip_grad_norm_(
                    amp.master_params(optimizer), cfg.grad_norm)
                TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()

            # Check if there is None grad
            none_grads = [
                p[0] for p in model.named_parameters()
                if p[1].requires_grad and p[1].grad is None]

            assert len(none_grads) == 0, f"{none_grads}"

            with optimizer.skip_synchronize():
                optimizer.step()
                optimizer.zero_grad()
            restorer.step()

            # checkpoint
            if global_step % cfg.valid_steps == 0:
                LOGGER.info(f'Step {global_step}: start validation and Save')
                _, t2vr1 = validate(model, inference_loaders, cfg)
                model_saver.save(step=global_step, model=model)
                if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
                    best_model_saver.save(step=global_step, model=model)
                    best_model_saver.bestr1 = t2vr1
            else:
                if global_step % cfg.only_valid_steps == 0:
                    LOGGER.info(f'Step {global_step}: start inference')
                    _, t2vr1 = validate(model, inference_loaders, cfg)
                    if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
                        best_model_saver.save(step=global_step, model=model)
                        best_model_saver.bestr1 = t2vr1
                        LOGGER.info('*' * 25)
                        LOGGER.info(f'Best R1 is {best_model_saver.bestr1} at step {global_step}')

        if global_step >= cfg.num_train_steps:
            break

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        _, t2vr1 = validate(model, inference_loaders, cfg)

        model_saver.save(step=global_step, model=model)
        if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
            best_model_saver.save(step=global_step, model=model)
            best_model_saver.bestr1 = t2vr1
            LOGGER.info(f'Best R1 is {best_model_saver.bestr1} at step {global_step}')


def start_fk_training():
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

    if hvd.rank() != 0:
        LOGGER.disabled = True

    model = setup_fk_model(cfg, device=device)
    model.train()

    optimizer = setup_e2e_optimizer(model, cfg)

    # Horovod: (optional) compression algorithm.compressin
    compression = hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression)

    #  Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    model, optimizer = amp.initialize(
        model, optimizer, enabled=cfg.fp16, opt_level=cfg.amp_level,
        keep_batchnorm_fp32=True if cfg.amp_level == 'O2' else None)

    # prepare data
    tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip_config)
    train_loader, val_loaders, inference_loaders = setup_dataloaders(cfg, tokenizer)

    img_norm = None
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loaders = {k: PrefetchLoader(v, img_norm)
                   for k, v in val_loaders.items()}
    inference_loaders = {k: PrefetchLoader(v, img_norm)
                         for k, v in inference_loaders.items()}

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
    fk_validate(model, inference_loaders, cfg)

    loss_func = build_loss_func(cfg.loss_config)
    generation_loss_func = build_generation_loss_func(cfg.generation_loss_config)

    for step, batch in enumerate(InfiniteIterator(train_loader)):
        outputs = model(**batch)
        if cfg.loss_config.if_gather:
            text_feats = hvd.allgather(outputs['text_feats'])
            text_all_feats = hvd.allgather(outputs['text_all_feats'])
            final_vis_feats = hvd.allgather(outputs['final_video_feats'])
            generated_text_feats = hvd.allgather(outputs['generated_text_feats'])
            generated_global_text_feats = hvd.allgather(outputs['generated_global_text_feats'])

            if cfg.loss_config.loss_name in ["NCELearnableTempLoss", "NCELearnableTempDSLLoss"]:
                if hasattr(model, 'module'):
                    logit_scale = model.module.encoder.clipmodel.logit_scale
                else:
                    logit_scale = model.encoder.clipmodel.logit_scale

                text_embeds = text_feats / text_feats.norm(dim=-1, keepdim=True)
                final_vis_embeds = final_vis_feats / final_vis_feats.norm(dim=-1, keepdim=True)

                retrieval_loss = loss_func(final_vis_embeds, text_embeds, logit_scale)
                # in this case: it is patch2word
                # if cfg.generator_config.num_query_token > 1:
                #     selected_text_all_feats = text_all_feats[:, 1:cfg.generator_config.num_query_token, :]
                #     selected_text_all_feats = torch.cat([selected_text_all_feats, text_feats.unsqueeze(1)], dim=1)
                #     generation_loss = cfg.generation_loss_config.generation_loss_lambda * generation_loss_func(generated_text_feats, selected_text_all_feats, temp=None)
                # else:
                # # in this case: it is patch2eot
                #     generation_loss = cfg.generation_loss_config.generation_loss_lambda * generation_loss_func(generated_global_text_feats, text_feats,
                #                                                                     temp=None)
                # loss = retrieval_loss + generation_loss
                loss = retrieval_loss
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
            LOGGER.info(
                'Step {}: loss {} | retrieval_loss {} | lr {} | logit_scale {}'.format(
                    global_step, loss, retrieval_loss, lr_,
                    logit_scale_))

        running_loss(loss.item())

        delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0
        with amp.scale_loss(
                loss, optimizer, delay_unscale=delay_unscale
        ) as scaled_loss:
            scaled_loss.backward()
            # zero_none_grad(model)
            optimizer.synchronize()

        # optimizer
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            global_step += 1
            TB_LOGGER.log_scalar_dict({'vtc_loss': running_loss.val})
            n_epoch = int(1. * cfg.gradient_accumulation_steps *
                          global_step / n_steps_in_epoch)
            # learning rate scheduling transformer
            lr_this_step = get_lr_sched(
                global_step, cfg.decay, cfg.learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.step_decay_epochs, multi_step_epoch=n_epoch)

            for pg_n, param_group in enumerate(
                    optimizer.param_groups):
                if pg_n in [0, 1]:
                    param_group['lr'] = (
                            cfg.lr_mul * lr_this_step)
                elif pg_n in [2, 3]:
                    param_group['lr'] = lr_this_step

            TB_LOGGER.add_scalar(
                "train/lr", lr_this_step,
                global_step)

            # update model params
            if cfg.grad_norm != -1:
                grad_norm = clip_grad_norm_(
                    amp.master_params(optimizer), cfg.grad_norm)
                TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()

            # Check if there is None grad
            none_grads = [
                p[0] for p in model.named_parameters()
                if p[1].requires_grad and p[1].grad is None]

            # TODO: solve not used params in qformer!
            # assert len(none_grads) == 0, f"{none_grads}"

            with optimizer.skip_synchronize():
                optimizer.step()
                optimizer.zero_grad()
            restorer.step()

            # checkpoint
            if global_step % cfg.valid_steps == 0:
                LOGGER.info(f'Step {global_step}: start validation and Save')
                _, t2vr1 = fk_validate(model, inference_loaders, cfg)
                model_saver.save(step=global_step, model=model)
                if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
                    best_model_saver.save(step=global_step, model=model)
                    best_model_saver.bestr1 = t2vr1
            else:
                if global_step % cfg.only_valid_steps == 0:
                    LOGGER.info(f'Step {global_step}: start inference')
                    _, t2vr1 = fk_validate(model, inference_loaders, cfg)
                    if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
                        best_model_saver.save(step=global_step, model=model)
                        best_model_saver.bestr1 = t2vr1
                        LOGGER.info('*' + '-' * 20 + '*')
                        LOGGER.info(f'Best R1 is {best_model_saver.bestr1} at step {global_step}')

        if global_step >= cfg.num_train_steps:
            break

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        _, t2vr1 = fk_validate(model, inference_loaders, cfg)

        model_saver.save(step=global_step, model=model)
        if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
            best_model_saver.save(step=global_step, model=model)
            best_model_saver.bestr1 = t2vr1
            LOGGER.info(f'Best R1 is {best_model_saver.bestr1} at step {global_step}')


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
    mse_matrix_mean_reduction = mse_matrix_mean_reduction[mask].view(1000, 999)

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


if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    start_fk_training()

# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python src/tasks/run_video_retrieval.py --config src/configs/msrvtt_retrieval_debug.json  --blob_mount_dir /blob_mount/
