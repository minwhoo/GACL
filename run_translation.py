import argparse
import json
import logging
import math
import os
import random
import shutil
import copy
import contextlib
from pathlib import Path
from itertools import chain

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler
from datasets import load_dataset, concatenate_datasets
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MBartTokenizer,
    MBartTokenizerFast,
    M2M100Tokenizer,
    get_scheduler,
)
from tokenization_small100 import SMALL100Tokenizer
from tokenization_nllb200 import NllbTokenizer, NllbTokenizerFast

from gender_eval import MTGenEvalEvaluator
from grad_cache import RandContext


logger = get_logger(__name__)


FLORES200_LANG_CODE_MAP = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "fr": "fra_Latn",
    "uk": "ukr_Cyrl",
    "he": "heb_Hebr",
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "cs": "ces_Latn",
    "hi": "hin_Deva",
    "pt": "por_Latn",
    "pl": "pol_Latn",
}

SUPPORTED_TARGET_LANGS = { "ru", "es", "fr", "it", "ar", "de", "hi", "pt", "uk", "he", "cs", "pl" }


def set_tokenizer_src_tgt_langs(tokenizer, src_lang, tgt_lang):
    # For translation we set the codes of our source and target languages
    if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast, M2M100Tokenizer, SMALL100Tokenizer)):
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
    elif isinstance(tokenizer, (NllbTokenizer, NllbTokenizerFast)):
        tokenizer.src_lang = FLORES200_LANG_CODE_MAP[src_lang]
        tokenizer.tgt_lang = FLORES200_LANG_CODE_MAP[tgt_lang]
    else:
        raise KeyError("Unknown tokenizer type", type(tokenizer))


def get_inverse_sqrt_schedule_with_warmup(
    optimizer, num_warmup_steps, lr_init=1e-7, last_epoch=-1
):
    """
    Code taken from: https://fairseq.readthedocs.io/en/latest/_modules/fairseq/optim/lr_scheduler/inverse_square_root_schedule.html
    Create a schedule with a learning rate that decreases proportionally to inverse sqrt of num steps

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        lr_init (`float`, *optional*, defaults to 1e-7):
            Initial learning rate during warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr = optimizer.defaults["lr"]
    if not (lr_init < lr):
        raise ValueError(f"lr_init ({lr_init}) must be be smaller than lr ({lr})")
    num_warmup_steps = max(1, num_warmup_steps)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            lr_step = (lr - lr_init) / float(num_warmup_steps)
            return (lr_init + current_step * lr_step) / lr  # as LambdaLR multiplies by lr
        else:
            decay_factor = float(num_warmup_steps) ** 0.5  # as LambdaLR multiplies by lr
            return decay_factor * current_step**-0.5

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_gender_based_inbatch_embs(orig_emb, gc_emb, gender):
    # get gender indices
    genders_mat = gender.t().repeat(len(gender),1)  # [B, B]
    indices_mat = torch.arange(len(gender)).unsqueeze(0).repeat(len(gender),1)  # [B, B]

    other_indices_mat = remove_diag(indices_mat)
    other_genders_mat = remove_diag(genders_mat)

    opposite_gender_indices = torch.logical_xor(other_genders_mat, gender.unsqueeze(1))  # [B, B-1]
    # print(opposite_gender_indices)
    # 1 if different, 0 if same

    orig_same_inbatch_embs = torch.stack((orig_emb, gc_emb))[opposite_gender_indices.to(torch.long), other_indices_mat, :]  # [B, B-1, D]
    gc_same_inbatch_embs = torch.stack((orig_emb, gc_emb))[(~opposite_gender_indices).to(torch.long), other_indices_mat, :]  # [B, B-1, D]

    return orig_same_inbatch_embs, gc_same_inbatch_embs


def supcon_loss(a, p, n, temperature=1.0, return_additional_data=False, summation_inside_log=False):
    positive_logits = torch.einsum('ij,ikj->ik', a, p)  # [B, M]
    negative_logits = torch.einsum('ij,ikj->ik', a, n)  # [B, N]
    all_logits = torch.cat([positive_logits, negative_logits], dim=-1)  # [B, M+N]
    logprobs_all = F.log_softmax(all_logits / temperature, dim=-1)  # [B, M+N]
    if summation_inside_log:
        if p.shape[1] == 1: # single positive pair
            loss = - (logprobs_all[:,0]).mean()
        else:
            # log(pos/all) = log(pos/pos1*pos1/all) = log(pos1/all) - log(pos1/pos)
            logprobs_pos = F.log_softmax(positive_logits / temperature, dim=-1)  # [B, M]
            loss = - (logprobs_all[:,0] - logprobs_pos[:, 0]).mean()
    else:
        loss = - logprobs_all[:,:positive_logits.shape[1]].mean()
    if return_additional_data:
        additional_data = {
            'pos_logit_mean': positive_logits.mean().detach().float(),
            'pos_logit_std': positive_logits.std().detach().float(),
            'neg_logit_mean': negative_logits.mean().detach().float(),
            'neg_logit_std': negative_logits.std().detach().float(),
        }
        return loss, additional_data
    return loss


def knowledge_distillation_loss(student_logits, teacher_logits, attention_mask):
    """Knowledge distillation loss while ignoring padded parts
    student_logits: [B, T, V]
    teacher_logits: [B, T, V]
    attention_mask: binary tensor of [B, T] where 0 if padded else 1
    """
    # sum_v = p (log(p) - log(q))
    elem = F.softmax(teacher_logits, dim=-1) * (F.log_softmax(teacher_logits, dim=-1) - F.log_softmax(student_logits, dim=-1))  # [B, T, V]
    token_wise_loss = (elem * attention_mask.unsqueeze(-1)).sum(dim=-1)  # [B, T]
    count =  torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)  # [B, T]
    batch_wise_loss = token_wise_loss.sum(dim=-1) / count  # [B]
    return batch_wise_loss.mean()


def remove_diag(a):
    """[N, N] -> [N, N-1] with diagonals removed"""
    n = len(a)
    return a[torch.eye(n) != 1].view(n,n-1)


def mean_pool(emb, attention_mask):
    """Mean pooling while ignoring padded parts
    emb: [B, T, D]
    attention_mask: binary tensor of [B, T] where 0 if padded else 1
    """
    emb_sum = (emb * attention_mask.unsqueeze(-1)).sum(dim=1)
    count =  torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)
    return emb_sum / count


class EarlyStopSaveBestModule:
    def __init__(self, best_eval="validation", second_eval=None, second_eval_threshold=None):
        self.best_eval = best_eval
        self.best_metric = "bleu"
        self.second_eval = second_eval
        self.second_eval_threshold = second_eval_threshold
        if "mtgeneval" in self.best_eval:
            self.best_metric = "accuracy"
        elif "winomt" in self.best_eval:
            self.best_metric = "acc"
        elif "validation" in self.best_eval:
            self.best_metric = "bleu"
        else:
            raise KeyError("Unknown metric to use for eval of", self.best_eval)

        self.best_ckpts = []
        self.num_best = 1
        self.earlystop_patience = 10
        self.second_eval_ckpt_path = None

        self.earlystop_cnt = 0

    def save_if_best(self, accelerator, completed_steps, output_dir, eval_results, eval_name):
        # save separate ckpt for the last ckpt where the metric is still over a threshold
        if self.second_eval is not None and eval_name == self.second_eval:
            metric_val = eval_results["bleu"]
            if metric_val >= self.second_eval_threshold:
                if self.second_eval_ckpt_path is not None:
                    shutil.rmtree(self.second_eval_ckpt_path)
                self.second_eval_ckpt_path = output_dir + "_threshold"
                accelerator.save_state(self.second_eval_ckpt_path)

        elif eval_name == self.best_eval:
            metric_val = eval_results[self.best_metric]
            last_best_step = None if len(self.best_ckpts) == 0 else self.best_ckpts[0]['steps']
            if len(self.best_ckpts) < self.num_best or metric_val > self.best_ckpts[-1]['val']:
                # save
                accelerator.save_state(output_dir)
                new_best_ckpt = {
                    'steps': completed_steps,
                    'val': metric_val,
                    'path': output_dir
                }
                self.best_ckpts = sorted(self.best_ckpts + [new_best_ckpt], key=lambda x: x['val'], reverse=True)
                while len(self.best_ckpts) > self.num_best:
                    del_ckpt = self.best_ckpts.pop()
                    if accelerator.is_main_process:
                        accelerator.print(f"Deleting old ckpt (step={del_ckpt['steps']})")
                        shutil.rmtree(del_ckpt['path'])

            # reset earlystop_cnt if top-1 best model changed
            if last_best_step != self.best_ckpts[0]['steps']:
                self.earlystop_cnt = 0
            else:
                self.earlystop_cnt += 1

    def is_earlystop(self):
        return self.earlystop_cnt >= self.earlystop_patience

    def get_best_ckpt_info(self):
        return self.best_ckpts[0]


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    # main params (dataset/model/output)
    parser.add_argument("--dataset_name", type=str, default=None, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).",)
    parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.")
    parser.add_argument("--target_langs", action='store', nargs="*")
    parser.add_argument("--eval_langs", action='store', nargs="*")
    parser.add_argument("--exclude_mt_loss", action='store_true', help="Don't apply mt loss")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=False,)
    parser.add_argument("--debug", action='store_true')

    # input params
    parser.add_argument("--max_source_length", type=int, default=1024, help=( "The maximum total input sequence length after " "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."),)
    parser.add_argument("--max_target_length", type=int, default=128, help=( "The maximum total sequence length for target text after " "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded." "during ``evaluate`` and ``predict``."),)

    # generation params
    parser.add_argument("--num_beams", type=int, default=None, help=( "Number of beams to use for evaluation. This argument will be " "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."),)
    parser.add_argument("--val_max_target_length", type=int, default=None, help=( "The maximum total sequence length for validation " "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be " "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` " "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."),)

    # preprocess params
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.",)
    parser.add_argument("--overwrite_cache", action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.",)

    # train params
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.",)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt"],)
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")

    # eval params
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.",)
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluate every n steps during training",)
    parser.add_argument("--eval_only", action='store_true')

    # gc params
    parser.add_argument("--gc_training", action='store_true')
    parser.add_argument("--gc_lambda", type=float, default=0.1, help="Contrastive loss hyperparameter")
    parser.add_argument("--kd_lambda", type=float, default=0.0, help="Knowledge distillation loss hyperparameter")
    parser.add_argument("--gc_loss_type", default="infonce", choices=["margin", "infonce", "soft_nn", "supcon", "supcon_in"])
    parser.add_argument("--gc_loss_dir", default="original", choices=["original", "reversed"])
    parser.add_argument("--gc_anchor_type", default="original", choices=["original", "augmented", "both"])
    parser.add_argument("--gc_positive_type", default="inbatch", choices=["dropout", "inbatch", "both"])
    parser.add_argument("--gc_negative_type", default="direct", choices=["direct", "full"])
    parser.add_argument("--gc_loss_temp", type=float, default=1.0, help="infonce loss temperature")

    args = parser.parse_args()

    do_train = not args.eval_only

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = Accelerator(logging_dir=args.output_dir, gradient_accumulation_steps=args.gradient_accumulation_steps) if do_train else Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        # transformers.utils.logging.set_verbosity_info()  # <- prints GenerateConfig on every model.generate call
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output dir creation
    if do_train:
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        if args.debug:
            args.eval_steps = 100

    if len(args.target_langs) > 0:
        raw_datasets = datasets.DatasetDict()
        raw_datasets[f"train_de"] = load_dataset("json", data_files="data/wmt18_cleaned_en_de_train_balanced.jsonl", split='train[:1%]' if args.debug else 'train')

        if args.eval_langs is None:
            args.eval_langs = args.target_langs
        source_lang = "en"
        logger.info(f"Dataset source lang: {source_lang}")
        logger.info(f"Dataset target langs: {args.target_langs}")
    else:
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found. You can easily tweak this behavior (see below).
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, split=['train', 'validation'])
        else:
            raw_datasets = datasets.DatasetDict()
            data_files = {}
            if args.train_file is not None:
                extension = args.train_file.split(".")[-1]
                if extension == "jsonl":
                    extension = "json"
                if args.debug:
                    raw_datasets["train"] = load_dataset(extension, data_files=args.train_file, split='train[:1%]')
                else:
                    raw_datasets["train"] = load_dataset(extension, data_files=args.train_file, split='train')
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
                extension = args.validation_file.split(".")[-1]
                if extension == "jsonl":
                    extension = "json"
                if args.debug:
                    raw_datasets["validation"] = load_dataset(extension, data_files=args.validation_file, split='train[:5%]')
                else:
                    raw_datasets["validation"] = load_dataset(extension, data_files=args.validation_file, split='train')

        # Get the language codes for input/target.
        source_lang = "en"
        target_lang = [k for k in raw_datasets["validation"][0]["translation"].keys() if k in SUPPORTED_TARGET_LANGS][0]
        logger.info(f"Dataset source lang: {source_lang}")
        logger.info(f"Dataset target lang: {target_lang}")

        if args.eval_langs is None:
            args.eval_langs = [target_lang]

    # Load pretrained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    if "small100" in model.config._name_or_path:
        logger.info("Using SMALL100 Tokenizer")
        tokenizer = SMALL100Tokenizer.from_pretrained(args.model_name_or_path)
    elif "nllb-200" in model.config._name_or_path:
        logger.info("Using Fixed NLLB200 Tokenizer")
        tokenizer = NllbTokenizerFast.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model.resize_token_embeddings(len(tokenizer))

    # Add linear projection module
    linear_projection_module = nn.Sequential(
        nn.Dropout(model.config.dropout),
        nn.Linear(model.model.shared.embedding_dim, model.model.shared.embedding_dim),
    )
    model.add_module('linear_projection', linear_projection_module)

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, (MBartTokenizer)):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    def preprocess_function(examples):
        columns = set(examples["translation"][0].keys())
        target_langs = columns.intersection(SUPPORTED_TARGET_LANGS)
        if len(target_langs) == 1:
            target_lang = list(target_langs)[0]
        elif len(target_langs) == 0:
            if "target_lang" in columns:
                target_lang = examples["translation"][0]["target_lang"]
            else:
                raise KeyError("Couldn't detect target language for dataset dataset")
        else:
            raise KeyError("Detected multiple target languages in single dataset", target_langs)

        # For translation we set the codes of our source and target languages
        set_tokenizer_src_tgt_langs(tokenizer, source_lang, target_lang)

        batched = { key: [ex[key] for ex in examples['translation']] for key in columns }

        processed = {}
        for key, batch in batched.items():
            if key == "gender":
                processed[key] = [int(b == "male") for b in batch]
            elif key.startswith(source_lang):
                key = key.replace(source_lang, 'src', 1)
                tokenized = tokenizer(batch, max_length=args.max_source_length, truncation=True)
                for t_key, val in tokenized.items():
                    processed[f"{t_key}_{key}"] = val
            elif key.startswith(target_lang):
                key = key.replace(target_lang, 'tgt', 1)
                tokenized = tokenizer(batch, max_length=args.max_target_length, truncation=True)
                for t_key, val in tokenized.items():
                    processed[f"{t_key}_{key}"] = val
            elif key == "target_lang":
                processed["target_lang"] = batched["target_lang"]
            else:
                raise KeyError("Unknkown key", key)
        if "target_lang" not in processed:
            processed['target_lang'] = [target_lang for _ in batch]
        return processed

    evaluators = {}
    # evaluators["validation"] = evaluate.load("sacrebleu")
    for lang in args.eval_langs:
        evaluators[f"mtgeneval_{lang}"] = MTGenEvalEvaluator("dev", lang)

        raw_datasets[f"mtgeneval_{lang}"] = evaluators[f"mtgeneval_{lang}"].get_dataset()

    # processed_datasets = raw_datasets
    with accelerator.main_process_first():
        Path("./dataset_cache").mkdir(exist_ok=True)
        cache_file_names = {k: "./dataset_cache/"+ str(k) + ".arrow" for k in raw_datasets}
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=['translation'],
            load_from_cache_file=not args.overwrite_cache,
            cache_file_names=cache_file_names,
            desc="Running tokenizer on dataset",
        )

    if do_train:
        if len(args.target_langs) > 0:
            processed_datasets["train"] = concatenate_datasets([d for key, d in processed_datasets.items() if key.startswith("train")])
        remove_keys = [key for key in processed_datasets if key.startswith("train_")]
        for key in remove_keys:
            processed_datasets.pop(key)

        # Log a few random samples from the training set:
        for index in random.sample(range(len(processed_datasets["train"])), 3):
            logger.info(f"Sample {index} of the training set: {processed_datasets['train'][index]}.")

    def data_collator(batch):
        padding_kwargs = {
            'padding': True,
            'max_length': None,
            'pad_to_multiple_of': 8 if accelerator.use_fp16 else None,
            'return_tensors': "pt",
        }

        inputs = tokenizer.pad({
            'input_ids': [b['input_ids_src'] for b in batch],
            'attention_mask': [b['attention_mask_src'] for b in batch],
        }, **padding_kwargs)
        if 'input_ids_tgt' in batch[0]:
            labels = tokenizer.pad({
                'input_ids': [b['input_ids_tgt'] for b in batch],
                'attention_mask': [b['attention_mask_tgt'] for b in batch],
            }, **padding_kwargs)
            inputs['labels'] = labels['input_ids']
            inputs['labels'][inputs['labels'] == tokenizer.pad_token_id] = -100
        if args.gc_training and 'gender' in batch[0]:
            inputs['genders'] = torch.tensor([b['gender'] for b in batch])

        return inputs

    logger.info(f"Processed datasets: {list(processed_datasets.keys())}")

    if do_train:
        train(processed_datasets, evaluators, data_collator, args, model, accelerator, tokenizer)
    else:
        for dataset_name, dataset in processed_datasets.items():
            if dataset_name == "train":
                continue
            accelerator.print(f"Evaluate on {dataset_name}")
            run_evaluate(dataset, data_collator, args, model, accelerator, tokenizer, evaluators[dataset_name])


def run_evaluate(eval_dataset, data_collator, args, model, accelerator, tokenizer, metric):
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    eval_results = evaluate_loop(args, model, eval_dataloader, accelerator, tokenizer, metric)

    accelerator.wait_for_everyone()
    accelerator.print("Evaluation finished!")
    for k, v in eval_results.items():
        accelerator.print(f" * {k:10} : {v:5.4f}")


class GenderBalancedRandomSampler(Sampler):
    def __init__(self, genders) -> None:
        self.genders = torch.as_tensor(genders)

    def __iter__(self):
        shuffled_indices = torch.randperm(len(self.genders))

        shuffled_genders = self.genders[shuffled_indices]

        male_indices = shuffled_indices[shuffled_genders == 1].tolist()
        female_indices = shuffled_indices[shuffled_genders != 1].tolist()
        assert len(male_indices) == len(female_indices)

        yield from chain.from_iterable(zip(male_indices, female_indices))

    def __len__(self):
        return len(self.genders)


def train(datasets, evaluators, data_collator, args, model, accelerator: Accelerator, tokenizer):
    sampler = GenderBalancedRandomSampler([d['gender'] for d in datasets['train']])
    # target_langs = [d['target_lang'] for d in datasets["train"]]
    # target_lang_cntr = Counter(target_langs)
    # target_lang_weights = [1. / target_lang_cntr[g] for g in target_langs]
    # sampler = WeightedRandomSampler(target_lang_weights, len(target_lang_weights))
    train_dataloader = DataLoader(
        datasets["train"], sampler=sampler, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    eval_names, eval_dataloaders = [], []
    for dataset_name, dataset in datasets.items():
        if dataset_name == "train":
            continue
        eval_names.append(dataset_name)
        eval_dataloaders.append(
            DataLoader(dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    new_params = []
    orig_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if 'linear_projection' in name:
            new_params.append(param)
        elif any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            orig_params.append(param)

    optimizer_grouped_parameters = [
        { "params": new_params, "weight_decay": 1.0, },
        { "params": orig_params, "weight_decay": args.weight_decay, },
        { "params": no_decay_params, "weight_decay": 0, },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.lr_scheduler_type == "inverse_sqrt":
        lr_scheduler = get_inverse_sqrt_schedule_with_warmup(optimizer, args.num_warmup_steps, lr_init=0.1 * args.learning_rate)
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    # Prepare teacher model for knowledge distillation
    if args.kd_lambda > 0.0:
        logger.info("Creating teacher model")
        teacher_model = copy.deepcopy(model)
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()
        teacher_model = accelerator.prepare(teacher_model)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler, *eval_dataloaders = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, *eval_dataloaders
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if accelerator.is_main_process:
        experiment_config = vars(args)
        accelerator.init_trackers("gacl", experiment_config)
        # set summary value of wandb metric to be max (default=last)

    for eval_name in eval_names:
        if "mtgeneval" in eval_name:
            best_eval_name = eval_name
            break
    else:
        best_eval_name = eval_names[0]
    logger.info(f"Early stopping / saving based on eval of: {best_eval_name}")
    essb = EarlyStopSaveBestModule(best_eval=best_eval_name)

    def model_forward_fn(batch, skip_loss=True):
        # compute representations for contrastive learning
        if args.gc_training:
            original_batch = {k:v for k,v in batch.items() if k not in {'genders', }}
            outputs = model(**original_batch, output_hidden_states=True)

            emb = mean_pool(
                outputs.encoder_last_hidden_state,
                original_batch['attention_mask']
                )

            if torch.distributed.is_initialized():
                linear_module = model.module.linear_projection
            else:
                linear_module = model.linear_projection
            emb = linear_module(emb)

            orig_emb = emb[batch['genders'] == 0]
            gc_emb = emb[batch['genders'] != 0]
            assert orig_emb.shape == gc_emb.shape

            reps = {
                'orig_emb': orig_emb,  # [B, D]
                'gc_emb': gc_emb,  # [B, D]
                'gender': torch.zeros(len(batch['genders'])//2),  # [B]
            }
        else:
            reps = None

        if skip_loss:
            return reps
        else:
            # compute loss
            if not args.gc_training:
                outputs = model(**batch)
            mt_loss = outputs.loss
            loss = mt_loss
            log_losses = {
                "train/mt_loss": mt_loss.detach().float(),
            }

            if args.kd_lambda > 0.0:
                if not args.gc_training:
                    teacher_outputs = teacher_model(**batch)
                    kd_mask = batch['labels'] != -100
                else:
                    teacher_outputs = teacher_model(**original_batch)
                    kd_mask = original_batch['labels'] != -100
                kd_loss = knowledge_distillation_loss(outputs.logits, teacher_outputs.logits, kd_mask)
                if args.exclude_mt_loss:
                    loss = kd_loss
                else:
                    loss = (1 - args.kd_lambda) * loss + args.kd_lambda * kd_loss
                log_losses["train/kd_loss"] = kd_loss.detach().float()

            return reps, loss, log_losses

    def contrastive_loss_fn(orig_emb, gc_emb, gender, orig_emb_dropout=None, gc_emb_dropout=None):
        # normalize embs
        orig_emb = F.normalize(orig_emb, dim=-1) # [B, D]
        gc_emb = F.normalize(gc_emb, dim=-1) # [B, D]
        if orig_emb_dropout is not None:
            orig_emb_dropout = F.normalize(orig_emb_dropout, dim=-1) # [B, D]
        if gc_emb_dropout is not None:
            gc_emb_dropout = F.normalize(gc_emb_dropout, dim=-1) # [B, D]

        # get gender-based in-batch groups
        orig_same_inbatch_embs, gc_same_inbatch_embs = get_gender_based_inbatch_embs(orig_emb, gc_emb, gender)

        pos_neg_mapping_by_anchor_type = {
            "original": {
                "anchor": orig_emb,
                "dropout_pos": orig_emb_dropout,
                "inbatch_pos": orig_same_inbatch_embs,
                "direct_neg": gc_emb,
                "inbatch_neg": gc_same_inbatch_embs,
            },
            "augmented": {
                "anchor": gc_emb,
                "dropout_pos": gc_emb_dropout,
                "inbatch_pos": gc_same_inbatch_embs,
                "direct_neg": orig_emb,
                "inbatch_neg": orig_same_inbatch_embs,
            },
        }
        if args.gc_anchor_type == "both":
            anchor_types = ["original", "augmented"]
        else:
            anchor_types = [args.gc_anchor_type]

        additional_data = {}
        contrastive_loss = None
        for anchor_type in anchor_types:
            m = pos_neg_mapping_by_anchor_type[anchor_type]

            # get positive emb
            if args.gc_positive_type == "dropout":
                pos = m["dropout_pos"].unsqueeze(dim=1)
            elif args.gc_positive_type == "inbatch":
                pos = m["inbatch_pos"]
            elif args.gc_positive_type == "both":
                pos = torch.cat([m["dropout_pos"].unsqueeze(dim=1), m["inbatch_pos"]], dim=1)
            else:
                raise KeyError("unknown args.gc_negative_type", args.gc_negative_type)

            # get negative emb
            if args.gc_negative_type == "direct":
                neg = m["direct_neg"].unsqueeze(dim=1)
            elif args.gc_negative_type == "full":
                neg = torch.cat([m["direct_neg"].unsqueeze(dim=1), m["inbatch_neg"]], dim=1)
            else:
                raise KeyError("unknown args.gc_negative_type", args.gc_negative_type)

            if args.gc_loss_dir == "reversed":
                pos, neg = neg, pos

            # compute loss
            loss, d = supcon_loss(m["anchor"], pos, neg, temperature=args.gc_loss_temp, return_additional_data=True, summation_inside_log=args.gc_loss_type == "supcon_in")
            if args.gc_loss_dir == "reversed":
                loss *= -1.0
                for k, v in d.items():
                    if 'pos' in k:
                        additional_data[f"{anchor_type}_{k.replace('pos', 'neg')}"] = v
                    elif 'neg' in k:
                        additional_data[f"{anchor_type}_{k.replace('neg', 'pos')}"] = v
                    else:
                        additional_data[f"{anchor_type}_{k}"] = v
            else:
                additional_data.update({f"{anchor_type}_{k}": v for k,v in d.items()})

            if contrastive_loss is None:
                contrastive_loss = loss
            else:
                contrastive_loss += loss
        contrastive_loss /= len(anchor_types)

        return contrastive_loss, additional_data


    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(datasets['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="Training")
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]
        # need to multiply `gradient_accumulation_steps` to reflect real steps
        resume_step = int(training_difference.replace("step-", "")) * args.gradient_accumulation_steps
        starting_epoch = resume_step // len(train_dataloader)
        resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    do_grad_cache = args.gc_training and accelerator.gradient_accumulation_steps > 1 \
        and not (args.gc_positive_type == "dropout" and args.gc_negative_type == "direct" )

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()

        inputs_cache = []
        reps_cache = []
        rnd_states_cache = []
        dropout_reps_cache = []

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            inputs_cache.append(batch)

            # pre-compute gradients with forward-only
            if do_grad_cache:
                rnd_states_cache.append(RandContext(batch))
                with torch.no_grad():
                    reps = model_forward_fn(batch, skip_loss=True)
                    reps_cache.append(reps)

            if args.gc_positive_type in ["dropout", "both"]:
                with torch.no_grad():
                    reps = model_forward_fn(batch, skip_loss=True)
                    dropout_reps_cache.append({k + "_dropout": v for k,v in reps.items() if k.endswith('emb')})
            else:
                dropout_reps_cache.append({})


            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # build grad cache
                if do_grad_cache:
                    reps_batch = { k: torch.cat([r[k] for r in reps_cache], dim=0) for k in reps_cache[0].keys() }
                    # [B*G, ...]
                    reps_detached = { k: v.detach().requires_grad_() if k.endswith('emb') else v for k,v in reps_batch.items() }
                    # [B*G, ...]
                    reps_detached_gathered = { k: accelerator.gather(v) for k,v in reps_detached.items() }
                    # [B*G*P, ...]

                    dropout_reps_batch = { k: torch.cat([r[k] for r in dropout_reps_cache], dim=0) for k in dropout_reps_cache[0].keys() }
                    dropout_reps_gathered = { k: accelerator.gather(v) for k,v in dropout_reps_batch.items() }

                    full_contrastive_loss, additional_log = contrastive_loss_fn(**reps_detached_gathered, **dropout_reps_gathered)
                    full_contrastive_loss.backward()
                    rep_grads_chunked = { k: rep.grad.split([len(r[k]) for r in reps_cache], dim=0) for k, rep in reps_detached.items() if rep.requires_grad }
                    # {k: G*[B, ...]}
                    rep_grads_cache = [{k: v[i] for k, v in rep_grads_chunked.items()} for i in range(len(inputs_cache))]
                    # [G*{k:[B, ...]}]
                else:
                    rep_grads_cache = [None] * len(inputs_cache)
                    rnd_states_cache = [contextlib.nullcontext()] * len(inputs_cache)

                # actual forward-backward & optimization step
                log_losses = {}
                for batch_, rep_grad, rnd_state, dropout_reps in zip(inputs_cache, rep_grads_cache, rnd_states_cache, dropout_reps_cache):
                    with accelerator.accumulate(model):
                        with rnd_state:
                            reps, loss, log_losses_batch = model_forward_fn(batch_, skip_loss=False)
                            for key, val in log_losses_batch.items():
                                log_losses[key] = log_losses.get(key, 0.0) + val / args.gradient_accumulation_steps
                        if args.gc_training:
                            if do_grad_cache:
                                contrastive_loss = 0
                                for k, grad in rep_grad.items():
                                    contrastive_loss += torch.dot(reps[k].flatten(), grad.flatten())
                                # contrastive loss is already computed as the average of full batch, so to undo the effect of division done by accelerator.backward
                                if args.exclude_mt_loss and args.kd_lambda == 0.0:
                                    loss = args.gc_lambda * contrastive_loss * args.gradient_accumulation_steps
                                else:
                                    loss += args.gc_lambda * contrastive_loss * args.gradient_accumulation_steps
                            else:  # if no grad accumulation, just compute contrastive loss directly
                                reps_gathered = { k: accelerator.gather(v) for k,v in reps.items() }
                                dropout_reps_gathered = { k: accelerator.gather(v) for k,v in dropout_reps.items() }
                                # [B*P, ...]
                                full_contrastive_loss, additional_log = contrastive_loss_fn(**reps_gathered, **dropout_reps_gathered)
                                if args.exclude_mt_loss and args.kd_lambda == 0.0:
                                    loss = args.gc_lambda * full_contrastive_loss
                                else:
                                    loss += args.gc_lambda * full_contrastive_loss
                        else:
                            additional_log = {}
                        accelerator.backward(loss)

                if args.gc_training:
                    log_losses["train/contrastive_loss"] = full_contrastive_loss.detach().float()
                inputs_cache = []
                reps_cache = []
                rnd_states_cache = []
                dropout_reps_cache = []

                # accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if isinstance(args.eval_steps, int) and completed_steps % args.eval_steps == 0:
                    log_results = {}
                    for eval_name, eval_dataloader in zip(eval_names, eval_dataloaders):
                        if eval_name.startswith("validation"):
                            eval_results = evaluate_loop(args, model, eval_dataloader, accelerator, tokenizer, evaluators["validation"], eval_name)
                            eval_results["bleu"] = eval_results.pop("score")
                        else:
                            eval_results = evaluate_loop(args, model, eval_dataloader, accelerator, tokenizer, evaluators[eval_name], eval_name)
                        for key, val in eval_results.items():
                            log_results[f"{eval_name.replace('_','/')}/{key}"] = val

                        output_dir = f"step-{completed_steps}"
                        output_dir = os.path.join(args.output_dir, output_dir)
                        if eval_name == essb.best_eval or eval_name == essb.second_eval:
                            essb.save_if_best(accelerator, completed_steps, output_dir, eval_results, eval_name)

                    accelerator.log(
                        {
                            **log_results,
                            **log_losses,
                            **additional_log,
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

                    if essb.is_earlystop():
                        accelerator.print("Stopping training due to early stop")
                        break
                elif completed_steps % 100 == 0:
                    accelerator.log(
                        {
                            **log_losses,
                            **additional_log,
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

            if completed_steps >= args.max_train_steps:
                break

        if essb.is_earlystop():
            break

    accelerator.end_training()

    accelerator.wait_for_everyone()
    best_ckpt_info = essb.get_best_ckpt_info()
    accelerator.load_state(best_ckpt_info['path'])
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )

    accelerator.print("Training finished!")
    accelerator.print("Best checkpoint:")
    accelerator.print(f" * step  : {best_ckpt_info['steps']}")
    accelerator.print(f" * score : {best_ckpt_info['val']}")
    accelerator.print(f" * path  : {best_ckpt_info['path']}")

    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(best_ckpt_info, f)
    raise RuntimeError("Force program shutdown (bypass wandb bug)")  # sometimes script doesn't end even when train finished


def evaluate_loop(args, model, eval_dataloader, accelerator: Accelerator, tokenizer, metric, name=""):
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process, desc=f"Evaluating {name}")

    model.eval()

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else model.config.max_length,
        "num_beams": args.num_beams,
    }
    if isinstance(tokenizer, (M2M100Tokenizer)):
        lang = name.split("_")[-1]
        if lang in SUPPORTED_TARGET_LANGS:
            gen_kwargs["forced_bos_token_id"] = tokenizer.get_lang_id(lang)
        else:
            raise KeyError("Unable to set correct tokenizer target lang for eval", name)
    elif isinstance(tokenizer, (NllbTokenizer, NllbTokenizerFast)):
        lang = name.split("_")[-1]
        if lang in SUPPORTED_TARGET_LANGS:
            gen_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id[FLORES200_LANG_CODE_MAP[lang]]
        else:
            raise KeyError("Unable to set correct tokenizer target lang for eval", name)

    pred_lens = []
    for batch in eval_dataloader:
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            if "labels" in batch:
                labels = batch["labels"]
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = accelerator.gather_for_metrics(generated_tokens).cpu().numpy()
            if "labels" in batch:
                labels = accelerator.gather_for_metrics(labels).cpu().numpy()

                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            if "labels" in batch:
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [pred.strip() for pred in decoded_preds]
            if "labels" in batch:
                decoded_labels = [[label.strip()] for label in decoded_labels]
            else:
                decoded_labels = None

            for pred in decoded_preds:
                pred_lens.append(len(pred))
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            progress_bar.update(1)
    progress_bar.close()
    eval_metric = metric.compute()
    if "score" in eval_metric:
        results = {"score": eval_metric["score"], "gen_len": np.mean(pred_lens)}
    else:
        results = {**eval_metric, "gen_len": np.mean(pred_lens)}

    logger.info(results)
    return results


if __name__ == "__main__":
    main()
