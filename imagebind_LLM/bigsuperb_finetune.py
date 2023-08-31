import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from llama.llama_adapter import LLaMA_adapter

from data.dataset import FinetuneDataset, transform_train

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

from engine_finetune import train_one_epoch

# hank
from ImageBind.data import my_load_and_transform_audio_data
import copy
import llama
from llama import Tokenizer
from datasets import load_dataset, Audio, load_from_disk, concatenate_datasets


def get_args_parser():
    parser = argparse.ArgumentParser('imagebind-llm pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='7B_chinese', type=str,
                        help='Type of LLaMA model') #
    parser.add_argument('--llama_path', default='/path/to/llama', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='path to checkpoint from pretrain stage')
    parser.add_argument('--max_words', default=512, type=int,
                        help='max number of input words')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_config', default='configs/data/finetune/EN.yaml', type=str,
                        help='dataset config path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)


    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


tokenizer = Tokenizer(model_path="/home/u8915687/lab/big-superb/Macaw-LLM2/weights/llama_7B/tokenizer.model")

# def collate_fn(b):
#     batch = {}
    
#     audios = []
#     prompts = []
#     labels = []
#     for data in b:
#         audio = my_load_and_transform_audio_data(
#                         torch.tensor(data["audio"]["array"], dtype=torch.float32
#                     ).unsqueeze(0))[0]
#         audios.append(audio)
#         text = data.get("text").lower() if data.get("text") else None
#         instruction = data["instruction"].lower()
        
#         prompts.append(llama.format_prompt(instruction, text))
#         labels.append(data["label"])
    
#     batch["audio"] = torch.stack(audios)
#     batch["prompts"] = prompts
#     batch["instructions"] = [d["instruction"] for d in b]
#     batch["labels"] = [d["label"] for d in b]
#     return batch
# def collate_fn(b):
#     max_length = 128

#     all_audios = []
#     all_input_ids = []
#     all_labels = []
#     all_input_mask = []

#     for data in b:
#         audio = my_load_and_transform_audio_data(
#             torch.tensor(data["audio"]["array"]).unsqueeze(0), dtype=torch.float32
#         )[0]
#         text = data.get("text").lower() if data.get("text") else None
#         instruction = data["instruction"].lower()

#         input1 = llama.format_prompt(instruction, text)
#         input2 = input1 + data["label"]
#         input1 = torch.tensor(
#             tokenizer.encode(input1, bos=True, eos=False), dtype=torch.long
#         )
#         input2 = torch.tensor(tokenizer.encode(input2, bos=True, eos=True), dtype=torch.long)
#         padding = max_length - input2.size(0)
#         if padding > 0:
#             input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.long) - 1))
#         else:
#             input2 = input2[:max_length]
        
#         labels = copy.deepcopy(input2)
#         labels[:input1.size(0)] = -1

#         input2_mask = input2.ge(0)
#         label_mask = labels.ge(0)
#         input2[~input2_mask] = 0
#         labels[~label_mask] = 0

#         input2_mask = input2_mask.float()
#         label_mask = label_mask.float()

#         all_audios.append(audio)
#         all_input_ids.append(input2)
#         all_input_mask.append(input2_mask)
#         all_labels.append(labels)
    
#     return {
#         "audio": torch.stack(all_audios),
#         "input_ids": torch.stack(all_input_ids),
#         "labels": torch.stack(all_labels),
#         "input_mask": torch.stack(all_input_mask)
#     }

# def collate_fn(b):
#     pass


def main(args):
    print("Start")
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # define the model
    llama_type = args.llama_type
    llama_ckpt_dir = os.path.join(args.llama_path, llama_type)
    llama_tokenzier_path = os.path.join(args.llama_path, 'tokenizer.model')
    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    print(f"Total Params: {sum(p.numel() for p in model.parameters()) / 1024 / 1024:.2f} M")
    print(f"Trainable Params:: {sum(p.numel() for p in model.parameters() if p.requires_grad)/ 1024 / 1024:.2f} M")

    for key, val in model.named_parameters():
        print(key, val.shape, val.requires_grad)
    # print([(key, val.shape) for key, val in model.named_parameters() if val.requires_grad])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # training detail
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(model_without_ddp, args.pretrained_path)

    # hank
    all_datasets = [
        'BigSuperbPrivate/SpoofDetection_Asvspoof2017',
        'BigSuperbPrivate/DailyTalk_DialogueActClassification',
        'BigSuperbPrivate/PronounciationEvaluationProsodic_Speechocean762',
        'BigSuperbPrivate/PronounciationEvaluationFluency_Speechocean762',
        'BigSuperbPrivate/PronounciationEvaluationOverall_Speechocean762',
        'BigSuperbPrivate/PronounciationEvaluationAccuracy_Speechocean762',
        # 'BigSuperbPrivate/HowFarAreYou_DeeplyParentChildVocalInteraction',
        # 'BigSuperbPrivate/HowFarAreYou_KoreanReadSpeechCorpus',
        'BigSuperbPrivate/SpeakerVerification_Tedlium2Train',
        'BigSuperbPrivate/SpeechDetection_Aishell1Train',
        'BigSuperbPrivate/SpeakerVerification_LibrispeechTrainClean100',
        'BigSuperbPrivate/SpeakerVerification_Aishell1Train',
        'BigSuperbPrivate/SpeechDetection_Voxceleb1Train',
        'BigSuperbPrivate/SpeakerVerification_Voxceleb1Train',
        'BigSuperbPrivate/SpokenTermDetection_Tedlium2Train',
        'BigSuperbPrivate/NoiseSNRLevelPredictionSpeech_VoxcelebMusan',
        'BigSuperbPrivate/SpeechDetection_LibrispeechTrainClean100',
        'BigSuperbPrivate/NoiseSNRLevelPredictionNoise_VoxcelebMusan',
        'BigSuperbPrivate/SpeechDetection_Tedlium2Train',
        'BigSuperbPrivate/EnhancementDetection_LibrittsTrainClean360Wham',
        'BigSuperbPrivate/SpeakerCounting_LibrittsTrainClean100',
        'BigSuperbPrivate/NoiseSNRLevelPredictionGaussian_VoxcelebMusan',
        'BigSuperbPrivate/NoiseSNRLevelPredictionMusic_VoxcelebMusan',
        'BigSuperbPrivate/SpeechTextMatching_Tedlium2Train',
        'BigSuperbPrivate/ReverberationDetectionSmallRoom_VoxcelebRirsNoises',
        'BigSuperbPrivate/ReverberationDetectionMediumRoom_VoxcelebRirsNoises',
        'BigSuperbPrivate/SpokenTermDetection_LibrispeechTrainClean100',
        'BigSuperbPrivate/ReverberationDetectionLargeRoom_VoxcelebRirsNoises',
        'BigSuperbPrivate/NoiseDetectionSpeech_VoxcelebMusan',
        'BigSuperbPrivate/NoiseDetectionNoise_VoxcelebMusan',
        'BigSuperbPrivate/NoiseDetectionMusic_VoxcelebMusan',
        'BigSuperbPrivate/NoiseDetectionGaussian_VoxcelebMusan',
        'BigSuperbPrivate/SpoofDetection_ASVspoof2015',
        'BigSuperbPrivate/SpeechTextMatching_LibrispeechTrainClean100'
    ]
    
    data_path = Path("/work/u8915687/big-superb")
    all_train_dataset = []
    for dataset_name in all_datasets:
        # dataset_path = (data_path/"train_datasets"/(dataset_name.split("/")[-1]))
        dataset_path = (data_path/(dataset_name+"_train5000"))
        train_dataset = load_from_disk(dataset_path)
        if "text" not in train_dataset.features:
            train_dataset = train_dataset.add_column("text", [None]*len(train_dataset))
        train_dataset = train_dataset.remove_columns(
            list(set(train_dataset.column_names) - {'audio', 'instruction', 'label', 'text'})
        )

        # train_dataset = train_dataset.remove_columns(
        #     list(set(train_dataset.column_names) - {'audio', 'input_ids', 'labels', 'input_mask'})
        # ) # only audio, input_ids, labels, input_mask
        # assert train_dataset.features["audio"].feature.feature.feature.feature.dtype == "float32", dataset_name
        # if train_dataset.features["audio"].feature.feature.feature.feature.dtype != "float32":
        #     print("Error", dataset_name)
        all_train_dataset.append(train_dataset)
    all_train_dataset = concatenate_datasets(all_train_dataset)

    print("All train dataset size:", len(all_train_dataset))



    # print(dataset_train)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        all_train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        all_train_dataset, sampler=sampler_train,
        collate_fn=collate_fn, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # SummaryWrite
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        misc.save_model_with_grad(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     **{f'val_{k}': v for k, v in train_stats.items()}}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
