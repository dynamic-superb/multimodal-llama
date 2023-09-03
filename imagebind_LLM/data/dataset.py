import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
import torchaudio
import whisper

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

class FinetuneDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        ann = []
        for meta_path in self.config['META']:
            meta_l = json.load(open(meta_path))
            print(f"{meta_path}: len {len(meta_l)}")
            ann += meta_l
        self.ann = ann
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item = self.ann[index]
        if 'image' in data_item.keys():
            filename = data_item['image']
            question = data_item['conversations'][0]['value']
            answer = data_item['conversations'][1]['value']
            # < fill path substitution logics here>
            # filename = url.replace("/data0/data/coco/", "/mnt/petrelfs/leimeng/datasets/coco/")

            image = Image.open(filename).convert('RGB')
            image = self.transform(image)
            format_instruction = question
            format_input = None
        else:
            image = torch.zeros(3, 224, 224)
            format_instruction = data_item['instruction'],
            format_input = data_item['input']
            answer = data_item['output']
        input1 = llama.utils.format_prompt(format_instruction, format_input)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image


class PretrainDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        images, captions = [], []
        for meta_path in self.config['META']:
            images_this_meta, captions_this_meta = [], []
            for chunk in pd.read_csv(meta_path, sep='\t', lineterminator='\n', chunksize=10 ** 6):
                images_this_meta.extend(chunk['url'].tolist())
                captions_this_meta.extend(chunk['caption'].tolist())
            print(f"{meta_path}: len {len(images_this_meta)}")
            images.extend(images_this_meta)
            captions.extend(captions_this_meta)

        self.data_list = []
        for x, y in zip(images, captions):
            self.data_list.append({'url': x, 'caption': y})
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, caption = sample['url'], sample['caption']
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption)

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        format_instruction = "Generate caption of this image"
        input1 = llama.utils.format_prompt(format_instruction, None)
        input2 = input1 + caption

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image


# Hank
import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
import torchaudio


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class BigSuperbDataset(Dataset):
    def __init__(self, data_path, tokenizer, data_path2=None, max_length=128, used_data_split="train", audio_input_type="imagebind"):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.datas = []
        self.used_datasets = []
        self.used_data_split = used_data_split
        self.audio_input_type = audio_input_type

        for task_path in data_path.iterdir():
            if self._filter_dataset(task_path):
                continue
            
            self.used_datasets.append(task_path.stem)
            
            for data_split in task_path.iterdir():
                if data_split.stem != self.used_data_split:
                    continue
                
                json_data = json.load((data_split/"metadata.json").open())
                for d in json_data.values():
                    d["file"] = str(data_split/d["file"])
                    if d.get("file2"):
                        if "." not in d["file2"]:
                            d["file2"] = d["file2"] + ".wav"
                            
                        if (data_split/d["file2"]).exists():
                            d["file2"] = str(data_split/d["file2"])
                        elif (task_path/"missing_files"/d["file2"]).exists():
                            d["file2"] = str(task_path/"missing_files"/d["file2"])
                        else:
                            assert False, d["file2"]
                            
                    self.datas.append(d)
        # exclude
        if data_path2 is not None:
            for task_path in data_path2.iterdir():
                if self._filter_dataset(task_path):
                    continue
                
                self.used_datasets.append(task_path.stem)
                for data_split in task_path.iterdir():
                    if data_split.stem != self.used_data_split:
                        continue
                    
                    json_data = json.load((data_split/"metadata.json").open())
                    for file_name, d in json_data.items():
                        d["file"] = str(data_split/file_name)
                                
                        self.datas.append(d)


        # Audio loader
        self.clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=2, clips_per_video=3
        )
        print("Used datasets", len(self.used_datasets), self.used_datasets)
        print(len(self.datas))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        new_data = {}

        instruction = data["instruction"].lower()
        text = data["text"].lower() if data.get("text") else None
        input1 = llama.format_prompt(instruction, text)
        input2 = input1 + data["label"]
        
        input1 = torch.tensor(
            self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.long
        )
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.long)
        padding = self.max_length - input2.size(0)
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.long) - 1))
        else:
            input2 = input2[:self.max_length]
        
        labels = copy.deepcopy(input2)
        labels[:input1.size(0)] = -1
        
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        
        new_data["instruction"] = data["instruction"]
        new_data["input_ids"] = input2
        new_data["labels"] = labels
        new_data["input_mask"] = input2_mask

        if self.audio_input_type == "imagebind":
            if data.get("file2"):
                audio = self._load_and_transform_audio([data["file"], data["file2"]])
            else:
                audio = self._load_and_transform_audio([data["file"]])
                
            
        elif self.audio_input_type == "whisper":
            if data.get("file2"):
                audio = self._load_whisper_audio([data["file"], data["file2"]])
            else:
                audio = self._load_whisper_audio([data["file"]])
        new_data["audio"] = audio
        return new_data
    
    def _load_whisper_audio(self, audio_paths):
        wavforms = []
        for audio_path in audio_paths:
            waveform = torch.tensor(whisper.load_audio(audio_path))
            if waveform.size(0) == 0:
                waveform = torch.zeros([16000*3])
                print(audio_path)
            
            wavforms.append(
                waveform
            )
        audio = torch.cat(wavforms, dim=0)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        
        return mel

    
    def _load_and_transform_audio(self, 
            audio_paths,
            num_mel_bins=128,
            target_length=204,
            sample_rate=16000,
            clip_duration=2,
            clips_per_video=3,
            mean=-4.268,
            std=9.138
        ):

        waveforms = []
        for audio_path in audio_paths:
            waveform, sr = torchaudio.load(audio_path)

            if waveform.size(1) == 0:
                waveform = torch.zeros([1, 16000*3])
                sr = 16000
                # logging.warning(f"An audio is set to zero, {audio_path}")
                print(audio_path)
                
            if sample_rate != sr:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=sample_rate
                )

            waveforms.append(waveform)
        waveform = torch.cat(waveforms, dim=1)
            
        all_clips_timepoints = self._get_clip_timepoints(
            self.clip_sampler, waveform.size(1) / sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate) : int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            waveform_melspec = self._def_waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)

        return all_clips
        
    def _get_clip_timepoints(self, clip_sampler, duration):
        # Read out all clips in this video
        all_clips_timepoints = []
        is_last_clip = False
        end = 0.0
        while not is_last_clip:
            start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
            all_clips_timepoints.append((start, end))
        return all_clips_timepoints

    def _def_waveform2melspec(self, waveform, sample_rate, num_mel_bins, target_length):
        # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
        waveform -= waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=10,
        )
        # Convert to [mel_bins, num_frames] shape
        fbank = fbank.transpose(0, 1)
        # Pad to target_length
        n_frames = fbank.size(1)
        p = target_length - n_frames
        # if p is too large (say >20%), flash a warning
        # if abs(p) / n_frames > 0.2:
            # logging.warning(
            #     "Large gap between audio n_frames(%d) and "
            #     "target_length (%d). Is the audio_target_length "
            #     "setting correct?",
            #     n_frames,
            #     target_length,
            # )
        # cut and pad
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
        elif p < 0:
            fbank = fbank[:, 0:target_length]
        # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
        # channel image
        fbank = fbank.unsqueeze(0)
        return fbank
    
    def _filter_dataset(self, task_path):
        if task_path.stem.startswith("HowFarAreYou"):
            return True
        else:
            return False
