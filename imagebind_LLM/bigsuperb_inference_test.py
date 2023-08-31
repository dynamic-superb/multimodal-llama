import os
import torch
from llama.llama_adapter import LLaMA_adapter
import util.misc as misc
from datasets import load_from_disk, load_dataset, Dataset, disable_caching
from llama import Tokenizer
import llama
from ImageBind.data import my_load_and_transform_audio_data
from pathlib import Path
from tqdm import tqdm
import json
import logging
import argparse

disable_caching()

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--decode_300", action="store_true")
    
    return parser.parse_args()

tokenizer = Tokenizer(model_path="/home/u8915687/lab/big-superb/Macaw-LLM2/weights/llama_7B/tokenizer.model")

@torch.inference_mode()
def forward_inference(self, visual_feats, tokens, start_pos: int):
    _bsz, seqlen = tokens.shape
    h = self.llama.tok_embeddings(tokens)
    freqs_cis = self.llama.freqs_cis.to(h.device)
    freqs_cis = freqs_cis[start_pos:start_pos + seqlen]
    mask = None
    mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
    mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)


    for layer in self.llama.layers[:-1 * self.query_layer]:
        h = layer(h, start_pos, freqs_cis, mask)
    prefix_query = self.prefix_query.weight.reshape(
        self.query_layer, 1, 4096).unsqueeze(1)
    prefix_index = 0
    
    visual_proj = []
    for i in range(_bsz):
        visual_proj.append(
            visual_feats[i, :, i, :]
        )
    visual_proj = torch.stack(visual_proj)
#     print(visual_proj)
        # B, 1, D
    for layer in self.llama.layers[-1 * self.query_layer:]:
        h = layer(h, start_pos, freqs_cis, mask, visual_proj + prefix_query[prefix_index].repeat(_bsz, 1, 1))
        prefix_index = prefix_index + 1

    h = self.llama.norm(h)
    output = self.llama.output(h[:, -1, :])

    return output.float()


@torch.inference_mode()
def my_generate(
        self,
        inputs,
        prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
        cache_size=10,
        cache_t=20,
        cache_weight=0.5
):
    bsz = len(prompts)
    params = self.llama.params
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    with torch.cuda.amp.autocast():
        visual_query = self.forward_visual(inputs, cache_size, cache_t, cache_weight)

    if isinstance(prompts[0], str):
        prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    min_prompt_size = min([len(t) for t in prompts])
    max_prompt_size = max([len(t) for t in prompts])

    total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

    tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
    
    reach_eos = torch.zeros(bsz, dtype=torch.bool)
    for k, t in enumerate(prompts):
        tokens[k, : len(t)] = torch.tensor(t).cuda().long()
    input_text_mask = tokens != self.tokenizer.pad_id
    start_pos = min_prompt_size
    prev_pos = 0
    for cur_pos in range(start_pos, total_len):
        with torch.cuda.amp.autocast():
            logits = forward_inference(self, visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)

        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        # trick: early stop if bsz==1
        

        for i in range(bsz):
            if next_token[i] == self.tokenizer.eos_id:
                reach_eos[i] = True
        if (bsz == 1 and next_token[0] == self.tokenizer.eos_id) or (all(reach_eos)):
            break
        
    
        prev_pos = cur_pos

    decoded = []
    for i, t in enumerate(tokens.tolist()):

        # cut to max gen len
        t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
        # cut to eos tok if any
        try:
            t = t[: t.index(self.tokenizer.eos_id)]
        except ValueError:
            pass
        decoded.append(self.tokenizer.decode(t))

    return decoded


        
def collate_fn(b):
    batch = {}
    
    audios = []
    prompts = []
    labels = []
    for data in b:
        audio = my_load_and_transform_audio_data(
                        torch.tensor(data["audio"]["array"], dtype=torch.float32
                    ).unsqueeze(0))[0]
        audios.append(audio)
        text = data.get("text").lower() if data.get("text") else None
        instruction = data["instruction"].lower()
        
        prompts.append(llama.format_prompt(instruction, text))
        labels.append(data["label"])
    
    batch["audio"] = torch.stack(audios)
    batch["prompts"] = prompts
    batch["instructions"] = [d["instruction"] for d in b]
    batch["labels"] = [d["label"] for d in b]
    return batch


def main(args):
    all_datasets = ['SpeechBigBench/AccentClassification_AccentdbExtended',
    'SpeechBigBench/BirdSoundDetection_Warblrb10k',
    'SpeechBigBench/ChordClassification_AcousticGuitarAndPiano',
    'SpeechBigBench/Deeply_Parent_Child_Vocal_Interaction',
    'SpeechBigBench/DialogueActClassification_DailyTalk',
    'SpeechBigBench/DialogueEmotionClassification_DailyTalk',
    'SpeechBigBench/EmotionRecognition_MultimodalEmotionlinesDataset',
    'SpeechBigBench/EnhancementDetection_LibrittsTestCleanWham',
    'SpeechBigBench/EnvironmentalSoundClassification_AnimalsESC50',
    'SpeechBigBench/EnvironmentalSoundClassification_ExteriorAndUrbanNoisesESC50',
    'SpeechBigBench/EnvironmentalSoundClassification_HumanAndNonSpeechSoundsESC50',
    'SpeechBigBench/EnvironmentalSoundClassification_InteriorAndDomesticSoundsESC50',
    'SpeechBigBench/EnvironmentalSoundClassification_NaturalSoundscapesAndWaterSoundsESC50',
    'SpeechBigBench/HowFarAreYou_3DSpeaker',
    'SpeechBigBench/IntentClassification_FluentSpeechCommands',
    'SpeechBigBench/Korean_Read_Speech_Corpus',
    'SpeechBigBench/LanguageIdentification_VoxForge',
    'SpeechBigBench/NoiseDetectiongaussian_LJSpeechMusan',
    'SpeechBigBench/NoiseDetectiongaussian_VCTKMusan',
    'SpeechBigBench/NoiseDetectionmusic_LJSpeechMusan',
    'SpeechBigBench/NoiseDetectionmusic_VCTKMusan',
    'SpeechBigBench/NoiseDetectionnoise_LJSpeechMusan',
    'SpeechBigBench/NoiseDetectionnoise_VCTKMusan',
    'SpeechBigBench/NoiseDetectionspeech_LJSpeechMusan',
    'SpeechBigBench/NoiseDetectionspeech_VCTKMusan',
    'SpeechBigBench/NoiseSNRLevelPredictiongaussian_VCTKMusan',
    'SpeechBigBench/NoiseSNRLevelPredictionmusic_VCTKMusan',
    'SpeechBigBench/NoiseSNRLevelPredictionnoise_VCTKMusan',
    'SpeechBigBench/NoiseSNRLevelPredictionspeech_VCTKMusan',
    'SpeechBigBench/ReverberationDetectionlargeroom_LJSpeechRirsNoises',
    'SpeechBigBench/ReverberationDetectionlargeroom_VCTKRirsNoises',
    'SpeechBigBench/ReverberationDetectionmediumroom_LJSpeechRirsNoises',
    'SpeechBigBench/ReverberationDetectionmediumroom_VCTKRirsNoises',
    'SpeechBigBench/ReverberationDetectionsmallroom_LJSpeechRirsNoises',
    'SpeechBigBench/ReverberationDetectionsmallroom_VCTKRirsNoises',
    'SpeechBigBench/SarcasmDetection_Mustard',
    'SpeechBigBench/SpeakerCounting_LibriTTSTestClean',
    'SpeechBigBench/SpeechCommandRecognition_GoogleSpeechCommandsV1',
    'SpeechBigBench/SpeechDetection_LJSpeech',
    'SpeechBigBench/SpeechDetection_LibriSpeechTestClean',
    'SpeechBigBench/SpeechDetection_LibriSpeechTestOther',
    'SpeechBigBench/SpeechTextMatching_LJSpeech',
    'SpeechBigBench/SpeechTextMatching_LibriSpeechTestClean',
    'SpeechBigBench/SpeechTextMatching_LibriSpeechTestOther',
    'SpeechBigBench/SpokenTermDetection_LJSpeech',
    'SpeechBigBench/SpokenTermDetection_LibriSpeechTestClean',
    'SpeechBigBench/SpokenTermDetection_LibriSpeechTestOther',
    'SpeechBigBench/SpoofDetection_ASVspoof2015',
    'SpeechBigBench/SpoofDetection_ASVspoof2017',
    'SpeechBigBench/StressDetection_MIRSD']

    # Config #
    
    DATA_PATH = Path("/work/u8915687/big-superb")
    ROOT_PATH = Path("/home/u8915687/lab/big-superb/LLaMA-Adapter/imagebind_LLM")
    EXP_PATH = ROOT_PATH/args.exp_path # exp/finetune2
    pretrained_path = EXP_PATH/args.model_path # "checkpoint-4.pth"
    is_decode_300 = args.decode_300 # true / false

    RESULT_PATH = EXP_PATH/"results"
    if is_decode_300:
        RESULT_PATH /= "subset"
    else:
        RESULT_PATH /= "full"


    RESULT_PATH.mkdir(parents=True, exist_ok=True)


    logging.basicConfig(filename=str(RESULT_PATH/"inference.log"), level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.info(f"pretrained_path: {pretrained_path}")
    logging.info(f"result_path: {RESULT_PATH}")
    logging.info(f"is_decode_300: {is_decode_300}")

    # Config #

    llama_dir = "/home/u8915687/lab/big-superb/Macaw-LLM2/weights/llama/"

    model = llama.load(pretrained_path, llama_dir, knn=True, max_batch_size=16)
    model.eval()

    for dataset_name in all_datasets:

        logging.info(dataset_name)
        task_name = dataset_name.split("/")[-1]
        
        dataset = load_from_disk(
            DATA_PATH/dataset_name
        )
        
        if dataset.get("test"):
            dataset = dataset["test"]
        elif dataset.get("train"):
            dataset = dataset["train"]
        else:
            logging.info(f"Error {dataset_name}")
            continue

        if ("instruction" not in dataset.column_names) or ("label" not in dataset.column_names) or ("audio" not in dataset.column_names):
            logging.info(f"Error {dataset_name}, column not found")
            continue
        if (RESULT_PATH/f"{task_name}.json").exists():
            logging.info(f"SKIP {dataset_name}")
            continue
        
        if is_decode_300:
            dataset = Dataset.from_dict(dataset.shuffle(42)[:300])

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        correct = 0
        count = 0
        predictions = []
        label_count = {}
        pred_count = {}
        for batch in tqdm(data_loader):
            inputs = {
                "Audio": [batch["audio"].to("cuda"), 1]
            }
            results = my_generate(
                model,
                inputs,
                batch["prompts"],
                max_gen_len=150,
                temperature=0,
                top_p=0
            )

            
            for r,l, ins in zip(results, batch["labels"], batch["instructions"]):
                r = r.strip()
                if r == l:
                    correct += 1
                count += 1
                
                label_count[l] = label_count.get(l, 0) + 1
                pred_count[r] = pred_count.get(r, 0) + 1
                predictions.append({
                    "pred": r,
                    "label": l,
                    "instruction": ins
                })
        logging.info(f"{correct} / {len(predictions)}")
        json.dump(
            {
                "predictions": predictions,
                "label_count": label_count,
                "pred_count": pred_count,
                "accuracy": correct / len(predictions)
            },
            (RESULT_PATH/f"{task_name}.json").open("w"), indent=2
        )
        logging.info("="*100)


if __name__ == "__main__":
    args = get_args_parser()
    main(args=args)