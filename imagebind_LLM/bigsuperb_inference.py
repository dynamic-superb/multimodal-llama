import os
import torch
from llama.llama_adapter import LLaMA_adapter
import util.misc as misc
from datasets import load_from_disk, load_dataset, Dataset
from llama import Tokenizer
import llama
from ImageBind.data import my_load_and_transform_audio_data
from pathlib import Path
import logging
import json



DATA_PATH = Path("/work/u8915687/big-superb")
ROOT_PATH = Path("/home/u8915687/lab/big-superb/LLaMA-Adapter/imagebind_LLM")
EXP_PATH = ROOT_PATH/ "exp/finetune2"
RESULT_PATH = EXP_PATH/"results3"

if __name__ == "__main__":

    llama_dir = "/home/u8915687/lab/big-superb/Macaw-LLM2/weights/llama/"
    tokenizer = Tokenizer(model_path="/home/u8915687/lab/big-superb/Macaw-LLM2/weights/llama_7B/tokenizer.model")
    
    finetuned_path = EXP_PATH/"checkpoint-3.pth"
    model = llama.load(finetuned_path, llama_dir, knn=True)

    
    # finetune_dict = torch.load(pretrained_path, map_location="cpu")
    # logging.info(model.load_state_dict(finetune_dict))
    model.eval()

    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    print(RESULT_PATH)
    logging.basicConfig(filename=str(RESULT_PATH/"inference.log"), level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


    
    all_datasets = ['BigSuperbPrivate/SpoofDetection_Asvspoof2017', 'BigSuperbPrivate/DailyTalk_DialogueActClassification', 'BigSuperbPrivate/PronounciationEvaluationProsodic_Speechocean762', 'BigSuperbPrivate/PronounciationEvaluationFluency_Speechocean762', 'BigSuperbPrivate/PronounciationEvaluationOverall_Speechocean762', 'BigSuperbPrivate/PronounciationEvaluationAccuracy_Speechocean762', 'BigSuperbPrivate/HowFarAreYou_DeeplyParentChildVocalInteraction', 'BigSuperbPrivate/HowFarAreYou_KoreanReadSpeechCorpus', 'BigSuperbPrivate/SpeakerVerification_Tedlium2Train', 'BigSuperbPrivate/SpeechDetection_Aishell1Train', 'BigSuperbPrivate/SpeakerVerification_LibrispeechTrainClean100', 'BigSuperbPrivate/SpeakerVerification_Aishell1Train', 'BigSuperbPrivate/SpeechDetection_Voxceleb1Train', 'BigSuperbPrivate/SpeakerVerification_Voxceleb1Train', 'BigSuperbPrivate/SpokenTermDetection_Tedlium2Train', 'BigSuperbPrivate/NoiseSNRLevelPredictionSpeech_VoxcelebMusan', 'BigSuperbPrivate/SpeechDetection_LibrispeechTrainClean100', 'BigSuperbPrivate/NoiseSNRLevelPredictionNoise_VoxcelebMusan', 'BigSuperbPrivate/SpeechDetection_Tedlium2Train', 'BigSuperbPrivate/EnhancementDetection_LibrittsTrainClean360Wham', 'BigSuperbPrivate/SpeakerCounting_LibrittsTrainClean100', 'BigSuperbPrivate/NoiseSNRLevelPredictionGaussian_VoxcelebMusan', 'BigSuperbPrivate/NoiseSNRLevelPredictionMusic_VoxcelebMusan', 'BigSuperbPrivate/SpeechTextMatching_Tedlium2Train', 'BigSuperbPrivate/ReverberationDetectionSmallRoom_VoxcelebRirsNoises', 'BigSuperbPrivate/ReverberationDetectionMediumRoom_VoxcelebRirsNoises', 'BigSuperbPrivate/SpokenTermDetection_LibrispeechTrainClean100', 'BigSuperbPrivate/ReverberationDetectionLargeRoom_VoxcelebRirsNoises', 'BigSuperbPrivate/NoiseDetectionSpeech_VoxcelebMusan', 'BigSuperbPrivate/NoiseDetectionNoise_VoxcelebMusan', 'BigSuperbPrivate/NoiseDetectionMusic_VoxcelebMusan', 'BigSuperbPrivate/NoiseDetectionGaussian_VoxcelebMusan', 'BigSuperbPrivate/SpoofDetection_ASVspoof2015', 'BigSuperbPrivate/SpeechTextMatching_LibrispeechTrainClean100']
    
    for dataset_name in all_datasets:
        logging.info(f"Start {dataset_name}")
        task_name = dataset_name.split("/")[-1]
        dataset = load_from_disk(
            DATA_PATH/dataset_name,
        )
        if not dataset.get("validation"):
            continue
        
        dataset = dataset["validation"].shuffle(seed=42)
        dataset = Dataset.from_dict(dataset[:300])

        predictions = []
        for i in range(len(dataset)):
            batch = dataset[i]
            inputs = {}
            audio = my_load_and_transform_audio_data(torch.tensor(batch["audio"]["array"], dtype=torch.float32).unsqueeze(0))[0].unsqueeze(0)
            audio = audio.to("cuda")
            inputs["Audio"] = [audio, 1]
            
            
            prompt = llama.format_prompt(batch["instruction"])
            
            results = model.generate(
                inputs,
                [prompt],
                max_gen_len=150,
                temperature=0,
                top_p=0
            )
            result = results[0].strip()
            
            predictions.append({
                "pred": result,
                "label": batch["label"],
                "text": batch.get("text"),
                "instruction": batch["instruction"]
            })
        
        
        c = 0
        label_count = {}
        for r in predictions:
            if r["pred"] == r["label"]:
                c += 1
            label_count[r["label"]] = label_count.get(r["label"], 0) + 1
        logging.info(f"{c} / {len(predictions)}")
        logging.info(label_count)
        json.dump(
            {
                "predictions": predictions,
                "label_count": label_count,
                "accuracy": c / len(predictions)
            },
            (RESULT_PATH/f"{task_name}.json").open("w"), indent=2
        )