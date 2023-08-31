from pathlib import Path
from datasets import load_dataset, Audio, Dataset, load_from_disk, disable_caching
from llama import Tokenizer
import llama
import torch
import copy
from ImageBind.data import my_load_and_transform_audio_data
import logging


disable_caching()
logging.basicConfig(filename="test_preprocess.log", level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

tokenizer = Tokenizer(model_path="/home/u8915687/lab/big-superb/Macaw-LLM2/weights/llama_7B/tokenizer.model")
def prepare_dataset(b):
    max_length = 128
    
    batch = {}
    instruction = b["instruction"].lower()
    text = b["text"].lower() if b.get("text") else None
    input1 = llama.format_prompt(instruction, text)
    input2 = input1 + b["label"]
    
    input1 = torch.tensor(
        tokenizer.encode(input1, bos=True, eos=False), dtype=torch.long
    )
    input2 = torch.tensor(tokenizer.encode(input2, bos=True, eos=True), dtype=torch.long)
    
    
    padding = max_length - input2.size(0)
    if padding > 0:
        input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.long) - 1))
    else:
        input2 = input2[:max_length]
    
    labels = copy.deepcopy(input2)
    labels[:input1.size(0)] = -1
    
    input2_mask = input2.ge(0)
    label_mask = labels.ge(0)
    input2[~input2_mask] = 0
    labels[~label_mask] = 0
    
    input2_mask = input2_mask.float()
    label_mask = label_mask.float()
    
    batch["instruction"] = b["instruction"]
    batch["input_ids"] = input2
    batch["labels"] = labels
    batch["input_mask"] = input2_mask

    assert b["audio"]["sampling_rate"] == 16000

    if len(b["audio"]["array"]) == 0:
        b["audio"]["array"] = [0.] * 16000 * 5 # 5 sec
        logging.warning(f"An audio is set to zero, {b['audio']['path']}")
    
    try:
        batch["audio"] = my_load_and_transform_audio_data(
            torch.tensor(b["audio"]["array"]).unsqueeze(0)
        )[0]
    except:
        logging.error(
            torch.tensor(b["audio"]["array"]).size()
        )
    return batch


def main():
    data_path = Path("/work/u8915687/big-superb")
    # all_datasets = ['BigSuperbPrivate/SpoofDetection_Asvspoof2017', 'BigSuperbPrivate/DailyTalk_DialogueActClassification', 'BigSuperbPrivate/PronounciationEvaluationProsodic_Speechocean762', 'BigSuperbPrivate/PronounciationEvaluationFluency_Speechocean762', 'BigSuperbPrivate/PronounciationEvaluationOverall_Speechocean762', 'BigSuperbPrivate/PronounciationEvaluationAccuracy_Speechocean762', 'BigSuperbPrivate/HowFarAreYou_DeeplyParentChildVocalInteraction', 'BigSuperbPrivate/HowFarAreYou_KoreanReadSpeechCorpus', 'BigSuperbPrivate/SpeakerVerification_Tedlium2Train', 'BigSuperbPrivate/SpeechDetection_Aishell1Train', 'BigSuperbPrivate/SpeakerVerification_LibrispeechTrainClean100', 'BigSuperbPrivate/SpeakerVerification_Aishell1Train', 'BigSuperbPrivate/SpeechDetection_Voxceleb1Train', 'BigSuperbPrivate/SpeakerVerification_Voxceleb1Train', 'BigSuperbPrivate/SpokenTermDetection_Tedlium2Train', 'BigSuperbPrivate/NoiseSNRLevelPredictionSpeech_VoxcelebMusan', 'BigSuperbPrivate/SpeechDetection_LibrispeechTrainClean100', 'BigSuperbPrivate/NoiseSNRLevelPredictionNoise_VoxcelebMusan', 'BigSuperbPrivate/SpeechDetection_Tedlium2Train', 'BigSuperbPrivate/EnhancementDetection_LibrittsTrainClean360Wham', 'BigSuperbPrivate/SpeakerCounting_LibrittsTrainClean100', 'BigSuperbPrivate/NoiseSNRLevelPredictionGaussian_VoxcelebMusan', 'BigSuperbPrivate/NoiseSNRLevelPredictionMusic_VoxcelebMusan', 'BigSuperbPrivate/SpeechTextMatching_Tedlium2Train', 'BigSuperbPrivate/ReverberationDetectionSmallRoom_VoxcelebRirsNoises', 'BigSuperbPrivate/ReverberationDetectionMediumRoom_VoxcelebRirsNoises', 'BigSuperbPrivate/SpokenTermDetection_LibrispeechTrainClean100', 'BigSuperbPrivate/ReverberationDetectionLargeRoom_VoxcelebRirsNoises', 'BigSuperbPrivate/NoiseDetectionSpeech_VoxcelebMusan', 'BigSuperbPrivate/NoiseDetectionNoise_VoxcelebMusan', 'BigSuperbPrivate/NoiseDetectionMusic_VoxcelebMusan', 'BigSuperbPrivate/NoiseDetectionGaussian_VoxcelebMusan', 'BigSuperbPrivate/SpoofDetection_ASVspoof2015', 'BigSuperbPrivate/SpeechTextMatching_LibrispeechTrainClean100']

    all_datasets = ['SpeechBigBench/AccentClassification_AccentdbExtended', 'SpeechBigBench/BirdSoundDetection_Warblrb10k', 'SpeechBigBench/ChordClassification_AcousticGuitarAndPiano', 'SpeechBigBench/Deeply_Parent_Child_Vocal_Interaction', 'SpeechBigBench/DialogueActClassification_DailyTalk', 'SpeechBigBench/DialogueEmotionClassification_DailyTalk', 'SpeechBigBench/EmotionRecognition_MultimodalEmotionlinesDataset', 'SpeechBigBench/EnhancementDetection_LibrittsTestCleanWham', 'SpeechBigBench/EnvironmentalSoundClassification_AnimalsESC50', 'SpeechBigBench/EnvironmentalSoundClassification_ExteriorAndUrbanNoisesESC50', 'SpeechBigBench/EnvironmentalSoundClassification_HumanAndNonSpeechSoundsESC50', 'SpeechBigBench/EnvironmentalSoundClassification_InteriorAndDomesticSoundsESC50', 'SpeechBigBench/EnvironmentalSoundClassification_NaturalSoundscapesAndWaterSoundsESC50', 'SpeechBigBench/HowFarAreYou_3DSpeaker', 'SpeechBigBench/IntentClassification_FluentSpeechCommands', 'SpeechBigBench/Korean_Read_Speech_Corpus', 'SpeechBigBench/LanguageIdentification_VoxForge', 'SpeechBigBench/NoiseDetectiongaussian_LJSpeechMusan', 'SpeechBigBench/NoiseDetectiongaussian_VCTKMusan', 'SpeechBigBench/NoiseDetectionmusic_LJSpeechMusan', 'SpeechBigBench/NoiseDetectionmusic_VCTKMusan', 'SpeechBigBench/NoiseDetectionnoise_LJSpeechMusan', 'SpeechBigBench/NoiseDetectionnoise_VCTKMusan', 'SpeechBigBench/NoiseDetectionspeech_LJSpeechMusan', 'SpeechBigBench/NoiseDetectionspeech_VCTKMusan', 'SpeechBigBench/NoiseSNRLevelPredictiongaussian_VCTKMusan', 'SpeechBigBench/NoiseSNRLevelPredictionmusic_VCTKMusan', 'SpeechBigBench/NoiseSNRLevelPredictionnoise_VCTKMusan', 'SpeechBigBench/NoiseSNRLevelPredictionspeech_VCTKMusan', 'SpeechBigBench/Nonverbal_Vocalization', 'SpeechBigBench/PronounciationEvaluationAccuracy_Speechocean762', 'SpeechBigBench/PronounciationEvaluationFluency_Speechocean762', 'SpeechBigBench/PronounciationEvaluationOverall_Speechocean762', 'SpeechBigBench/PronounciationEvaluationProsodic_Speechocean762', 'SpeechBigBench/ReverberationDetectionlargeroom_LJSpeechRirsNoises', 'SpeechBigBench/ReverberationDetectionlargeroom_VCTKRirsNoises', 'SpeechBigBench/ReverberationDetectionmediumroom_LJSpeechRirsNoises', 'SpeechBigBench/ReverberationDetectionmediumroom_VCTKRirsNoises', 'SpeechBigBench/ReverberationDetectionsmallroom_LJSpeechRirsNoises', 'SpeechBigBench/ReverberationDetectionsmallroom_VCTKRirsNoises', 'SpeechBigBench/SarcasmDetection_Mustard', 'SpeechBigBench/SpeakerCounting_LibriTTSTestClean', 'SpeechBigBench/SpeechCommandRecognition_GoogleSpeechCommandsV1', 'SpeechBigBench/SpeechDetection_LJSpeech', 'SpeechBigBench/SpeechDetection_LibriSpeechTestClean', 'SpeechBigBench/SpeechDetection_LibriSpeechTestOther', 'SpeechBigBench/SpeechTextMatching_LJSpeech', 'SpeechBigBench/SpeechTextMatching_LibriSpeechTestClean', 'SpeechBigBench/SpeechTextMatching_LibriSpeechTestOther', 'SpeechBigBench/SpokenTermDetection_LJSpeech', 'SpeechBigBench/SpokenTermDetection_LibriSpeechTestClean', 'SpeechBigBench/SpokenTermDetection_LibriSpeechTestOther', 'SpeechBigBench/SpoofDetection_ASVspoof2015', 'SpeechBigBench/SpoofDetection_ASVspoof2017', 'SpeechBigBench/StressDetection_MIRSD', 'SpeechBigBench/arabic_speech_corpus', 'SpeechBigBench/speech_commands']


    for dataset_name in all_datasets:
        logging.info(f"Start {dataset_name}")
        
        dataset_path = (data_path/(dataset_name))
        target_path = (data_path/"test_datasets"/(dataset_name.split("/")[-1]))
        logging.info(f"{dataset_path}")

        if target_path.exists():
            logging.info(f"Skip {target_path}")
        else:
        
            test_dataset = load_from_disk(dataset_path)
            if test_dataset.get("test"):
                test_dataset = test_dataset["test"]
                logging.info(f"Loaded {dataset_path}")
            else:
                logging.error(f"No test set {dataset_name}")
            test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
            test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)
            test_dataset = test_dataset.with_format("torch")
            
            # /work/u8915687/big-superb/test_datasets/ AAA_BBB
            test_dataset.save_to_disk(target_path)
            logging.info(f"Saved {target_path}")
        logging.info(f"="*70)


if __name__ == "__main__":
    main()