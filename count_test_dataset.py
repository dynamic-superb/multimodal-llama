from datasets import load_from_disk, Dataset, disable_caching
disable_caching()
from pathlib import Path
from collections import Counter, OrderedDict
from llama import Tokenizer
from tqdm import tqdm
import json
data_path = Path("/work/u8915687/big-superb")
tokenizer = Tokenizer(model_path="/home/u8915687/lab/big-superb/Macaw-LLM2/weights/llama_7B/tokenizer.model")
all_datasets = [
    'SpeechBigBench/NoiseDetectionspeech_LJSpeechMusan', 'SpeechBigBench/AccentClassification_AccentdbExtended', 'SpeechBigBench/BirdSoundDetection_Warblrb10k', 'SpeechBigBench/ChordClassification_AcousticGuitarAndPiano', 'SpeechBigBench/Deeply_Parent_Child_Vocal_Interaction', 'SpeechBigBench/DialogueActClassification_DailyTalk', 'SpeechBigBench/DialogueEmotionClassification_DailyTalk', 'SpeechBigBench/EmotionRecognition_MultimodalEmotionlinesDataset', 'SpeechBigBench/EnhancementDetection_LibrittsTestCleanWham', 'SpeechBigBench/EnvironmentalSoundClassification_AnimalsESC50', 'SpeechBigBench/EnvironmentalSoundClassification_ExteriorAndUrbanNoisesESC50', 'SpeechBigBench/EnvironmentalSoundClassification_HumanAndNonSpeechSoundsESC50', 'SpeechBigBench/EnvironmentalSoundClassification_InteriorAndDomesticSoundsESC50', 'SpeechBigBench/EnvironmentalSoundClassification_NaturalSoundscapesAndWaterSoundsESC50', 'SpeechBigBench/HowFarAreYou_3DSpeaker', 'SpeechBigBench/IntentClassification_FluentSpeechCommands', 'SpeechBigBench/Korean_Read_Speech_Corpus', 'SpeechBigBench/LanguageIdentification_VoxForge', 'SpeechBigBench/NoiseDetectiongaussian_LJSpeechMusan', 'SpeechBigBench/NoiseDetectiongaussian_VCTKMusan', 'SpeechBigBench/NoiseDetectionmusic_LJSpeechMusan', 'SpeechBigBench/NoiseDetectionmusic_VCTKMusan', 'SpeechBigBench/NoiseDetectionnoise_LJSpeechMusan', 'SpeechBigBench/NoiseDetectionnoise_VCTKMusan', 'SpeechBigBench/NoiseDetectionspeech_VCTKMusan', 'SpeechBigBench/NoiseSNRLevelPredictiongaussian_VCTKMusan', 'SpeechBigBench/NoiseSNRLevelPredictionmusic_VCTKMusan', 'SpeechBigBench/NoiseSNRLevelPredictionnoise_VCTKMusan', 'SpeechBigBench/NoiseSNRLevelPredictionspeech_VCTKMusan', 'SpeechBigBench/Nonverbal_Vocalization', 'SpeechBigBench/PronounciationEvaluationAccuracy_Speechocean762', 'SpeechBigBench/PronounciationEvaluationFluency_Speechocean762', 'SpeechBigBench/PronounciationEvaluationOverall_Speechocean762', 'SpeechBigBench/PronounciationEvaluationProsodic_Speechocean762', 'SpeechBigBench/ReverberationDetectionlargeroom_LJSpeechRirsNoises', 'SpeechBigBench/ReverberationDetectionlargeroom_VCTKRirsNoises', 'SpeechBigBench/ReverberationDetectionmediumroom_LJSpeechRirsNoises', 'SpeechBigBench/ReverberationDetectionmediumroom_VCTKRirsNoises', 'SpeechBigBench/ReverberationDetectionsmallroom_LJSpeechRirsNoises', 'SpeechBigBench/ReverberationDetectionsmallroom_VCTKRirsNoises', 'SpeechBigBench/SarcasmDetection_Mustard', 'SpeechBigBench/SpeakerCounting_LibriTTSTestClean', 'SpeechBigBench/SpeechCommandRecognition_GoogleSpeechCommandsV1', 'SpeechBigBench/SpeechDetection_LJSpeech', 'SpeechBigBench/SpeechDetection_LibriSpeechTestClean', 'SpeechBigBench/SpeechDetection_LibriSpeechTestOther', 'SpeechBigBench/SpeechTextMatching_LJSpeech', 'SpeechBigBench/SpeechTextMatching_LibriSpeechTestClean', 'SpeechBigBench/SpeechTextMatching_LibriSpeechTestOther', 'SpeechBigBench/SpokenTermDetection_LJSpeech', 'SpeechBigBench/SpokenTermDetection_LibriSpeechTestClean', 'SpeechBigBench/SpokenTermDetection_LibriSpeechTestOther', 'SpeechBigBench/SpoofDetection_ASVspoof2015', 'SpeechBigBench/SpoofDetection_ASVspoof2017', 'SpeechBigBench/StressDetection_MIRSD', 'SpeechBigBench/arabic_speech_corpus', 'SpeechBigBench/speech_commands']

stats = OrderedDict()
stats_sampled = OrderedDict()
for dataset_name in all_datasets:
    print(dataset_name)
    dataset = load_from_disk(data_path/dataset_name)
    if not dataset.get("test"):
        print(f"Error {dataset_name}")

        json.dump({"msg": "no test set"}
                  , Path(f"./stats/{dataset_name.split('/')[-1]}.json").open("w"), indent=2)
        continue
        
    dataset = dataset["test"]
    if ("label" not in dataset.features) or ("instruction" not in dataset.features) or ("audio" not in dataset.features):
        print(f"Error {dataset_name}")

        json.dump({"msg": "no field", "field": list(dataset.features)}
                  , Path(f"./stats/{dataset_name.split('/')[-1]}.json").open("w"), indent=2)
        continue
    
    if Path(f"./stats/{dataset_name.split('/')[-1]}.json").exists():
        print("Skip", dataset_name)
        continue
    print(dataset)
    
    label_counter = Counter()
    ins_counter = Counter()
    audio_sec = 0
    
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        label_counter[data["label"]] += 1
        ins_counter[data["instruction"]] += 1
        audio_sec += len(data["audio"]["array"]) / data["audio"]["sampling_rate"]
    
    stats_ = {
        "number": len(dataset),
        "labels": label_counter,
        "instructions": ins_counter,
        "total_audio_length": audio_sec,
        "avg_audio_length": audio_sec / len(dataset)
    }
    stats[dataset_name] = stats_
    
    
    json.dump(stats_, Path(f"./stats/{dataset_name.split('/')[-1]}.json").open("w"), indent=2)
    
    print(len(dataset))
    
    # Sampled
    dataset = dataset.shuffle(42)[:3000]
    dataset = Dataset.from_dict(dataset)
    print(dataset)
    
    label_counter = Counter()
    ins_counter = Counter()
    audio_sec = 0
    
    
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        label_counter[data["label"]] += 1
        ins_counter[data["instruction"]] += 1
        audio_sec += len(data["audio"]["array"]) / data["audio"]["sampling_rate"]
    
    stats_ = {
        "number": len(dataset),
        "labels": label_counter,
        "instructions": ins_counter,
        "total_audio_length": audio_sec,
        "avg_audio_length": audio_sec / len(dataset)
    }
    stats_sampled[dataset_name] = stats_
    
    json.dump(stats_, Path(f"./stats/{dataset_name.split('/')[-1]}_random.json").open("w"), indent=2)
    
    print(len(dataset))
    print("="*100)
    
