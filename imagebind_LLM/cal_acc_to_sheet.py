from pathlib import Path
import json

all_datasets = """BirdSoundDetection_Warblrb10k
ChordClassification_AcousticGuitarAndPiano
EnvironmentalSoundClassification_AnimalsESC50
EnvironmentalSoundClassification_ExteriorAndUrbanNoisesESC50
EnvironmentalSoundClassification_HumanAndNonSpeechSoundsESC50
EnvironmentalSoundClassification_InteriorAndDomesticSoundsESC50
EnvironmentalSoundClassification_NaturalSoundscapesAndWaterSoundsESC50
SpeechDetection_LJSpeech
SpeechDetection_LibriSpeechTestClean
SpeechDetection_LibriSpeechTestOther
SpeechTextMatching_LJSpeech
SpeechTextMatching_LibriSpeechTestClean
SpeechTextMatching_LibriSpeechTestOther
SpokenTermDetection_LJSpeech
SpokenTermDetection_LibriSpeechTestClean
SpokenTermDetection_LibriSpeechTestOther
SpeechCommandRecognition_GoogleSpeechCommandsV1
EnhancementDetection_LibrittsTestCleanWham
NoiseDetectiongaussian_LJSpeechMusan
NoiseDetectiongaussian_VCTKMusan
NoiseDetectionmusic_LJSpeechMusan
NoiseDetectionmusic_VCTKMusan
NoiseDetectionnoise_LJSpeechMusan
NoiseDetectionnoise_VCTKMusan
NoiseDetectionspeech_LJSpeechMusan
NoiseDetectionspeech_VCTKMusan
NoiseSNRLevelPredictiongaussian_VCTKMusan
NoiseSNRLevelPredictionmusic_VCTKMusan
NoiseSNRLevelPredictionnoise_VCTKMusan
NoiseSNRLevelPredictionspeech_VCTKMusan
ReverberationDetectionlargeroom_LJSpeechRirsNoises
ReverberationDetectionlargeroom_VCTKRirsNoises
ReverberationDetectionmediumroom_LJSpeechRirsNoises
ReverberationDetectionmediumroom_VCTKRirsNoises
ReverberationDetectionsmallroom_LJSpeechRirsNoises
ReverberationDetectionsmallroom_VCTKRirsNoises
AccentClassification_AccentdbExtended
DialogueEmotionClassification_DailyTalk
EmotionRecognition_MultimodalEmotionlinesDataset
HowFarAreYou_3DSpeaker
StressDetection_MIRSD
SpoofDetection_ASVspoof2015
SpoofDetection_ASVspoof2017
DialogueActClassification_DailyTalk
Intent_Classification_FluentSpeechCommands_Action
Intent_Classification_FluentSpeechCommands_Location
Intent_Classification_FluentSpeechCommands_Object
SarcasmDetection_Mustard
SpeakerCounting_LibriTTSTestClean""".split("\n")

from pathlib import Path
import json


def cal_accuracy(predictions, training_instructions=None):
    seen_count = 0
    unseen_count = 0
    seen_correct = 0
    unseen_correct = 0
    for pred in predictions:
        if "The answer" in pred["instruction"]:
            ins = pred["instruction"].split("The answer")[0].strip()
        else:
            ins = pred["instruction"]

        if training_instructions is not None:
            if ins in training_instructions:
                seen_count += 1
                if pred["label"] == pred["pred"]:
                    seen_correct += 1
            else:
                unseen_count += 1
                if pred["label"] == pred["pred"]:
                    unseen_correct += 1
        else:
            unseen_count += 1
            if pred["label"] == pred["pred"]:
                unseen_correct += 1
    total_count = (seen_count + unseen_count)
    total_correct = (seen_correct + unseen_correct)
    acc = {
        "seen_accuracy": seen_correct / seen_count if seen_count != 0 else None,
        "unseen_accuracy": unseen_correct / unseen_count if unseen_count != 0 else None,
        "total_accuracy":  total_correct / total_count,

        "total_count": total_count,
        "seen_count": seen_count,
        "unseen_count": unseen_count,

        "total_correct": total_correct,
        "seen_correct": seen_correct,
        "unseen_correct": unseen_correct,

        "unseen_ratio": unseen_count / total_count
    }
    acc = {k: f"{v:.4f}" if v is not None else "" for k,v in acc.items()}
    return acc

train_data_path = Path("/work/u8915687/big-superb/big-superb-train-data")
task2path = {}
for d in train_data_path.iterdir():
    task2path[d.stem.split("_")[0].lower()] = d
    
exp_path = Path("/home/u8915687/lab/big-superb/LLaMA-Adapter/imagebind_LLM/exp/new_train2/results/full")
# In exp_path: 
# - task_A.json
# - task_B.json
# - task_C.json

# in task_X.json
# {
#   predictions: list of dict
#                 [{"pred": "true", 
#                  "label":"true", 
#                  "instruction": "Determine..."
#                 }, ...]
# }

for dataset_name in all_datasets:
    task_data_name = dataset_name
    task_name = task_data_name.split("_")[0].lower()

    row = [task_data_name]
    result_file = (exp_path/f"{task_data_name}.json")
    if (result_file.exists() and
        not task_name.startswith("how")
       ):
        results = json.load(result_file.open())
        if task2path.get(task_name):
            metadata_file = task2path.get(task_name) / "train/metadata.json"
            metadata = json.load(metadata_file.open())

            # gather training instructions and remove label prompts.
            training_instructions = set([v["instruction"].split("The answer")[0].strip() for v in metadata.values()])

            acc = cal_accuracy(results["predictions"], training_instructions)
            row += [acc["total_accuracy"], acc["seen_accuracy"], acc["unseen_accuracy"]]
        else:
            acc = cal_accuracy(results["predictions"])
            row += [acc["total_accuracy"], "", acc["unseen_accuracy"]]
        
    else:
        row += ["", "", ""]
    print(";".join(row))
