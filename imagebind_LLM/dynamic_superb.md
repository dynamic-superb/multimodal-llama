# Dynamic-SUPERB

## Download dataset

- You can use `rsync` to copy datasets from server.

```
Dataset
├── big-superb-test-data-renamed
│   ├── AccentClassification_AccentdbExtended
│   ├── BirdSoundDetection_Warblrb10k
│   ├── ChordClassification_AcousticGuitarAndPiano
│   ├── ...
└── big-superb-train-data-renamed
    ├── DialogueActClassification_DailyTalk
    ├── DialogueActPairing_DailyTalk
    ├── DialogueEmotionClassification_DailyTalk
    ├── EnhancementDetection_LibrittsTrainClean360Wham
    ├── NoiseDetectionGaussian_VoxcelebMusan
    ├── NoiseSNRLevelPredictionGaussian_VoxcelebMusan
    ...
```

## Install

- Follow the installation of Imagebind-LLM and LLaMA-Adapter
- `pip install -r requirements_bigsuperb.txt`
    - This is the virtual environment in the ICASSP paper.
- Try to run finetune script and fix some environmental bugs.


## Finetune & inference scripts

- See: `bigsuperb_finetune.py`
    - `encoder_type`: `whisper`/`imagebind`
- See: `bigsuperb_inference.py`


```shell
conda activate imagebind_LLM

. exp_scripts/bigsuperb_finetune_imagebind.sh 
# . exp_scripts/bigsuperb_finetune_whisper.sh

```

### Inference from a checkpoint

```shell
python bigsuperb_inference.py --exp_path "$OUTPUT_DIR" --model_path "checkpoint-latest.pth" --output_dir results --encoder_type whisper --dataset_list test_dataset.txt
```

```
exp/imagebind_newdata/imagebind1/results3/full
├── AccentClassification_AccentdbExtended.json
├── BirdSoundDetection_Warblrb10k.json
├── ChordClassification_AcousticGuitarAndPiano.json
├── DialogueActClassification_DailyTalk.json
├── DialogueActPairing_DailyTalk.json
```

```json
{
  "accuracy": 0.8973282442748092,
  "pred_count": { "no": 1291, "yes": 1329 },
  "label_count": { "no": 1306, "yes": 1314 },
  "predictions": [
    {
      "id": "SpeechDetection_LibriSpeechTestClean_test_1995-1837-0019_reversed.flac",
      "pred": "no",
      "label": "no",
      "instruction": "Analyze the audio and determine whether it consists of real speech or not. The answer could be yes or no."
    }, ...
  ]
}
```

## Calculate accuracy and format for google sheet

- See: `cal_acc_to_sheet.py`
- `train_data_path` is for calculating seen/unseen accuracy

```shell
python cal_acc_to_sheet.py --result_path exp/whisper_newdata/whisper1/results3/full --train_data_path /home/u2619111/hank/Dataset/big-superb-train-data-renamed
```

Will look like this:
```
BirdSoundDetection_Warblrb10k;0.1467;;0.1467
ChordClassification_AcousticGuitarAndPiano;0.5844;;0.5844
EnvironmentalSoundClassification_AnimalsESC50;0.1175;;0.1175
EnvironmentalSoundClassification_ExteriorAndUrbanNoisesESC50;0.0350;;0.0350
EnvironmentalSoundClassification_HumanAndNonSpeechSoundsESC50;0.0600;;0.0600
EnvironmentalSoundClassification_InteriorAndDomesticSoundsESC50;0.0775;;0.0775
EnvironmentalSoundClassification_NaturalSoundscapesAndWaterSoundsESC50;0.0925;;0.0925
SpeechDetection_LJSpeech;0.9999;1.0000;0.9997
...
```