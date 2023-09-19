# Dynamic-SUPERB

## Download dataset

- you can use `rsync` to copy datasets from server.

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

- follow the installation of Imagebind-LLM and LLaMA-Adapter
- `pip install -r requirements_bigsuperb.txt`
    - this is the virtual environment for ICASSP.
- try to run finetune script and fix some environmental bugs.


## Fine-tune & inference scripts

- See: `bigsuperb_finetune.py`
    - `encoder_type`: `whisper`/`imagebind`
- See: `bigsuperb_inference.py`


```shell
conda activate imagebind_LLM

. exp_scripts/bigsuperb_finetune_imagebind.sh # or . exp_scripts/bigsuperb_finetune_whisper.sh

```

### Inference from a checkpoint

```shell
python bigsuperb_inference.py --exp_path "$OUTPUT_DIR" --model_path "checkpoint-latest.pth" --output_dir results --encoder_type whisper --dataset_list test_dataset.txt
```

## paste to google sheet

- See: `cal_acc_to_sheet.py`
- `train_data_path` is for calculating seen/unseen accuracy

```shell
python cal_acc_to_sheet.py --result_path exp/whisper_newdata/whisper1/results3/full --train_data_path /home/u2619111/hank/Dataset/big-superb-train-data-renamed
```