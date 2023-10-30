# Multimodal-LLaMA

This model is based on [ImageBind-LLM](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/imagebind_LLM) from [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter).

## Download dataset

You can use the Preprocessing Scripts in [dynamic-superb/api/preprocess](https://github.com/dynamic-superb/dynamic-superb/tree/main/api/preprocess)

```
/path/to/datasets
├── instance1
│   ├── instance1_001.wav
│   ├── instance1_002.wav
│   ├── instance1_003.wav
│   └── metadata.json
├── instance2
└── instance3
```

## Setup
- Python environment
```
conda create --name multimodal-llama python=3.10
conda activate multimodal-llama
pip install -r requirements.txt
```

- Download the model weight
  - Download LLaMA-1 7B from Meta.
  - Download multimodal-llama checkpoints from huggingface:
    - [whisper-llama-latest.pth](https://huggingface.co/DynamicSuperb/multimodal-llama/resolve/main/whisper-llama-latest.pth)
    - [imagebind-llama-latest.pth](https://huggingface.co/DynamicSuperb/multimodal-llama/resolve/main/imagebind-llama-latest.pth)

- Organize the downloaded file in the following structure:
```
ckpts/
├── llama_model_weights/
│   ├── 7B
│   │   ├── checklist.chk
│   │   ├── consolidated.00.pth
│   │   └── params.json
│   └── tokenizer.model
└── dynamic-superb/
    ├── imagebind-llama-latest.pth
    └── whisper-llama-latest.pth
```

## Inference

- `model_path`: fine-tuned checkpoint
- `encoder_type`: `whisper` / `imagebind`
- `dataset_list`: list of instance names to inference

```shell
python dynamicsuperb_inference.py \
--model_path ckpts/dynamic-superb/whisper-llama-latest.pth \
--encoder_type whisper \
--output_dir results/whisper \
--dataset_list data/test_dataset.txt \
--data_path /path/to/dataset \
--llama_path ckpts/llama_model_weights
```

### Result: 

```
results/whisper
├── AccentClassification_AccentdbExtended.json
├── BirdSoundDetection_Warblrb10k.json
├── ChordClassification_AcousticGuitarAndPiano.json
├── DialogueActClassification_DailyTalk.json
├── DialogueActPairing_DailyTalk.json
...
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