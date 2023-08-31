from datasets import load_dataset, Dataset, disable_caching, load_from_disk
from pathlib import Path
import logging

disable_caching()

def main():

    logging.basicConfig(filename="test_download_full.log", level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    data_path = Path("/work/u8915687/big-superb")

    # training datasets
    # all_datasets = ['BigSuperbPrivate/SpoofDetection_Asvspoof2017', 'BigSuperbPrivate/DailyTalk_DialogueActClassification', 'BigSuperbPrivate/PronounciationEvaluationProsodic_Speechocean762', 'BigSuperbPrivate/PronounciationEvaluationFluency_Speechocean762', 'BigSuperbPrivate/PronounciationEvaluationOverall_Speechocean762', 'BigSuperbPrivate/PronounciationEvaluationAccuracy_Speechocean762', 'BigSuperbPrivate/HowFarAreYou_DeeplyParentChildVocalInteraction', 'BigSuperbPrivate/HowFarAreYou_KoreanReadSpeechCorpus', 'BigSuperbPrivate/SpeakerVerification_Tedlium2Train', 'BigSuperbPrivate/SpeechDetection_Aishell1Train', 'BigSuperbPrivate/SpeakerVerification_LibrispeechTrainClean100', 'BigSuperbPrivate/SpeakerVerification_Aishell1Train', 'BigSuperbPrivate/SpeechDetection_Voxceleb1Train', 'BigSuperbPrivate/SpeakerVerification_Voxceleb1Train', 'BigSuperbPrivate/SpokenTermDetection_Tedlium2Train', 'BigSuperbPrivate/NoiseSNRLevelPredictionSpeech_VoxcelebMusan', 'BigSuperbPrivate/SpeechDetection_LibrispeechTrainClean100', 'BigSuperbPrivate/NoiseSNRLevelPredictionNoise_VoxcelebMusan', 'BigSuperbPrivate/SpeechDetection_Tedlium2Train', 'BigSuperbPrivate/EnhancementDetection_LibrittsTrainClean360Wham', 'BigSuperbPrivate/SpeakerCounting_LibrittsTrainClean100', 'BigSuperbPrivate/NoiseSNRLevelPredictionGaussian_VoxcelebMusan', 'BigSuperbPrivate/NoiseSNRLevelPredictionMusic_VoxcelebMusan', 'BigSuperbPrivate/SpeechTextMatching_Tedlium2Train', 'BigSuperbPrivate/ReverberationDetectionSmallRoom_VoxcelebRirsNoises', 'BigSuperbPrivate/ReverberationDetectionMediumRoom_VoxcelebRirsNoises', 'BigSuperbPrivate/SpokenTermDetection_LibrispeechTrainClean100', 'BigSuperbPrivate/ReverberationDetectionLargeRoom_VoxcelebRirsNoises', 'BigSuperbPrivate/NoiseDetectionSpeech_VoxcelebMusan', 'BigSuperbPrivate/NoiseDetectionNoise_VoxcelebMusan', 'BigSuperbPrivate/NoiseDetectionMusic_VoxcelebMusan', 'BigSuperbPrivate/NoiseDetectionGaussian_VoxcelebMusan', 'BigSuperbPrivate/SpoofDetection_ASVspoof2015', 'BigSuperbPrivate/SpeechTextMatching_LibrispeechTrainClean100']

    # testing datasets
    all_datasets = ['SpeechBigBench/SpoofDetection_ASVspoof2015', 'SpeechBigBench/NoiseSNRLevelPredictionspeech_VCTKMusan', 'SpeechBigBench/NoiseSNRLevelPredictionnoise_VCTKMusan', 'SpeechBigBench/NoiseSNRLevelPredictionmusic_VCTKMusan', 'SpeechBigBench/NoiseSNRLevelPredictiongaussian_VCTKMusan', 'SpeechBigBench/NoiseDetectionspeech_VCTKMusan', 'SpeechBigBench/NoiseDetectionnoise_VCTKMusan', 'SpeechBigBench/NoiseDetectionmusic_VCTKMusan', 'SpeechBigBench/NoiseDetectiongaussian_VCTKMusan', 'SpeechBigBench/Intent_Classification_FluentSpeechCommands_Location', 'SpeechBigBench/Intent_Classification_FluentSpeechCommands_Object', 'SpeechBigBench/Intent_Classification_FluentSpeechCommands_Action', 'SpeechBigBench/DialogueEmotionClassification_DailyTalk', 'SpeechBigBench/DialogueActClassification_DailyTalk', 'SpeechBigBench/NoiseSNRLevelPredictionspeech_VCTKMusan_full', 'SpeechBigBench/NoiseSNRLevelPredictionnoise_VCTKMusan_full', 'SpeechBigBench/NoiseSNRLevelPredictionmusic_VCTKMusan_full', 'SpeechBigBench/NoiseSNRLevelPredictiongaussian_VCTKMusan_full', 'SpeechBigBench/SpeechTextMatching_LibriSpeechTestClean', 'SpeechBigBench/SpoofDetection_ASVspoof2017', 'SpeechBigBench/SpoofDetection_ASVspoof2015_full', 'SpeechBigBench/EnhancementDetection_LibrittsTestCleanWham', 'SpeechBigBench/SpeakerCounting_LibriTTSTestClean', 'SpeechBigBench/SpeechCommandRecognition_GoogleSpeechCommandsV1', 'SpeechBigBench/EmotionRecognition_MultimodalEmotionlinesDataset', 'SpeechBigBench/AccentClassification_AccentdbExtended', 'SpeechBigBench/IntentClassification_FluentSpeechCommands', 'SpeechBigBench/SarcasmDetection_Mustard', 'SpeechBigBench/PronounciationEvaluationProsodic_Speechocean762', 'SpeechBigBench/PronounciationEvaluationFluency_Speechocean762', 'SpeechBigBench/PronounciationEvaluationOverall_Speechocean762', 'SpeechBigBench/PronounciationEvaluationAccuracy_Speechocean762', 'SpeechBigBench/HowFarAreYou_3DSpeaker', 'SpeechBigBench/ReverberationDetectionsmallroom_VCTKRirsNoises', 'SpeechBigBench/ReverberationDetectionlargeroom_VCTKRirsNoises', 'SpeechBigBench/ReverberationDetectionmediumroom_VCTKRirsNoises', 'SpeechBigBench/ReverberationDetectionmediumroom_LJSpeechRirsNoises', 'SpeechBigBench/ReverberationDetectionsmallroom_LJSpeechRirsNoises', 'SpeechBigBench/NoiseDetectionspeech_VCTKMusan_full', 'SpeechBigBench/ReverberationDetectionlargeroom_LJSpeechRirsNoises', 'SpeechBigBench/NoiseDetectionmusic_VCTKMusan_full', 'SpeechBigBench/NoiseDetectionnoise_VCTKMusan_full', 'SpeechBigBench/NoiseDetectionspeech_LJSpeechMusan', 'SpeechBigBench/NoiseDetectiongaussian_VCTKMusan_full', 'SpeechBigBench/NoiseDetectionmusic_LJSpeechMusan', 'SpeechBigBench/NoiseDetectionnoise_LJSpeechMusan', 'SpeechBigBench/NoiseDetectiongaussian_LJSpeechMusan', 'SpeechBigBench/SpokenTermDetection_LibriSpeechTestOther', 'SpeechBigBench/SpokenTermDetection_LibriSpeechTestClean', 'SpeechBigBench/SpokenTermDetection_LJSpeech', 'SpeechBigBench/BirdSoundDetection_Warblrb10k', 'SpeechBigBench/ChordClassification_AcousticGuitarAndPiano', 'SpeechBigBench/SpeechTextMatching_LJSpeech', 'SpeechBigBench/StressDetection_MIRSD', 'SpeechBigBench/EnvironmentalSoundClassification_AnimalsESC50', 'SpeechBigBench/EnvironmentalSoundClassification_NaturalSoundscapesAndWaterSoundsESC50', 'SpeechBigBench/EnvironmentalSoundClassification_HumanAndNonSpeechSoundsESC50', 'SpeechBigBench/EnvironmentalSoundClassification_InteriorAndDomesticSoundsESC50', 'SpeechBigBench/EnvironmentalSoundClassification_ExteriorAndUrbanNoisesESC50', 'SpeechBigBench/SpeechDetection_LJSpeech', 'SpeechBigBench/SpeechDetection_LibriSpeechTestClean', 'SpeechBigBench/SpeechDetection_LibriSpeechTestOther', 'SpeechBigBench/SpeechTextMatching_LibriSpeechTestOther', 'SpeechBigBench/Nonverbal_Vocalization', 'SpeechBigBench/Korean_Read_Speech_Corpus', 'SpeechBigBench/Deeply_Parent_Child_Vocal_Interaction', 'SpeechBigBench/LanguageIdentification_VoxForge', 'SpeechBigBench/speech_commands', 'SpeechBigBench/arabic_speech_corpus']


    for dataset_name in all_datasets:
        if not dataset_name.endswith("_full"):
            continue
        else:
            dataset_name = dataset_name.replace("_full", "")
        logging.info(f"==> Start {dataset_name}")

        # Download dataset to disk
        if (data_path/dataset_name).exists():
            logging.info(f"==> SKIP {dataset_name}", )
            dataset = load_from_disk(data_path/dataset_name)
        else:
            logging.info(f"==> Not yet downloaded {dataset_name}")
            dataset = load_dataset(dataset_name)

            logging.info(f"==> Save original {dataset_name}")
            (data_path/dataset_name).mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(data_path/dataset_name)

        # Build subset
        # logging.info(f"==> Subset {dataset_name}")
        # if dataset.get("test"):
        #     dataset = dataset["test"].shuffle(seed=42)
        #     dataset = Dataset.from_dict(dataset[:subset_size])
        #     logging.info(len(dataset))
        # else:
        #     logging.warning(f"No test in {dataset_name}")
            
        logging.info("="*100)

if __name__ == "__main__":
    main()