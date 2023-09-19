from pathlib import Path
import json
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument("--decode_300", action="store_true")

    return parser.parse_args()

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


def main(args):
    # training dataset for getting instruction
    train_data_path = Path("/home/u2619111/hank/Dataset/big-superb-train-data-renamed")
    task2path = {}
    for d in train_data_path.iterdir():
        task2path[d.stem.lower()] = d
        task2path[d.stem.split("_")[0].lower()] = d

    # start calculating
    exp_path = Path(args.result_path)
    all_datasets = open("data/test_dataset.txt").read().split("\n")

    data_stats = {}
    for dataset_name in all_datasets:
        task_data_name = dataset_name
        task_name = task_data_name.split("_")[0].lower()

        row = [task_data_name]
        result_file = (exp_path/f"{task_data_name}.json")
        if (result_file.exists()):
            results = json.load(result_file.open())
            if task2path.get(task_name) or task2path.get(task_data_name.lower()):
                if task2path.get(task_data_name.lower()):
                    # in training set
                    metadata_file = task2path.get(task_data_name.lower()) / "train/metadata.json"
                    metadata = json.load(metadata_file.open())
                else:
                    metadata_file = task2path.get(task_name) / "train/metadata.json"
                    metadata = json.load(metadata_file.open())

                # gather training instructions and remove label prompts.
                training_instructions = set([v["instruction"].split("The answer")[0].strip() for v in metadata.values()])

                acc = cal_accuracy(results["predictions"], training_instructions)
                row += [acc["total_accuracy"], acc["seen_accuracy"], acc["unseen_accuracy"]]
            else:
                acc = cal_accuracy(results["predictions"])
                row += [acc["total_accuracy"], "", acc["unseen_accuracy"]]
            
            data_stats[task_data_name] = {
                "seen_total": int(float(acc["seen_count"])),
                "unseen_total": int(float(acc["unseen_count"]))
            }
        else:
            row += ["", "", ""]
        print(";".join(row))


if __name__ == "__main__":
    args = get_args_parser()
    main(args=args)