
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
from data.dataset import BigSuperbDataset

disable_caching()

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--decode_subset", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--encoder_type", type=str, default="whisper")
    parser.add_argument("--dataset_list", type=str, default="test_dataset.txt")

    parser.add_argument("--llama_path", type=str, default="ckpts/llama")
    parser.add_argument("--data_path", type=str, default="Dataset/big-superb-train-data-renamed")
    
    return parser.parse_args()

@torch.inference_mode()
def forward_inference(self, visual_feats, tokens, start_pos: int, args=None):
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
    
    if args.encoder_type == "whisper":
        visual_proj = visual_feats # B, 1, D
    elif args.encoder_type == "imagebind":
        visual_proj = []
        for i in range(_bsz):
            visual_proj.append(
                visual_feats[i, :, i, :]
            )
        visual_proj = torch.stack(visual_proj)
    else:
        raise NotImplementedError()


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
        cache_weight=0.5,
        args=None
):
    bsz = len(prompts)
    params = self.llama.params
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    with torch.cuda.amp.autocast():
        if args.encoder_type == "whisper":
            visual_query = self.forward_whisper(inputs)
        elif args.encoder_type == "imagebind":
            visual_query = self.forward_visual(inputs, cache_size, cache_t, cache_weight)
        else:
            raise NotImplementedError()

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
            logits = forward_inference(self, visual_query, tokens[:, prev_pos:cur_pos], prev_pos, args=args)
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



def main(args):
    # Config #
    
    
    ROOT_PATH = Path("/home/u2619111/hank/lab/big-superb/LLaMA-Adapter/imagebind_LLM")
    EXP_PATH = ROOT_PATH/args.exp_path # exp/finetune2
    pretrained_path = EXP_PATH/args.model_path # "checkpoint-4.pth"
    is_decode_subset = args.decode_subset # true / false

    RESULT_PATH = EXP_PATH/args.output_dir
    if is_decode_subset:
        RESULT_PATH /= "subset"
    else:
        RESULT_PATH /= "full"


    RESULT_PATH.mkdir(parents=True, exist_ok=True)


    logging.basicConfig(filename=str(RESULT_PATH/"inference.log"), level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.info(f"pretrained_path: {pretrained_path}")
    logging.info(f"result_path: {RESULT_PATH}")
    logging.info(f"is_decode_subset: {is_decode_subset}")

    # Config #

    llama_dir = args.llama_path
    tokenizer = Tokenizer(model_path=f"{llama_dir}tokenizer.model")
    if args.encoder_type == "whisper":
        model = llama.whisper_llama_adapter.load(pretrained_path, llama_dir, knn=True, max_batch_size=16)
    elif args.encoder_type == "imagebind":
        model = llama.llama_adapter.load(pretrained_path, llama_dir, knn=True, max_batch_size=16)
    else:
        raise NotImplementedError()
    model.eval()

    DATA_PATH = Path(args.data_path)
    all_datasets = open(f"data/{args.dataset_list}").read().split("\n")
    # all_datasets = open("data/train_dataset.txt").read().split("\n")
    for task_name in all_datasets:

        logging.info(task_name)
        
        if (RESULT_PATH/f"{task_name}.json").exists():
            logging.info(f"SKIP {task_name}")
            continue
        dataset = BigSuperbDataset(DATA_PATH, tokenizer, audio_input_type=args.encoder_type, used_data_split="test", allowed_datasets=[task_name], phase="test")

        if len(dataset) == 0:
            logging.info("Dataset not found")
            continue
        
        def collate_fn(b):
            batch = {}
            batch["ids"] = [d["id"] for d in b]
            batch["audio"] = torch.stack([d["audio"] for d in b])
            batch["instructions"] = [d["instruction"] for d in b]
            batch["prompts"] = [d["prompt"] for d in b]

                    
            batch["labels"] = [d["label"] for d in b]
            return batch
        
        torch.manual_seed(42)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=args.decode_subset,
            collate_fn=collate_fn,
            num_workers=4,
        )
        
        correct = 0
        count = 0
        predictions = []
        label_count = {}
        pred_count = {}
        for batch in tqdm(data_loader):
            inputs = {
                "audio": [batch["audio"].to("cuda"), 1]
            }
            results = my_generate(
                model,
                inputs,
                batch["prompts"],
                max_gen_len=150,
                temperature=0,
                top_p=0,
                args=args
            )

            
            for id, pred, label, ins, prompt in zip(batch["ids"], results, batch["labels"], batch["instructions"], batch["prompts"]):
                pred = pred.strip()
                pred, label = pred.lower(), label.lower()
                if pred == label:
                    correct += 1
                count += 1
                
                label_count[label] = label_count.get(label, 0) + 1
                pred_count[pred] = pred_count.get(pred, 0) + 1
                predictions.append({
                    "id": id,
                    "pred": pred,
                    "label": label,
                    "instruction": ins,
                    "prompt": prompt
                })

            if is_decode_subset and count >= 500:
                break
        
        logging.info(f"{correct} / {len(predictions)}")

        json.dump(
            {
                "accuracy": correct / len(predictions),
                "pred_count": pred_count,
                "label_count": label_count,
                "predictions": predictions,
            },
            (RESULT_PATH/f"{task_name}.json").open("w"), indent=2
        )
        logging.info("="*100)


if __name__ == "__main__":
    args = get_args_parser()
    main(args=args)