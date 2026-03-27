import warnings

warnings.filterwarnings("ignore")
import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
from utils import *
import sys
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# /ssd/hust-tangxi/workspace/MOR/rq/models/qwen3_vl_embedding.py
from models.qwen3_vl_embedding import Qwen3VLEmbedder
from rich_logger import logger


def load_data(args):
    if args.root:
        logger.info("args.root: ", args.root)
    item2feature_path = os.path.join(args.root, f"{args.dataset}.item.json")
    item2feature = load_json(item2feature_path)
    return item2feature


def generate_textimg(item2feature):
    documents = []

    for item in tqdm(item2feature, ncols=100, desc=f"generate_textimg"):
        data = item2feature[item]
        document = {"text": [], "image": []}
        for meta_key in ["title", "description"]:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                cleaned = meta_value.strip()
                if cleaned != "":
                    document["text"].append(cleaned)
        document["image"] = data["images"]
        if len(document["text"]) == 0:
            document["text"] = ["unknown item"]
        document["text"] = " ".join(document["text"])

        try:
            item_id = int(item)
        except:
            item_id = item

        documents.append((item_id, document))

    return documents


def preprocess_textimg(args):
    logger.info("Process text data: ")
    logger.info("Dataset: ", args.dataset)
    item2feature = load_data(args)
    documents = generate_textimg(item2feature)
    return documents


def format_input_to_conversation(
    input_dict: Dict[str, Any], instruction: str = "Represent the user's input."
) -> List[Dict]:

    text = input_dict.get("text")
    images = input_dict.get("image")

    content = [{"type": "image", "image": image} for image in images if image]
    content.append({"type": "text", "text": text})

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]
    return conversation


def generate_item_embedding(
    args,
    documents,
    model,
    tokenizer=None,
    batch_size=1024,
):
    _, all_documents = zip(*documents)
    batch_size = 8
    embeddings = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(all_documents), batch_size), ncols=100, desc="Embedding"
        ):
            batch_documents = list(all_documents[i : i + batch_size])
            # logger.info(f"batch_documents = {json.dumps(batch_documents, indent=4)}")
            if args.plm_name in ["Qwen3-VL-Embedding-2B"]:
                _embeddings = model.process(batch_documents).cpu().numpy()
            else:
                batch_texts = [doc["text"] for doc in batch_documents]
                encoded_sentences = tokenizer(
                    batch_texts,
                    max_length=args.max_sent_len,
                    truncation=True,
                    return_tensors="pt",
                    padding=True,
                )
                input_ids = encoded_sentences.input_ids.cuda()
                attention_mask = encoded_sentences.attention_mask.cuda()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # outputs.last_hidden_state: [batch, seq, dim]
                last_hidden = outputs.last_hidden_state
                # [batch, seq] -> [batch, seq, 1]
                mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                )
                sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_output = sum_embeddings / sum_mask  # [batch, dim]
                _embeddings = mean_output.cpu().numpy()

            for emb in _embeddings:
                embeddings.append(emb)

    final_embeddings = np.stack(embeddings, axis=0)
    logger.info("Final Embeddings shape: ", final_embeddings.shape)
    file_path = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-td.npy")
    np.save(file_path, final_embeddings)
    logger.info(f"Saved to {file_path}")


def load_model(model_name, model_path):
    logger.info("Loading Qwen Model:", model_path)
    if model_name in ["Qwen3-VL-Embedding-2B"]:
        tokenizer = None
        model = Qwen3VLEmbedder(model_name_or_path=model_path, max_length=16384)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="Beauty", help="Beauty / Sports / Toys"
    )
    parser.add_argument("--root", type=str, default="")
    # parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument("--plm_name", type=str, default="qwen")
    parser.add_argument(
        "--plm_checkpoint", type=str, default="xxx", help="Qwen model path"
    )
    parser.add_argument("--max_sent_len", type=int, default=2048)
    parser.add_argument(
        "--word_drop_ratio", type=float, default=-1, help="word drop ratio"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    documents = preprocess_textimg(args)

    plm_model, plm_tokenizer = load_model(args.plm_name, args.plm_checkpoint)

    generate_item_embedding(
        args,
        documents,
        plm_model,
        plm_tokenizer,
        batch_size=args.batch_size,
    )
