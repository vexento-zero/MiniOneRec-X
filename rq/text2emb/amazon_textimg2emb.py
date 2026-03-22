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
from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image


def load_data(args):
    if args.root:
        print("args.root: ", args.root)
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
    print("Process text data: ")
    print("Dataset: ", args.dataset)
    item2feature = load_data(args)
    documents = generate_textimg(item2feature)
    return documents


def format_input_to_conversation(
    input_dict: Dict[str, Any], instruction: str = "Represent the user's input."
) -> List[Dict]:

    text = input_dict.get("text")
    images = input_dict.get("image")

    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": text})

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]
    return conversation


def prepare_vllm_inputs(
    input_dict: Dict[str, Any], llm, instruction: str = "Represent the user's input."
) -> Dict[str, Any]:
    image = input_dict.get("image")

    conversation = format_input_to_conversation(input_dict, instruction)

    prompt_text = llm.llm_engine.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    multi_modal_data = None
    if image:
        if isinstance(image, str):
            if image.startswith(("http", "https", "oss")):
                try:
                    image_obj = fetch_image(image)
                    multi_modal_data = {"image": image_obj}
                except Exception as e:
                    print(f"Warning: Failed to fetch image {image}: {e}")
            else:
                abs_image_path = os.path.abspath(image)
                if os.path.exists(abs_image_path):
                    from PIL import Image

                    image_obj = Image.open(abs_image_path)
                    multi_modal_data = {"image": image_obj}
                else:
                    print(f"Warning: Image file not found: {abs_image_path}")
        else:
            multi_modal_data = {"image": image}

    result = {"prompt": prompt_text, "multi_modal_data": multi_modal_data}
    return result


def generate_item_embedding(
    args,
    documents,
    model,
    batch_size=1024,
):
    _, all_documents = zip(*documents)
    local_results = []
    batch_size = 2
    embeddings = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(documents), batch_size),
            ncols=100,
            desc=f"generate embeddings",
        ):
            vllm_inputs = [
                prepare_vllm_inputs(inp, model)
                for inp in all_documents[i : i + batch_size]
            ]
            outputs = model.embed(vllm_inputs)
            _embeddings = [output.outputs.embedding for output in outputs]
            logger.info(f"_embeddings = {_embeddings[0].shape}")
            embeddings.append(_embeddings)
            exit(0)
            batch_texts = list(local_texts[i : i + batch_size])
            batch_ids = local_ids[i : i + batch_size]
            for idx, output in zip(batch_ids, outputs):
                local_results.append((idx, output.outputs.embedding))
            pbar.update(len(batch_texts))

    final_embeddings = np.stack(embeddings, axis=0)
    print("Final Embeddings shape: ", final_embeddings.shape)
    file_path = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-td.npy")
    np.save(file_path, final_embeddings)
    print(f"Saved to {file_path}")


def load_qwen_model(model_path):
    print("Loading Qwen Model:", model_path)
    # model = Qwen3VLEmbedder(model_name_or_path=model_path, max_length=16384)

    engine_args = EngineArgs(
        model=model_path,
        runner="pooling",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=16384,  # 设置最大序列长度
    )
    llm = LLM(**vars(engine_args))
    return llm


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

    plm_model = load_qwen_model(args.plm_checkpoint)

    generate_item_embedding(
        args,
        documents,
        plm_model,
        batch_size=args.batch_size,
    )
