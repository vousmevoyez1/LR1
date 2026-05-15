# Standard library
import base64
import io
import json
import os
import random
from enum import Enum
from typing import Callable, List, Union, Any

# Third-party
import torch
from PIL import Image

from collections import defaultdict
from torch.utils.data import Dataset
### typechecked 用于运行时类型检查（collator 构造时更稳）
from typeguard import typechecked

# Project files
from data.preprocessing.utils import (
    format_string,
    hash_did,
    hash_qid,
    get_mbeir_task_id,
)


class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"


class MBEIRDatasetBase(Dataset):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        img_preprocess_fn,
    ):
        """
        Initialize the MBEIRDataset.

        Args:
        - datapath (str): Path to the data.
        - img_preprocess_fn (function): Image preprocessing function.
        - mbeir_data_dir (str): Root directory of the MBEIR dataset.
        - training (bool): Indicator if the dataset is for training.
        """
        self.mbeir_data_dir = mbeir_data_dir
        self.img_preprocess_fn = img_preprocess_fn or (lambda x: x)

    def __len__(self):
        raise NotImplementedError("This method should be implemented in derived classes.")

    ### JSONL 读取：按行解析为 Python 对象列表。
    def _load_data_jsonl(self, datapath):
        data_entries = []
        with open(datapath, "r") as fin:
            for line in fin:
                data_entry = json.loads(line)
                data_entries.append(data_entry)
        return data_entries

    def _load_data(self, data_path):
        """Validate and load data."""
        full_data_path = os.path.join(self.mbeir_data_dir, data_path)
        assert os.path.exists(full_data_path), f"Data Path {full_data_path} does not exist"
        assert full_data_path.endswith(".jsonl"), f"Data Path {full_data_path} is not a jsonl file"
        data_entries = self._load_data_jsonl(full_data_path)
        return data_entries

    def _load_query_data(self, query_data_path):
        self.query_data = self._load_data(query_data_path)

    def _load_cand_pool(self, cand_pool_data_path):
        self.cand_pool = self._load_data(cand_pool_data_path)

    def _load_query_instructions(self, instructions_path):
        """Validate and load instructions."""
        full_instructions_path = os.path.join(self.mbeir_data_dir, instructions_path)
        # Validate the path and file extension
        assert os.path.exists(full_instructions_path), f"Instructions Path {full_instructions_path} does not exist"
        assert full_instructions_path.endswith(".tsv"), f"Instructions Path {full_instructions_path} is not a tsv file"
        prompts_dict = {}
        ### •	加载 TSV 指令模板（跳过首行 header）：
	    ###     •	键："{dataset_id}, {query_modality}, {cand_modality}"（注意下标 [3], [0], [1] 的列顺序；TSV 模板必须匹配）。
	    ###     •	值：从第 5 列起的所有非空指令候选列表。
        with open(full_instructions_path, "r") as f:
            next(f)  # Skip the header line
            for line in f.readlines():
                parts = line.strip().split("\t")
                ### query_modality	cand_modality	dataset_name	dataset_id	prompt_1	prompt_2	prompt_3	prompt_4
                # Construct the key to be dataset_id, query_modality, cand_modality
                key = f"{parts[3]}, {parts[0]}, {parts[1]}"
                prompts = [p for p in parts[4:] if p]  # Filters out any empty prompts
                prompts_dict[key] = prompts
        self.query_instructions = prompts_dict

    def _load_and_preprocess_image(self, query_img_path):
        """Load an image given a path"""
        if not query_img_path:
            return None
        ### 因为重新分了数据集，所以这里self.mbeir_data_dir要变一下
        # full_query_img_path = os.path.join(self.mbeir_data_dir, query_img_path)
        full_query_img_path = os.path.join("/data/M-BEIR/", query_img_path)
        assert os.path.exists(full_query_img_path), f"Image Path {full_query_img_path} does not exist"
        image = Image.open(full_query_img_path).convert("RGB")
        image = self.img_preprocess_fn(image)
        return image

    def _get_random_query_prompt(self, dataset_id, query_modality, cand_modality):
        key = f"{dataset_id}, {query_modality}, {cand_modality}"
        prompts = self.query_instructions.get(key, [])
        assert prompts, f"Cannot find prompts for {key}"
        prompt = format_string(random.choice(prompts))
        assert prompt, f"Prompt is empty for {key}"
        return prompt

    def _get_target_modality_from_task_id(self, task_id, fallback_modality=None):
        """Map task_id to target modality string.

        Rules:
        - task_id = 0 -> image
        - task_id = 2 -> image,text
        - task_id = 3 or 6 -> text
        """
        mapping = {
            0: "image",
            2: "image,text",
            3: "text",
            6: "text",
        }

        parsed_task_id = None
        try:
            if task_id is not None:
                parsed_task_id = int(task_id)
        except (TypeError, ValueError):
            parsed_task_id = None

        if parsed_task_id in mapping:
            return mapping[parsed_task_id]
        return fallback_modality or ""

    def _build_query_text_with_prompt(self, query_prompt, query_txt, query_modality, task_id, fallback_target_modality):
        """Build query prompt with retrieval intent + modality fields."""
        target_modality = self._get_target_modality_from_task_id(task_id, fallback_modality=fallback_target_modality)

        if query_txt != "":
            return format_string(
                f"Retrieval Intent: {query_prompt}\n"
                f"modality:{query_modality}\n"
                f"target modality:{target_modality}\n"
                f"Query: {query_txt}"
            )

        query_text = format_string(
            f"Retrieval Intent: {query_prompt}\n"
            f"modality:{query_modality}\n"
            f"target modality:{target_modality}"
        )
        return query_text + "\nQuery: "

    def __getitem__(self, index):
        raise NotImplementedError("This method should be implemented in derived classes.")


class MBEIRMainDataset(MBEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        query_data_path,  # Relate path to the query data
        cand_pool_path,  # Relate path to the candidate pool data
        query_instruct_path,  # Relate path to the query instructions
        img_preprocess_fn,
        mode=Mode.TRAIN,
        enable_query_instruct=True,  # Whether to enable instructions
        shuffle_cand=True,  # Whether to shuffle the candidates
        hard_neg_num=0,  # Number of negative examples in the batch
        returns=None,  # Catch any return-related settings
        print_config=True,  # Whether to print the dataset config
    ):
        ### 继承基类并加载三类资源：查询数据、候选池（字典形态）、指令模板
        super().__init__(mbeir_data_dir, img_preprocess_fn)

        self._load_query_data(query_data_path)
        self._load_cand_pool_as_dict(cand_pool_path)
        self._load_query_instructions(query_instruct_path)

        self.mode = mode
        self.shuffle_cand = shuffle_cand
        self.select_cand = self._get_random_cand if self.shuffle_cand else self._get_first_cand
        self.enable_query_instruct = enable_query_instruct
        self.hard_neg_num = hard_neg_num

        returns = {} if returns is None else returns
        ### 返回字段控制：用传入的 returns 覆盖默认（YAML 中可设置）。
        self.returns = {
            "hashed_qid": True,  # default value
            "task_id": False,  # default value
            "hashed_p_did": False,  # default value
            **returns,  # Overwrite defaults with any values provided in returns
        }
        if print_config:
            self.query_data_path = query_data_path
            self.cand_pool_path = cand_pool_path
            self.query_instruct_path = query_instruct_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Dataset Config---")
        print(f"Mode: {self.mode}")
        print(f"Query Data Path: {self.query_data_path}")
        print(f"Candidate Pool Path: {self.cand_pool_path}")
        print(f"Enable Query Instructions: {self.enable_query_instruct}")
        if self.enable_query_instruct:
            print(f"Query Instructions Path: {self.query_instruct_path}")
        print(f"Shuffle Candidates: {self.shuffle_cand}")
        print(f"Hard Negative Number: {self.hard_neg_num}")
        print(f"Returns: {self.returns}")
        print(f"--------------------------\n")

    ### 将候选池列表转为 {did: entry} 的字典，便于 O(1) 查询正/负例。
    def _load_cand_pool_as_dict(self, cand_pool_data_path):
        self._load_cand_pool(cand_pool_data_path)
        cand_pool_dict = {}
        ### cand_pool example: {"txt": null, "img_path": "mbeir_images/fashion200k_images/tops/blouses/56925626/56925626_2.jpg", "modality": "image", "did": "1:2", "src_content": null}
        for cand_pool_entry in self.cand_pool:
            did = cand_pool_entry.get("did")
            assert did, f"Cannot find did for {cand_pool_entry}"
            cand_pool_dict[did] = cand_pool_entry
        self.cand_pool = cand_pool_dict

    def __len__(self):
        return len(self.query_data)

    ### 正样本选择策略（随机 or 首个）。
    def _get_random_cand(self, cand_list):
        return random.choice(cand_list)

    def _get_first_cand(self, cand_list):
        return cand_list[0]

    def __getitem__(self, index):
        """Retrieve an item from the dataset by index."""
        ### jsonl文件中的每一行
        ### example: {"qid": "9:6", "query_txt": null, "query_img_path": "mbeir_images/mscoco_images/val2014/COCO_val2014_000000391895.jpg", "query_modality": "image", "query_src_content": null, "pos_cand_list": ["9:2", "9:3", "9:4", "9:5", "9:6"], "neg_cand_list": [], "task_id": 3}
        mbeir_entry = self.query_data[index]

        ### 读取一条查询：文本、图像路径、查询模态、qid 以及数据集 ID（从 qid 前缀派生）。
        query_txt = mbeir_entry.get("query_txt") or ""
        query_img_path = mbeir_entry.get("query_img_path", None)
        query_modality = mbeir_entry.get("query_modality", None)
        qid = mbeir_entry.get("qid", None)
        task_id = mbeir_entry.get("task_id", None)
        query_dataset_id = qid.split(":")[0] if qid else None

        # Randomly sample a positive example
        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        assert len(pos_cand_list) > 0, f"Cannot find positive candidates for {mbeir_entry}"

        # TODO: Fix this hack for OVEN and INFOSEEK
        # We only choose the one matched with the query dataset_id due to OVEN and INFOSEEK
        ### Hack：某些数据集（OVEN、INFOSEEK）中正例列表可能跨数据集，这里在评估时仅保留与查询同数据集的正例，避免跨域干扰。
        if self.mode == Mode.EVAL:
            pos_cand_list = [
                pos_cand_did for pos_cand_did in pos_cand_list if pos_cand_did.split(":")[0] == query_dataset_id
            ]

        ### 选定一个正例 did 并在候选池 dict 中查出对应条目，取其模态与文本，并规范化文本。
        selected_pos_cand_did = self.select_cand(pos_cand_list)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        assert pos_cand, f"Cannot find positive candidate {selected_pos_cand_did} for {mbeir_entry}"
        # Note: pos_cand_dataset_id should be the same as query_dataset_id but for OVEN and INFOSEEK it is not.
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        pos_cand_modality = pos_cand.get("modality", None)
        pos_cand_txt_raw = pos_cand.get("txt") or ""
        pos_cand_txt = format_string(
            f"modality: {pos_cand_modality}\n"
            f"passage: {pos_cand_txt_raw}"
        )

        # Randomly sample a query prompt
        # Note:query_modality and pos_cand_modality should define the golden modalities of the current mbeir_entry task.
        # neg_cand_modality could be different from pos_cand_modality.
        ### 基于（数据集ID、查询模态、正例模态）键选取一条指令模板并拼接到查询文本前（若启用指令）
        query_prompt = self._get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality)
        ### 检索意图的提示语需要重新构建
        # query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")
        query_txt_with_prompt = self._build_query_text_with_prompt(
            query_prompt=query_prompt,
            query_txt=query_txt,
            query_modality=query_modality,
            task_id=task_id,
            fallback_target_modality=pos_cand_modality,
        )
        query_txt_without_prompt = format_string(query_txt)

        # Sample negative examples
        ### •	训练模式下抽取显式负例：
        ###     •	从查询条目的 neg_cand_list 中抽取 hard_neg_num 个 did，支持循环回绕（%），保证数量满足。
        ###     •	若 shuffle_cand=True 则先随机打散。
        ###     •	查到候选并规范化文本，构成负例列表。
        selected_neg_cand_list = []
        if self.mode == Mode.TRAIN:
            neg_cand_id_list = mbeir_entry.get("neg_cand_list", [])
            if self.hard_neg_num > 0:
                assert len(neg_cand_id_list) > 0, f"Cannot find negative candidates for {mbeir_entry}"
                if self.shuffle_cand:
                    random.shuffle(neg_cand_id_list)
                selected_neg_cand_id_list = []
                for i in range(self.hard_neg_num):
                    selected_neg_cand_id_list.append(
                        neg_cand_id_list[i % len(neg_cand_id_list)]
                    )  # % Wrap around from idx 0.
                for neg_cand_did in selected_neg_cand_id_list:
                    neg_cand = self.cand_pool.get(neg_cand_did, None)
                    neg_cand_modality = neg_cand.get("modality", None)
                    neg_cand_txt_raw = neg_cand.get("txt") or ""
                    neg_cand_txt = format_string(
                        f"modality: {neg_cand_modality}\n"
                        f"passage: {neg_cand_txt_raw}"
                    )
                    neg_cand["txt"] = neg_cand_txt
                    selected_neg_cand_list.append(neg_cand)

        ### 小工具：按需加载图像并返回结构化字典。
        def _prepare_data_dict(txt, img_path):
            ### _load_and_preprocess_image里对image的操作
            ### image = Image.open(full_query_img_path).convert("RGB")
            ### image = self.img_preprocess_fn(image)
            img = self._load_and_preprocess_image(img_path)
            return {"txt": txt, "img": img}

        query = _prepare_data_dict(
            (query_txt_with_prompt if self.enable_query_instruct else query_txt_without_prompt),
            query_img_path,
        )
        ### query = {"txt": txt, "img": img}
        instance = {"query": query}

        ### 评估模式下可返回哈希后的 qid 与 task_id，便于对齐/分析。
        if self.mode == Mode.EVAL:
            if self.returns.get("hashed_qid"):
                instance.update({"qid": hash_qid(qid)})
            if self.returns.get("task_id"):
                instance.update({"task_id": get_mbeir_task_id(query_modality, pos_cand_modality)})
            # TODO: add src_content if needed

        ### 训练模式下：
        # •	可返回哈希后的正例 did（用于 in-batch 诊断、指标统计）。
        # •	组装正例与负例条目（含 txt/img）。
        # •	返回完整 instance。
        if self.mode == Mode.TRAIN:
            if self.returns.get("hashed_p_did"):
                instance.update({"p_did": hash_did(selected_pos_cand_did)})

            pos_cand = _prepare_data_dict(
                pos_cand_txt,
                pos_cand.get("img_path", None),
            )
            instance.update({"pos_cand": pos_cand})

            neg_cand_list = [
                _prepare_data_dict(
                    neg_cand["txt"],
                    neg_cand.get("img_path", None),
                )
                for neg_cand in selected_neg_cand_list
            ]
            if len(neg_cand_list) > 0:
                instance.update({"neg_cand_list": neg_cand_list})
        ### 返回的instance里面文本还依旧只是字符串，图像是img_preprocess_fn处理后的。
        ### cand_pool主要在训练中提供正负例，推理时不需要。
        return instance

### 推理专用数据集：MBEIRInferenceOnlyDataset
### •	该类用于在线推理/检索时，仅有查询输入（无需候选池），逻辑与主数据集相似，但无正负例抽样，支持 qid/task_id 返回，且 enable_query_instruct 控制是否拼接 prompt。


class MBEIRInferenceOnlyDataset(MBEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        queries,  # Relate path to the query data
        query_instruct_path,  # Relate path to the query instructions
        img_preprocess_fn,
        enable_query_instruct=True,  # Whether to enable instructions
        returns=None,  # Catch any return-related settings
        print_config=True,  # Whether to print the dataset config
    ):
        super().__init__(mbeir_data_dir, img_preprocess_fn)

        self.query_data = queries
        self._load_query_instructions(query_instruct_path)
        self.enable_query_instruct = enable_query_instruct

        returns = {} if returns is None else returns
        self.returns = {
            "hashed_qid": True,  # default value
            "task_id": False,  # default value
            "hashed_p_did": False,  # default value
            **returns,  # Overwrite defaults with any values provided in returns
        }
        if print_config:
            self.query_instruct_path = query_instruct_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Dataset Config---")
        print(f"Enable Query Instructions: {self.enable_query_instruct}")
        if self.enable_query_instruct:
            print(f"Query Instructions Path: {self.query_instruct_path}")
        print(f"Returns: {self.returns}")
        print(f"--------------------------\n")

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, index):
        """Retrieve an item from the dataset by index."""
        mbeir_entry = self.query_data[index]

        query_txt = mbeir_entry.get("query_txt") or ""
        query_img_path = mbeir_entry.get("query_img_path", None)
        query_modality = mbeir_entry.get("query_modality", None)
        candidate_modality = mbeir_entry.get("candidate_modality", None)
        qid = mbeir_entry.get("qid", None)
        task_id = mbeir_entry.get("task_id", None)
        query_dataset_id = qid.split(":")[0] if qid else None

        # Randomly sample a query prompt
        # Note:query_modality and cand_desired_modality should define the golden modalities of the current mbeir_entry task.
        query_prompt = self._get_random_query_prompt(query_dataset_id, query_modality, candidate_modality)
        ### 检索意图的提示语需要重新构建
        # query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")
        query_txt_with_prompt = self._build_query_text_with_prompt(
            query_prompt=query_prompt,
            query_txt=query_txt,
            query_modality=query_modality,
            task_id=task_id,
            fallback_target_modality=candidate_modality,
        )
        query_txt_without_prompt = format_string(query_txt)

        def _prepare_data_dict(txt, img_path):
            img = self._load_and_preprocess_image(img_path)
            return {"txt": txt, "img": img}

        query = _prepare_data_dict(
            (query_txt_with_prompt if self.enable_query_instruct else query_txt_without_prompt),
            query_img_path,
        )
        instance = {"query": query}

        if self.returns.get("hashed_qid"):
            instance.update({"qid": hash_qid(qid)})
        if self.returns.get("task_id"):
            instance.update({"task_id": get_mbeir_task_id(query_modality, candidate_modality)})

        return instance


class MBEIRCandidatePoolDataset(MBEIRDatasetBase):
    def __init__(
        self,
        mbeir_data_dir,  # Root directory of the MBEIR dataset
        cand_pool_data_path,  # Relate path to the candidate pool data
        img_preprocess_fn,
        returns=None,  # Catch any return-related settings
        print_config=True,  # Whether to print the dataset config
    ):
        super().__init__(mbeir_data_dir, img_preprocess_fn)
        self._load_cand_pool(cand_pool_data_path)

        returns = {} if returns is None else returns
        self.returns = {
            "src_content": False,  # default value
            "hashed_did": True,  # default value for candidate id
            **returns,
        }

        # Print dataset config
        if print_config:
            self.cand_pool_path = cand_pool_data_path
            self._print_config()

    def _print_config(self):
        # Print dataset config
        print(f"\n---Mbeir Candidate Pool Dataset Config---")
        print(f"Candidate Pool Path: {self.cand_pool_path}")
        print(f"Returns: {self.returns}")
        print(f"--------------------------\n")

    def __len__(self):
        return len(self.cand_pool)

    def __getitem__(self, index):
        ### example of self.cand_pool[index]: {"txt": null, "img_path": "mbeir_images/fashion200k_images/tops/blouses/56925626/56925626_2.jpg", "modality": "image", "did": "1:2", "src_content": null}
        mbeir_cand_pool_entry = self.cand_pool[index]
        img_path = mbeir_cand_pool_entry.get("img_path", None)
        img = self._load_and_preprocess_image(img_path)

        did = mbeir_cand_pool_entry.get("did", None)
        dataset_id = did.split(":")[0] if did else None
        cand_txt = mbeir_cand_pool_entry.get("txt") or ""
        cand_modality = mbeir_cand_pool_entry.get("modality", None)
        cand_txt = format_string(
            f"modality: {cand_modality}\n"
            f"passage: {cand_txt}"
        )

        instance = {
            "txt": cand_txt,
            "img": img,
            "modality": cand_modality,
        }
        if self.returns.get("hashed_did"):
            instance.update({"did": hash_did(did)})
        if self.returns.get("src_content"):
            instance.update({"src_content": mbeir_cand_pool_entry.get("src_content", None)})
        return instance


class MBEIRCollatorBase(object):
    ### Collator 基类：接收分词器与图像尺寸，构造统一的padding 图像与空文本（用于缺失模态的填充）。
    @typechecked
    def __init__(self, tokenizer: Callable[[List[str]], Any], image_size: Union[tuple, int]):
        """
        :param tokenizer: The tokenizer function to be used for text.
               It should take in a list of strings and return a corresponding tensor.
               Note: Pre-set properties like max_length, padding, and truncation
               should be configured before passing the tokenizer to this function.
        :param image_size: The size of the image to be used, should set in the config file.
        """
        self.tokenizer = tokenizer
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.H, self.W = image_size
        self.padded_image = torch.zeros((3, self.H, self.W))  # Note: this is a black image
        self.padded_txt = ""  # Note: this is an empty string

    ### 对文本/图像做缺省填充，并返回是否存在的 mask（1=有，0=无）。
    ### 并非是token的mask，而是样本级别的mask，因为query形式可能只有文本或图像其中一种模态。
    def _get_padded_text_with_mask(self, txt):
        return (txt, 1) if txt not in [None, ""] else (self.padded_txt, 0)

    def _get_padded_image_with_mask(self, img):
        return (img, 1) if img is not None else (self.padded_image, 0)

    def __call__(self, batch):
        raise NotImplementedError("This method should be implemented in derived classes.")


class MBEIRMainCollator(MBEIRCollatorBase):
    def __init__(self, tokenizer: Callable[[List[str]], Any], image_size: tuple, mode=Mode.TRAIN):
        super().__init__(tokenizer, image_size)
        self.mode = mode

    ### 考虑嵌入jina V4的需求，重写collator逻辑，process_text,process_image,process_mix的逻辑应该是在这里
    ### ### 关键设计：把 Query/Pos/Neg 的文本与图像扁平化拼成一批，便于模型做统一的批内张量化处理（加速）。
    def __call__(self, batch):
        # Note: I group txt/image from queries and candidates together to form a single tensor.
        # Allowing for efficient GPU-based processing.

        txt_list, txt_mask_list, img_list, img_mask_list = [], [], [], []

        ### index_mapping 记录每个样本（inst_idx）在扁平化后张量中的索引位置，以便模型在前向/损失处能把对应的 query/pos/neg 再映射回去。
        index_mapping = {
            "query": [[] for _ in range(len(batch))],
        }
        instance_keys = ["query"]

        # Handle EVAL mode-specific operations
        ### 评估模式：把 qid/task_id 从 instance 中弹出，单独存入列表（避免干扰后续打包）。
        qid_list, task_id_list = [], []
        if self.mode == Mode.EVAL:
            for instance in batch:
                qid = instance.pop("qid", None)
                task_id = instance.pop("task_id", None)
                if qid is not None:
                    qid_list.append(qid)
                if task_id is not None:
                    task_id_list.append(task_id)

        # Handle TRAIN mode-specific operations
        ### 训练模式：提取哈希的正例 did 列表，并在 index_mapping 中为 pos_cand 与（可选）neg_cand_list 新增索引槽位。
        ### positive的did，但是是hash后的
        p_did_list = []
        if self.mode == Mode.TRAIN:
            for instance in batch:
                p_did = instance.pop("p_did", None)
                if p_did is not None:
                    p_did_list.append(p_did)

            index_mapping.update({"pos_cand": [[] for _ in range(len(batch))]})
            instance_keys.extend(["pos_cand"])

            if "neg_cand_list" in batch[0]:
                index_mapping.update({"neg_cand_list": [[] for _ in range(len(batch))]})
                instance_keys.extend(["neg_cand_list"])

        # Generate Index Mapping
        ### 核心：遍历 batch 中的每个样本/每个子键（query/pos/neg…），将条目扁平化追加，并记录它在扁平批次中的位置。
        counter = 0
        for inst_idx, instance in enumerate(batch):
            ### instance_keys = ["query", "pos_cand"(if any), "neg_cand_list"(if any)]
            for instance_key in instance_keys:
                items = [instance[instance_key]] if instance_key != "neg_cand_list" else instance[instance_key]  # list
                for item in items:
                    txt = item["txt"]
                    img = item["img"]

                    index_mapping[instance_key][inst_idx].append(counter)  # Track current index
                    counter += 1
                    ### txt_mask/img_mask：1=有模态，0=无模态
                    padded_txt, txt_mask = self._get_padded_text_with_mask(txt)
                    padded_img, img_mask = self._get_padded_image_with_mask(img)
                    txt_list.append(padded_txt)
                    img_list.append(padded_img)
                    txt_mask_list.append(txt_mask)
                    img_mask_list.append(img_mask)

        ### ？ 如果是BLIP-FF，这里self.tokenizer(txt_list)不需要考虑image token吗？
        processed_batch = {
            "txt_batched": self.tokenizer(txt_list),
            "image_batched": torch.stack(img_list, dim=0),
            "txt_mask_batched": torch.tensor(txt_mask_list, dtype=torch.long),
            "image_mask_batched": torch.tensor(img_mask_list, dtype=torch.long),
            "index_mapping": index_mapping,
        }

        if self.mode == Mode.EVAL:
            if qid_list:
                processed_batch.update({"qid_list": qid_list})
            if task_id_list:
                processed_batch.update({"task_id_list": task_id_list})

        if self.mode == Mode.TRAIN:
            if p_did_list:
                processed_batch.update({"p_did_list": torch.tensor(p_did_list)})

        # TODO: Fix this hack for BLIP tokenizer.
        if hasattr(processed_batch["txt_batched"], "input_ids"):
            bs = processed_batch["txt_batched"]["input_ids"].size(0)
        else:
            bs = len(processed_batch["txt_batched"])
        assert bs == processed_batch["image_batched"].size(0)
        assert bs == processed_batch["txt_mask_batched"].size(0)
        assert bs == processed_batch["image_mask_batched"].size(0)
        return processed_batch


class MBEIRInferenceOnlyCollator(MBEIRCollatorBase):
    def __init__(self, tokenizer: Callable[[List[str]], Any], image_size: tuple):
        super().__init__(tokenizer, image_size)

    def __call__(self, batch):
        txt_list, txt_mask_list, img_list, img_mask_list = [], [], [], []
        qid_list, task_id_list = [], []
        for instance in batch:
            query = instance["query"]
            txt = query["txt"]
            img = query["img"]
            padded_txt, txt_mask = self._get_padded_text_with_mask(txt)
            padded_img, img_mask = self._get_padded_image_with_mask(img)
            txt_list.append(padded_txt)
            img_list.append(padded_img)
            txt_mask_list.append(txt_mask)
            img_mask_list.append(img_mask)
            qid = instance.pop("qid", None)
            if qid is not None:
                qid_list.append(qid)
            task_id = instance.pop("task_id", None)
            if task_id is not None:
                task_id_list.append(task_id)

        processed_batch = {
            "txt_batched": self.tokenizer(txt_list),
            "image_batched": torch.stack(img_list, dim=0),
            "txt_mask_batched": torch.tensor(txt_mask_list, dtype=torch.long),
            "image_mask_batched": torch.tensor(img_mask_list, dtype=torch.long),
            "qid_list": qid_list,
            "task_id_list": task_id_list,
        }

        if hasattr(processed_batch["txt_batched"], "input_ids"):
            bs = processed_batch["txt_batched"]["input_ids"].size(0)
        else:
            bs = len(processed_batch["txt_batched"])
        assert bs == processed_batch["image_batched"].size(0)
        assert bs == processed_batch["txt_mask_batched"].size(0)
        assert bs == processed_batch["image_mask_batched"].size(0)
        return processed_batch


class MBEIRCandidatePoolCollator(MBEIRCollatorBase):
    def __init__(self, tokenizer: Callable[[List[str]], Any], image_size: tuple):
        super().__init__(tokenizer, image_size)

    def __call__(self, batch):
        txt_list, txt_mask_list, img_list, img_mask_list, did_list = [], [], [], [], []
        # Candidate can be indexed directly from the batch
        for instance in batch:
            txt = instance["txt"]
            img = instance["img"]
            padded_txt, txt_mask = self._get_padded_text_with_mask(txt)
            padded_img, img_mask = self._get_padded_image_with_mask(img)
            txt_list.append(padded_txt)
            img_list.append(padded_img)
            txt_mask_list.append(txt_mask)
            img_mask_list.append(img_mask)

            did = instance.get("did", None)
            if did is not None:
                did_list.append(did)

        processed_batch = {
            "txt_batched": self.tokenizer(txt_list),
            "image_batched": torch.stack(img_list, dim=0),
            "txt_mask_batched": torch.tensor(txt_mask_list, dtype=torch.long),
            "image_mask_batched": torch.tensor(img_mask_list, dtype=torch.long),
        }

        if did_list:
            processed_batch.update({"did_list": did_list})

        if hasattr(processed_batch["txt_batched"], "input_ids"):
            bs = processed_batch["txt_batched"]["input_ids"].size(0)
        else:
            bs = len(processed_batch["txt_batched"])
        assert bs == processed_batch["image_batched"].size(0)
        assert bs == processed_batch["txt_mask_batched"].size(0)
        assert bs == processed_batch["image_mask_batched"].size(0)
        return processed_batch
