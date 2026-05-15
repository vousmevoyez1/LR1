import os
import re
import yaml
import subprocess
import shutil
import time

# ================= 硬件与环境配置 =================

# 1. GPU 数量 (对应 NPROC)
NUM_GPUS = 2
# 设置 Master Port
MASTER_PORT = "29556"
MASTER_ADDR = "127.0.0.1"

# 2. 基础路径
UNIIR_DIR = "/data/LR1-UniIR"
SRC_DIR = os.path.join(UNIIR_DIR, "src")
COMMON_DIR = os.path.join(SRC_DIR, "common")
MBEIR_DATA_DIR = "/data/M-BEIR" 

# 3. 源代码路径
MODELING_FILE = "/data/jina-v4-local-copy/modeling_jina_embeddings_v4.py"
# 注意：这里已经包含了两层 jina_v4t，根据你的反馈保持不变
WRAPPER_FILE = "/data/LR1-UniIR/src/models/jina_v4t/jina_v4t/jina_v4t.py"
CONFIG_UPDATER = os.path.join(COMMON_DIR, "config_updater.py")

# 4. [修复] 原始 YAML 模板路径
# 增加了中间的一层 jina_v4t 目录
CONFIG_BASE_DIR = os.path.join(SRC_DIR, "models/jina_v4t/jina_v4t/configs/large/eval/inbatch")

YAML_TEMPLATES = {
    "embed": os.path.join(CONFIG_BASE_DIR, "embed.yaml"),
    "index": os.path.join(CONFIG_BASE_DIR, "index.yaml"),
    "retrieval": os.path.join(CONFIG_BASE_DIR, "retrieval.yaml")
}

# ================= 任务定义 =================

TASKS = [
    {
        "name": "Task1_RS1_CkptMT1",
        "reason_steps": 1,
        "ckpt_full_path": "/data/LR1-UniIR/checkpoint/jina_v4t/Large/Instruct/InBatch/jina_v4t_latest.pth",
        "is_pretrained": False
    },
    {
        "name": "Task2_RS3_CkptMT3",
        "reason_steps": 3,
        "ckpt_full_path": "/data/LR1-UniIR/checkpointMT3/jina_v4t/Large/Instruct/InBatch/jina_v4t_latest.pth",
        "is_pretrained": False
    } ,   
    {
        "name": "Task3_RS0_CkptMT0",
        "reason_steps": 0,
        "ckpt_full_path": "/data/LR1-UniIR/checkpointMT0/jina_v4t/Large/Instruct/InBatch/jina_v4t_epoch_0.pth",
        "is_pretrained": False
    }
]

# ================= 核心逻辑 =================

def backup_file(filepath):
    bak_path = filepath + ".auto_bak"
    if not os.path.exists(bak_path):
        shutil.copy(filepath, bak_path)
        print(f"📦 Backup created: {bak_path}")

def update_source_code(task):
    """修改 reason_steps 和 default_ckpt (包含缩进修复)"""
    
    # --- 1. 修改 Reason Steps ---
    with open(MODELING_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern_rs = r"(self\.reason_steps\s*=\s*)(\d+)"
    if re.search(pattern_rs, content):
        content = re.sub(pattern_rs, f"\\g<1>{task['reason_steps']}", content)
        with open(MODELING_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"🔧 Set reason_steps = {task['reason_steps']}")
    else:
        print(f"⚠️ Warning: 'self.reason_steps' not found in {MODELING_FILE}")

    # --- 2. 修改 Default Checkpoint ---
    with open(WRAPPER_FILE, 'r', encoding='utf-8') as f:
        wrapper_content = f.read()
    
    if task["ckpt_full_path"] is None:
        new_assignment = "default_ckpt = None"
    else:
        new_assignment = f'default_ckpt = "{task["ckpt_full_path"]}"'
        
    # [缩进修复版正则]
    # ^(\s*) 捕获行首缩进
    pattern_ckpt = r"^(\s*)default_ckpt\s*=.*$"
    
    match = re.search(pattern_ckpt, wrapper_content, flags=re.MULTILINE)
    if match:
        new_content = re.sub(pattern_ckpt, f"\\g<1>{new_assignment}", wrapper_content, flags=re.MULTILINE)
        with open(WRAPPER_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"🔧 Set default_ckpt = {new_assignment}")
    else:
        # 如果还是找不到，打印前500个字符帮助定位
        print(f"❌ Critical Error: Could not find 'default_ckpt =' in {WRAPPER_FILE}")
        raise ValueError("Source code pattern mismatch: Check file path or variable name.")

def generate_configs(task):
    """生成包含新路径的 YAML"""
    # 检查基础路径是否存在
    if not os.path.exists(CONFIG_BASE_DIR):
        raise FileNotFoundError(f"Config dir not found: {CONFIG_BASE_DIR}")

    temp_dir = os.path.join(CONFIG_BASE_DIR, "temp_configs", task["name"])
    os.makedirs(temp_dir, exist_ok=True)
    
    new_paths = {}
    suffix = task["name"]
    
    for kind, path in YAML_TEMPLATES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML template not found: {path}")

        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 修改 CKPT 配置
        # if task["is_pretrained"]:
        #     config["model"]["ckpt_config"]["ckpt_name"] = None
        # else:
        #     config["model"]["ckpt_config"]["ckpt_dir"] = os.path.dirname(task["ckpt_full_path"])
        #     config["model"]["ckpt_config"]["ckpt_name"] = os.path.basename(task["ckpt_full_path"])

        # 修改输出目录
        if "embed_config" in config:
            config["embed_config"]["embed_dir_name"] = f"embed_{suffix}"
        if "index_config" in config:
            config["index_config"]["embed_dir_name"] = f"embed_{suffix}"
            config["index_config"]["index_dir_name"] = f"index_{suffix}"
        if "retrieval_config" in config:
            config["retrieval_config"]["index_dir_name"] = f"index_{suffix}"
            config["retrieval_config"]["results_dir_name"] = f"results_{suffix}"

        out_path = os.path.join(temp_dir, os.path.basename(path))
        with open(out_path, 'w') as f:
            yaml.dump(config, f)
        new_paths[kind] = out_path
        
    return new_paths

def update_yaml_instruct_status(yaml_path):
    print(f"  -> Updating instruct status for {os.path.basename(yaml_path)}")
    cmd = [
        "python", CONFIG_UPDATER,
        "--update_mbeir_yaml_instruct_status",
        "--mbeir_yaml_file_path", yaml_path,
        "--enable_instruct", "True"
    ]
    subprocess.run(cmd, check=True, cwd=COMMON_DIR)

def run_pipeline(yaml_paths):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}:{env.get('PYTHONPATH', '')}"
    env["MASTER_PORT"] = MASTER_PORT
    env["MASTER_ADDR"] = MASTER_ADDR
    
    cwd = COMMON_DIR

    # 1. Embedder
    print("🚀 [Step 1] Running Embedder...")
    update_yaml_instruct_status(yaml_paths["embed"])
    
    cmd_embed = [
        "python", "-m", "torch.distributed.run",
        f"--nproc_per_node={NUM_GPUS}",
        f"--master_port={MASTER_PORT}",
        f"--master_addr={MASTER_ADDR}",
        "/data/LR1-UniIR/src/common/mbeir_embedder.py",
        "--config_path", yaml_paths["embed"],
        "--uniir_dir", UNIIR_DIR,
        "--mbeir_data_dir", MBEIR_DATA_DIR
    ]
    subprocess.run(cmd_embed, check=True, env=env, cwd=cwd)
    
    # 2. Index
    print("🚀 [Step 2] Building Index...")
    update_yaml_instruct_status(yaml_paths["index"])
    
    cmd_index = [
        "python", "mbeir_retriever.py",
        "--config_path", yaml_paths["index"],
        "--uniir_dir", UNIIR_DIR,
        "--mbeir_data_dir", MBEIR_DATA_DIR,
        "--enable_create_index"
    ]
    subprocess.run(cmd_index, check=True, env=env, cwd=cwd)
    
    # 3. Retrieval
    print("🚀 [Step 3] Retrieving...")
    update_yaml_instruct_status(yaml_paths["retrieval"])
    
    cmd_retrieval = [
        "python", "mbeir_retriever.py",
        "--config_path", yaml_paths["retrieval"],
        "--uniir_dir", UNIIR_DIR,
        "--mbeir_data_dir", MBEIR_DATA_DIR,
        "--enable_retrieval"
    ]
    subprocess.run(cmd_retrieval, check=True, env=env, cwd=cwd)

# ================= 执行入口 =================

if __name__ == "__main__":
    # 1. 简单路径检查
    if not os.path.exists(MODELING_FILE):
        print(f"❌ Error: Modeling file not found: {MODELING_FILE}")
        exit(1)
    if not os.path.exists(WRAPPER_FILE):
        print(f"❌ Error: Wrapper file not found: {WRAPPER_FILE}")
        exit(1)

    backup_file(MODELING_FILE)
    backup_file(WRAPPER_FILE)
    
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set (Using All)')}")
    print(f"NUM_GPUS: {NUM_GPUS}")
    
    for task in TASKS:
        print(f"\n{'='*50}")
        print(f">>> Starting Task: {task['name']}")
        print(f"{'='*50}")
        
        try:
            update_source_code(task)
            yaml_paths = generate_configs(task)
            run_pipeline(yaml_paths)
            print(f"✅ Task {task['name']} Completed Successfully.")
            
        except Exception as e:
            print(f"❌ Task {task['name']} Failed: {e}")
            break