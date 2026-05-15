#!/usr/bin/env python3
"""
Quick validation script to check if Qwen3 integration is properly set up.
"""

import sys
import os
# 在 validate_setup.py 的开头添加
import sys
import os

# 动态添加路径，确保脚本能找到自己所在的包
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../models/qwen3
src_root = os.path.abspath(os.path.join(current_dir, "../../")) # .../src
qwen_root = "/data/Qwen3-VL-Embedding/src"

if src_root not in sys.path:
    sys.path.insert(0, src_root)
if qwen_root not in sys.path:
    sys.path.insert(0, qwen_root)
def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def main():
    print("=" * 80)
    print("Qwen3-VL Integration Validation")
    print("=" * 80)

    all_ok = True

    # Check source files
    print("\n1. Checking source files...")
    files_to_check = [
        ("/data/LR1/src/models/qwen3/__init__.py", "Package init"),
        ("/data/LR1/src/models/qwen3/engine.py", "Engine module"),
        ("/data/LR1/src/models/qwen3/utils.py", "Utils module"),
        ("/data/LR1/src/models/qwen3/qwen3_wrapper.py", "Wrapper class"),
        ("/data/LR1/src/models/qwen3/test_qwen3_wrapper.py", "Test script"),
        ("/data/LR1/src/models/qwen3/README.md", "Documentation"),
    ]

    for filepath, desc in files_to_check:
        all_ok &= check_file_exists(filepath, desc)

    # Check config files
    print("\n2. Checking config files...")
    config_files = [
        ("/data/LR1/src/models/qwen3/qwen3/configs/2b/eval/inbatch/embedood.yaml", "Embed config"),
        ("/data/LR1/src/models/qwen3/qwen3/configs/2b/eval/inbatch/indexood.yaml", "Index config"),
        ("/data/LR1/src/models/qwen3/qwen3/configs/2b/eval/inbatch/retrievalood.yaml", "Retrieval config"),
        ("/data/LR1/src/models/qwen3/qwen3/configs/2b/eval/inbatch/eval_qwen3_vl.sh", "Eval script"),
    ]

    for filepath, desc in config_files:
        all_ok &= check_file_exists(filepath, desc)

    # Check if eval script is executable
    print("\n3. Checking script permissions...")
    eval_script = "/data/LR1/src/models/qwen3/qwen3/configs/2b/eval/inbatch/eval_qwen3_vl.sh"
    if os.access(eval_script, os.X_OK):
        print(f"✓ Eval script is executable")
    else:
        print(f"✗ Eval script is NOT executable")
        all_ok = False

    # Check Qwen3-VL-Embedding repository
    print("\n4. Checking Qwen3-VL-Embedding repository...")
    qwen_repo = "/data/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py"
    all_ok &= check_file_exists(qwen_repo, "Qwen3-VL-Embedding model")

    # Check if utils.py has been updated
    print("\n5. Checking common/utils.py integration...")
    utils_path = "/data/LR1/src/common/utils.py"
    if os.path.exists(utils_path):
        with open(utils_path, 'r') as f:
            content = f.read()
            if 'Qwen3VLEmbedder' in content or 'Qwen3VLWrapper' in content:
                print(f"✓ common/utils.py has Qwen3 integration")
            else:
                print(f"✗ common/utils.py missing Qwen3 integration")
                all_ok = False
    else:
        print(f"✗ common/utils.py NOT FOUND")
        all_ok = False

    # Try importing modules
    # 6. Testing imports...
    print("\n6. Testing imports...")
    
    # 必须清除可能导致冲突的旧路径，并添加正确的父目录
    project_root = '/data/LR1/src'
    qwen_root = '/data/Qwen3-VL-Embedding/src'
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if qwen_root not in sys.path:
        sys.path.insert(1, qwen_root)

    try:
        # 现在 Python 会在 /data/LR1/src 下找到 models/qwen3
        from models.qwen3 import utils
        print("✓ Can import qwen3.utils")
    except Exception as e:
        print(f"✗ Cannot import qwen3.utils: {e}")
        all_ok = False

    try:
        from models.qwen3.qwen3_wrapper import Qwen3VLWrapper
        print("✓ Can import Qwen3VLWrapper")
    except Exception as e:
        print(f"✗ Cannot import Qwen3VLWrapper: {e}")
        all_ok = False

    # Summary
    print("\n" + "=" * 80)
    if all_ok:
        print("✓ All validation checks passed!")
        print("\nNext steps:")
        print("1. Run test script: python /data/LR1/src/models/qwen3/test_qwen3_wrapper.py")
        print("2. Run evaluation: bash /data/LR1/src/models/qwen3/qwen3/configs/2b/eval/inbatch/eval_qwen3_vl.sh")
    else:
        print("✗ Some validation checks failed. Please review the errors above.")
    print("=" * 80)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
