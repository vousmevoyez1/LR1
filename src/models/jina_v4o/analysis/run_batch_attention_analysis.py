import os
import json
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# 导入你的分析器组件
from attention_analyzer import (
    AttentionExtractor,
    compute_all_metrics,
    plot_attention_heatmap,
    plot_metrics_summary,
)

# ============================================================================
# 聚合级别的可视化函数
# ============================================================================
def plot_subtask_aggregate_metrics(
    aggregated_data: dict,
    save_dir: str,
    title_suffix: str = ""
):
    """绘制所有子任务的总体平均注意力指标（熵和JSD）对比图。"""
    os.makedirs(save_dir, exist_ok=True)
    tasks = list(aggregated_data.keys())
    
    if not tasks:
        print("No valid data to plot aggregates.")
        return

    colors = sns.color_palette("tab10", n_colors=len(tasks))
    
    # 1. 各子任务平均注意力熵趋势 (Line Plot)
    plt.figure(figsize=(10, 6))
    for i, task in enumerate(tasks):
        entropies = aggregated_data[task]["mean_entropies"]
        K = len(entropies)
        x_ticks = list(range(1, K + 1))
        plt.plot(x_ticks, entropies, marker='o', linewidth=2, 
                 color=colors[i], label=task)

    plt.xlabel("Thought Token Index", fontsize=12)
    plt.ylabel("Average Attention Entropy (nats)", fontsize=12)
    plt.title(f"Average Attention Entropy per Subtask {title_suffix}", fontsize=14, pad=15)
    plt.xticks(x_ticks, [f"T{i}" for i in x_ticks])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    entropy_save_path = os.path.join(save_dir, "aggregate_entropy_trend.png")
    plt.savefig(entropy_save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved aggregate entropy plot to: {entropy_save_path}")

    # 2. 各子任务平均注意力转移 JSD (Grouped Bar Chart)
    first_task = tasks[0]
    num_jsd_steps = len(aggregated_data[first_task]["mean_jsds"])
    
    if num_jsd_steps > 0:
        plt.figure(figsize=(12, 6))
        x = np.arange(num_jsd_steps) 
        width = 0.8 / len(tasks)
        
        for i, task in enumerate(tasks):
            jsds = aggregated_data[task]["mean_jsds"]
            offset = x - 0.4 + (i + 0.5) * width
            plt.bar(offset, jsds, width, label=task, color=colors[i], alpha=0.85)

        plt.xlabel("Adjacent Thought Token Pair", fontsize=12)
        plt.ylabel("Average Jensen-Shannon Divergence", fontsize=12)
        plt.title(f"Average Attention Shift (JSD) per Subtask {title_suffix}", fontsize=14, pad=15)
        
        step_labels = [f"T{i+1}→T{i+2}" for i in range(num_jsd_steps)]
        plt.xticks(x, step_labels)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        jsd_save_path = os.path.join(save_dir, "aggregate_jsd_shift.png")
        plt.savefig(jsd_save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved aggregate JSD plot to: {jsd_save_path}")

# ============================================================================
# 主运行逻辑
# ============================================================================
def main():
    # 路径配置
    input_json_path = "/data/LR1/src/models/jina_v4o/analysis/output/selected_cases_all_single_mt5_recall_mt0_miss.json"
    output_base_dir = "/data/LR1/src/models/jina_v4o/analysis/output/attention_results"
    
    # ================================================================
    # 1. 占位：加载模型和处理器 (请替换为你自己的模型加载代码)
    # ================================================================
    print("Loading model and processor...")
    # TODO: 在这里写入加载你的 JinaV4o 模型和处理器的代码
    # from your_model_module import load_model, load_processor
    # model = load_model(...)
    # processor = load_processor(...)
    
    # 下面两行是为了让脚本通过语法检查，实际运行时必须替换为真实的对象
    model = None 
    processor = None 
    
    # 如果 model 为空，则无法继续，这里做个安全拦截
    if model is None or processor is None:
        print("⚠️ 警告: 请先在脚本中补充模型(model)和处理器(processor)的加载代码！")
        # return  # 填好模型加载代码后，将此行注释掉

    num_thought_tokens = 5
    extractor = AttentionExtractor(
        model=model, 
        processor=processor, 
        num_thought_tokens=num_thought_tokens, 
        device="cuda:0"
    )

    # ================================================================
    # 2. 读取数据并按任务分组
    # ================================================================
    print(f"Reading cases from: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cases = data.get('cases', [])

    cases_by_task = defaultdict(list)
    for case in cases:
        cases_by_task[case['dataset']].append(case)

    # ================================================================
    # 3. 逐任务处理案例
    # ================================================================
    aggregated_data = {}

    for task, task_cases in cases_by_task.items():
        print(f"\n[{task}] Processing {len(task_cases)} cases...")
        
        task_entropies = []
        task_jsds = []
        
        # 为当前任务创建独立的保存文件夹（用于保存第一条数据的图表）
        task_out_dir = os.path.join(output_base_dir, task)
        os.makedirs(task_out_dir, exist_ok=True)

        for i, case in enumerate(task_cases):
            qid = case['qid']
            query_txt = case.get('query_txt')
            query_img_path = case.get('query_img_path')

            try:
                # 抽取注意力矩阵
                result = extractor.extract(
                    query_text=query_txt,
                    query_image_path=query_img_path
                )

                if result.thought_to_input_attn is None:
                    print(f"  Skipping {qid}: No thought-to-input attention available.")
                    continue

                # 计算指标
                metrics = compute_all_metrics(result.thought_to_input_attn)
                task_entropies.append(metrics['entropy_per_thought'])
                
                if metrics['jsd_between_adjacent']:
                    task_jsds.append(metrics['jsd_between_adjacent'])

                # 🌟【需求点】只为每个任务的第一条有效数据生成单例图表
                if len(task_entropies) == 1: 
                    print(f"  -> Generating plots for first valid case: {qid}")
                    plot_attention_heatmap(
                        result,
                        save_path=os.path.join(task_out_dir, f"{qid}_heatmap.png"),
                        title=f"[{task}] First Query - Heatmap"
                    )
                    plot_metrics_summary(
                        metrics,
                        save_path=os.path.join(task_out_dir, f"{qid}_metrics.png"),
                        title=f"[{task}] First Query - Metrics"
                    )

            except Exception as e:
                print(f"  Error processing {qid}: {str(e)}")

        # ================================================================
        # 4. 计算该任务的平均值
        # ================================================================
        if task_entropies:
            # axis=0 表示将所有样本在 T1, T2.. 的位置上分别求平均
            mean_entropies = np.mean(task_entropies, axis=0).tolist()
            mean_jsds = np.mean(task_jsds, axis=0).tolist() if task_jsds else []
            
            aggregated_data[task] = {
                "mean_entropies": mean_entropies,
                "mean_jsds": mean_jsds
            }
            print(f"[{task}] Done. Valid cases: {len(task_entropies)}")

    # ================================================================
    # 5. 生成所有任务的全局聚合对比图
    # ================================================================
    print("\nGenerating aggregate metrics plots...")
    plot_subtask_aggregate_metrics(
        aggregated_data=aggregated_data,
        save_dir=output_base_dir,
        title_suffix="(MT5 Recall Cases)"
    )
    print(f"\n🎉 All done! Results saved in: {output_base_dir}")

if __name__ == "__main__":
    main()