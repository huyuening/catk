# 基于冻结 PRE_BC 的 ECoSim 文本控制

本实现仅加入 ECoSim 风格的 `tag_only` 文本条件。基础模型固定为：

```text
/root/workspace/catk/logs/
pre_bc_history_dynamics_hard_ce_b200/
runs/2026-07-30_21-15-08/checkpoints/last.ckpt
```

该基础模型使用历史 11 帧重构的纵向加速度、角速度、横向加速度，并使用 hard-label cross entropy：`spatial_aware_smoothing=false`、`label_smoothing=0.0`。文本微调阶段继续使用相同的 hard CE，不执行普通 CAT-K CLSFT；地图、智能体 token、历史动力学、六个时空交互块、token head 和未来动力学分支全部冻结。仅训练：

- DistilBERT 最后六层注意力中 `q/k/v/out` 的 rank-16 LoRA；
- DistilBERT 到 256 维的文本投影；
- 六个智能体块之后的 FiLM。

配置中的 `action=finetune` 仅表示“从 PRE_BC 权重初始化一个全新训练任务”。它不会恢复 optimizer、scheduler、AMP scaler、epoch、global step、callback 或 W&B run ID。

## 1. 构建 ECoSim 标签

标签必须分别从训练集和验证集动作 CSV 构建，不能读取测试集未来轨迹。每行同时产生一个方向标签和一个纵向标签：

- 方向：`LeftTurn`、`RightTurn`、`LeftLaneChange`、`RightLaneChange` 或 `Straight`；
- 纵向：`Accelerate`、`Decelerate`、`Stopping`、`KeepSpeed` 或 `Parked`；
- `U_TURN` 在 V1 中不被错误归类为直行；
- 仅使用当前帧后的 80 帧，输出半开区间；相邻片段按 ECoSim 规则合并，过短片段被过滤。

示例：

```bash
cd /root/workspace/catk
source /root/anaconda3/etc/profile.d/conda.sh
conda activate trajtok

export TEXT_PROMPT_ROOT=/mnt/pfs/waymo_motion_1_3_0/text_control_tags
# 默认 80 个进程；若 PFS 小文件写入拥塞，可降低到 16 或 32
export TAG_WORKERS=80
export TAG_PROGRESS_EVERY=1000

ACTION_ROWS=/path/train-actions-000.csv.gz,/path/train-actions-001.csv.gz \
TEXT_SPLIT=train \
TEXT_MAPPING_PATH="$TEXT_PROMPT_ROOT/train_scenario_mapping.json" \
bash scripts/build_text_control_tags.sh

ACTION_ROWS=/path/val-actions-000.csv.gz \
TEXT_SPLIT=val \
TEXT_MAPPING_PATH="$TEXT_PROMPT_ROOT/val_scenario_mapping.json" \
bash scripts/build_text_control_tags.sh
```

gzip 输入仍按顺序读取；`TAG_WORKERS` 进程并行计算各场景标签并原子写入 JSON。进度输出到 stderr；将 `TAG_WORKERS=1` 可切换为用于诊断的串行模式。

训练标签只能来自训练集未来。验证集未来标签只用于 oracle validation，不能进入自定义推理。

如果已经生成过标签，只有在 `train_scenario_mapping.json`、`val_scenario_mapping.json` 以及它们引用的训练/验证 tag JSON 均存在时，才可以跳过本节。仅存在两个 mapping 文件但 tag 目录不完整时必须重新生成。

## 2. 训练前只读审计

先解析 Hydra 配置，不启动训练：

```bash
export PRE_BC_CKPT=/root/workspace/catk/logs/pre_bc_history_dynamics_hard_ce_b200/runs/2026-07-30_21-15-08/checkpoints/last.ckpt
export CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact
export TEXT_PROMPT_ROOT=/mnt/pfs/waymo_motion_1_3_0/text_control_tags

python -m src.run experiment=text_control_pre_bc --cfg job --resolve
```

再对真实 checkpoint 做 fail-closed 审计：

```bash
python -m src.smart.inference.audit_text_control "$PRE_BC_CKPT"
```

离线集群先下载 DistilBERT，然后运行：

```bash
export TEXT_MODEL_PATH=/path/to/distilbert-base-uncased
python -m src.smart.inference.audit_text_control \
  "$PRE_BC_CKPT" \
  --local-files-only
```

审计必须满足：所有 missing key 都位于 `encoder.agent_encoder.text_control_adapter.`，unexpected key 为 0，并打印：

```text
history_dynamics_mode: cached_reconstructed
loss_mode: spatial=False, label_smoothing=0.0
CFG disabled
```

输出还包含词表路径、SHA-256、冻结/可训练参数数量和全部可训练张量名称。任何一项不符都不要启动分布式训练。

也可以在模型构建后检查边界：

```python
trainable = [name for name, p in model.named_parameters() if p.requires_grad]
for name in trainable:
    assert any(key in name for key in ("lora_A", "lora_B", "projection", "film_layers")), name
print("\n".join(trainable))
```

## 3. 启动训练

从服务器当前代码开始，完整执行顺序为：

```bash
cd /root/workspace/catk
source /root/anaconda3/etc/profile.d/conda.sh
conda activate trajtok

git pull --ff-only origin main

export PRE_BC_CKPT=/root/workspace/catk/logs/pre_bc_history_dynamics_hard_ce_b200/runs/2026-07-30_21-15-08/checkpoints/last.ckpt
export CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact
export TEXT_PROMPT_ROOT=/mnt/pfs/waymo_motion_1_3_0/text_control_tags

python -m src.run experiment=text_control_pre_bc --cfg job --resolve
python -m src.smart.inference.audit_text_control "$PRE_BC_CKPT"
bash scripts/train_text_control_pre_bc.sh
```

前两条 Python 命令只解析配置和审计权重，不会开始训练。只有审计满足上一节列出的 hard-CE 契约，才执行最后一条训练命令。

```bash
bash scripts/train_text_control_pre_bc.sh
```

默认设置为 8 卡、每卡 batch size 4、10 epoch、K=32 闭环训练、温度 `1e-5`、10% 固定验证切片、每轮完整 Fast WOSAC 2025。脚本会取消 `WANDB_RUN_ID` 和 `WANDB_RESUME`，并设置 `logger.wandb.id=null`、`resume=never`，因此不会接回 PRE_BC 的 W&B run。

如果 DistilBERT 已保存在本地：

```bash
TEXT_MODEL_PATH=/path/to/distilbert-base-uncased \
bash scripts/train_text_control_pre_bc.sh \
  model.model_config.text_control.local_files_only=true
```

## 4. 单智能体反事实推理

先可视化场景当前帧，确定要控制的真实 agent ID。推理入口只读取场景 pickle 和已训练文本模型 checkpoint；它复制场景，仅保留前 11 帧信息，清空隐藏未来轨迹/朝向/速度/动力学，并用当前帧有效性构造 rollout mask。它不会读取 tag、mapping、validation GT 或 test GT。

```bash
CHECKPOINT=/path/to/text-control/last.ckpt
SCENARIO=/path/to/preprocessed_scenario/validation/SCENARIO_ID.pkl

# 左变道
bash scripts/infer_text_control.sh \
  "$CHECKPOINT" "$SCENARIO" 12345 \
  "The target vehicle is changing lanes left." \
  outputs/left_lane_change 32 7

# 加速
bash scripts/infer_text_control.sh \
  "$CHECKPOINT" "$SCENARIO" 12345 \
  "The target vehicle is accelerating." \
  outputs/accelerate 32 7

# 减速停车
bash scripts/infer_text_control.sh \
  "$CHECKPOINT" "$SCENARIO" 12345 \
  "The target vehicle is decelerating and slowing down to a stop." \
  outputs/brake 32 7

# 安全关键探索：直行加速通过红灯
bash scripts/infer_text_control.sh \
  "$CHECKPOINT" "$SCENARIO" 12345 \
  "The target vehicle accelerates straight through the red light." \
  outputs/red_light_probe 32 7
```

输出包括：

- `rollouts.pt`：全部智能体的 `pred_traj_10hz`、`pred_z_10hz`、`pred_head_10hz`、`pred_idx`、agent ID 和 seed；
- `request.json`：checkpoint、scenario、目标 ID、文本和 rollout 数；
- `rollout.mp4`：场景含有效 TFRecord 路径且 Waymo renderer 可用时生成；否则保留完整张量并给出警告。

推理只接受 `topk_prob`。`topk_prob_sampled_with_dist` 和 `topk_dist_sampled_with_prob` 会使用未来 GT 距离，因此被明确拒绝。文本只编码一次，并被 32 个条件 rollout 复用；不会再运行一遍无条件分支。

## 5. 控制能力边界

这是软文本条件，不是硬动作执行器。训练标签中的左/右转、左/右变道、加速、减速、停车和匀速具有直接监督；“闯红灯”“必须碰撞”并不是当前 V1 标签，因此自由文本可能产生分布外响应，但不能保证执行。若安全关键测试要求无论碰撞风险都必须变道或闯灯，仍需后续加入 hard token forcing、轨迹约束或 guidance 优化层，不能把本实现的成功个例当作硬保证。

FiLM 只直接作用于被选中的 agent。AUTO 智能体不会被自己的 FiLM 修改，但下一层 60 m agent-to-agent attention 能看到受控车改变后的状态，因此可以跟车、制动、避让或发生碰撞响应。

文本模式故意不使用 CFG。ECoSim 文本设置对应 `omega=1.0`，直接使用条件 logits；增加第二次无条件 forward 只会把 32 次解码变成 64 次，且不符合当前协议。

## 6. 评估协议

必须分开报告两条评估轨道：

1. **Oracle validation**：允许用验证集未来动作生成文本，仅用于测量 action adherence 和 realism；不得把它作为部署推理输入。
2. **Counterfactual inference**：仅使用历史 11 帧、用户文本和目标 agent ID，不加载任何未来标签或 GT。

建议分别报告：

- PRE_BC 与文本模型在无 prompt 下的逐位回归；
- 条件 Fast WOSAC 2025 realism、kinematic、interactive、map-based 指标；
- 动作成功率和目标车轨迹偏移；
- 碰撞、近碰撞、驶出道路和闯红灯率，并按碰撞类型分层；
- 不同 seed 的多样性；
- prompt 改写鲁棒性和目标 ID 错误拒绝率。

WOSAC 主要衡量场景真实性，不能单独证明模型遵循语言。动作成功率必须独立报告，尤其是闯红灯、强制变道和碰撞等分布外安全关键指令。
