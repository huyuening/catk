# WOMD 全数据集标注与可视化

CatK 内置了一份独立的 WOMD 道路、智能体尺寸代理和逐帧动作标注管线。
运行时不需要再保留 `WOMD-Traffic-Signal-Data-Improvement` 仓库，也不修改
CatK 的训练缓存、词表或模型输入。

## 标注内容

- 每帧主车地图标签：路段/路口、路口控制类型、车道数、路口支路数、
  高速主线/匝道、城市道路和停车场等；
- 第 11 帧智能体尺寸代理：大/小型车辆、摩托车、自行车/电助力车、
  成人/儿童行人；
- 每个有效帧的智能体动作：停车、掉头、左/右转、左/右变道、减速、
  保持和加速；
- 带地图、智能体框和文字标签的单场景 PNG；
- 汇总道路层级、智能体类型和动作分布的 PNG、PDF、SVG 与计数 CSV。

尺寸类别是根据 WOMD 包围盒和速度得到的观测代理，不等价于法规意义上的
车型、年龄或动力类型。

## 输入目录

`--input-root` 下应包含三个原始、未压缩的 TFRecord 目录：

```text
/path/to/uncompressed/scenario/
  training/
  validation/
  testing/
```

文件名只需包含 `tfrecord`，分片会按文件名稳定排序。TFRecord 由轻量流式
读取器直接解析，不需要 TensorFlow 数据管线。

## 一条命令运行

下面的命令会对 training、validation、testing 全量标注和统计，分别生成
100 张单场景示例图，并使用每个 split 的全部结果绘制汇总图：

```bash
python -m src.womd_labeling.run_dataset \
  --input-root /mnt/pfs/waymo_motion_1_3_0/uncompressed/scenario \
  --output-root /mnt/pfs/waymo_motion_1_3_0/catk_womd_labels \
  --splits training validation testing \
  --workers 24
```

若还要为每个场景都输出一张 PNG，增加：

```bash
--visualize-max-scenarios 0
```

这会产生数十万张图，磁盘占用和运行时间会远大于标注本身。汇总图无论该
参数为何值，始终使用整个 split 的标注和统计结果。

也可以使用包装脚本：

```bash
WOMD_ROOT=/mnt/pfs/waymo_motion_1_3_0/uncompressed/scenario \
LABEL_OUTPUT_ROOT=/mnt/pfs/waymo_motion_1_3_0/catk_womd_labels \
NUM_WORKERS=24 \
VISUALIZE_MAX_SCENARIOS=100 \
bash scripts/label_womd_dataset.sh
```

默认启用恢复模式。每个完成的标注分片会校验 gzip、schema、源文件名、
场景索引和记录数后再跳过；损坏或不兼容的结果不会被静默复用。统计和
汇总图完整时也会复用。使用 `--overwrite`（脚本方式为
`OVERWRITE=true`）可强制重算。

## 分阶段运行

只做全量标注：

```bash
python -m src.womd_labeling.run_dataset \
  --input-root /path/to/uncompressed/scenario \
  --output-root /path/to/catk_womd_labels \
  --stages annotations \
  --workers 24
```

在已有标注和统计上重画场景图与汇总图：

```bash
python -m src.womd_labeling.run_dataset \
  --input-root /path/to/uncompressed/scenario \
  --output-root /path/to/catk_womd_labels \
  --stages scenario-visualizations aggregate-visualization \
  --visualize-max-scenarios 100 \
  --workers 8 \
  --overwrite
```

各阶段也可直接调用：

```bash
python -m src.womd_labeling.annotate --help
python -m src.womd_labeling.statistics --help
python -m src.womd_labeling.visualize --help
python -m src.womd_labeling.plot_statistics --help
```

## 输出目录

```text
<output-root>/
  annotations/<split>/
    *.map-annotations.jsonl.gz
    summary.json
  statistics/<split>/
    current_frame_road_types.csv.gz
    current_frame_agent_sizes.csv.gz
    agent_actions_by_frame.csv.gz
    *_counts.csv
    errors.jsonl
    summary.json
  visualizations/scenarios/<split>/
    *.png
    manifest.csv
    summary.json
  visualizations/aggregate/
    <split>.png
    <split>.pdf
    <split>.svg
    <split>_counts.csv
  run_summary.json
```

`run_summary.json` 在每个 split 完成后原子更新，并记录输入、阶段结果、
输出位置与错误数。大文件先写入 `.partial`，完成后才原子改名。

## 单场景调试

先标注一个场景：

```bash
python -m src.womd_labeling.annotate \
  --input-path /path/to/training \
  --output-dir /tmp/catk_womd_annotations \
  --max-scenarios 1 \
  --workers 1
```

再绘图：

```bash
python -m src.womd_labeling.visualize \
  --input-path /path/to/training \
  --annotation-path /tmp/catk_womd_annotations \
  --output-dir /tmp/catk_womd_visualizations \
  --max-scenarios 1 \
  --workers 1
```

迁移代码的来源与许可见
`src/womd_labeling/LICENSE.WOMD_TRAFFIC_SIGNAL_DATA_IMPROVEMENT.txt`。
