# Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models


<p align="center">
     <img src="docs/catk_banner.png" alt="Closest Among Top-K (CAT-K) rollouts unroll the policy during fine-tuning in a way that visited states remain close to the ground-truth.", width=760px>
     <br/><strong>Closest Among Top-K (CAT-K) Rollouts</strong> unroll the policy during fine-tuning in a way that visited states remain close to the ground-truth (GT). At each time step, CAT-K first takes the top-K most likely action tokens according to the policy, then chooses the one leading to the state closest to the GT. As a result, CAT-K rollouts follow the mode of the GT (e.g., turning left), while random or top-K rollouts can lead to large deviations (e.g., going straight or right). Since the policy is essentially trained to minimize the distance between the rollout states and the GT states, the GT-based supervision remains effective for CAT-K rollouts, but not for random or top-K rollouts.
</p>

> **Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models**            
> [Zhejun Zhang](https://zhejz.github.io/), [Peter Karkus](https://karkus.tilda.ws/), [Maximilian Igl](https://maximilianigl.com/), [Wenhao Ding](https://wenhao.pub/), [Yuxiao Chen](https://research.nvidia.com/labs/avg/author/yuxiao-chen/), [Boris Ivanovic](https://www.borisivanovic.com/) and [Marco Pavone](https://web.stanford.edu/~pavone/index.html).<br/>
> 
> [Project Page](https://zhejz.github.io/catk)<br/>
> [arXiv Paper](https://arxiv.org/abs/2412.05334)

```bibtex
@inproceedings{zhang2025closed,
  title = {Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models},
  author = {Zhang, Zhejun and Karkus, Peter and Igl, Maximilian and Ding, Wenhao and Chen, Yuxiao and Ivanovic, Boris and Pavone, Marco},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025},
}
```

## News & Updates

Apr. 2025
- **Oral at CVPR 2025**: Cheers!
- **Top on the WOSAC Leaderboard 2024**: With the Waymo Challenges 2025 coming up, the WOSAC 2024 leaderboard is now closed and our method remains in the 1st place.

Feb. 2025
- **Paper accepted at CVPR 2025:** Cheers!

- **Model checkpoints for WOSAC:** You can obtain the checkpoints for our WOSAC submission (SMART-tiny-CLSFT) by sending an email to Zhejun (zhejun.zhang94@gmail.com). In accordance with Waymo's terms, you must attach a screenshot showing that you are registered and logged into the [My Submissions](https://waymo.com/open/challenges/submissions) page of the Waymo Open Dataset.

- **SMART-mini and SMART-nano:** SMART-tiny with 7M parameters requires training on 8x A100 for a few days, which may be unaffordable in some cases. To address this, we have added config files for two smaller model, [smart_mini_3M.yaml](configs/model/smart_mini_3M.yaml) and [smart_nano_1M.yaml](configs/model/smart_nano_1M.yaml). Specifically, SMART-nano-1M can be trained on a single A100, but its performance is significantly worse. After pre-training and CAT-K fine-tuning, we achieved an RMM of 0.74 with SMART-nano-1M, which is 0.03 lower than that of SMART-tiny-7M. 

Jan. 2025
- **SoTA performance on WOSAC:** CAT-K is now rank #1 on the [WOSAC leaderboard](https://waymo.com/open/challenges/2024/sim-agents/)! We resolved an issue in the agent token vocabulary, and now our fine-tuned model achieves an RMM of **0.7702**. Even our reproduced SMART-tiny-7M (not published on the leaderboard, trained only for 32 epochs via BC) achieves an RMM of **0.7671**, which is comparable to the current second-place method. Reproducing our results should be straightforward. Give it a try!

- **Issue in the agent token vocabulary:** We discovered that the [agent token vocabulary file](src/smart/tokens/cluster_frame_5_2048_remove_duplicate.pkl) we were using (borrowed from the [SMART repository](https://github.com/rainmaker22/SMART/blob/main/smart/tokens/cluster_frame_5_2048.pkl)) was intended only for sanity checks and not for reproducing optimal performance. To resolve this, we added a [script](src/smart/tokens/traj_clustering.py) and used it to build an [appropriate agent token vocabulary](src/smart/tokens/agent_vocab_555_s2.pkl). Our script is based on the [k-disk clustering script from SMART](https://github.com/rainmaker22/SMART/blob/main/scripts/traj_clstering.py). Thanks to the updated agent tokens, all our traffic simulation models saw a significant performance improvement of approximately +0.0060 RMM!



## Installation
- The easy way to setup the environment is to create a [conda](https://docs.conda.io/en/latest/miniconda.html) environment using the following commands
  ```
  conda create -y -n catk python=3.11.9
  conda activate catk
  conda install -y -c conda-forge ffmpeg=4.3.2
  pip install -r install/requirements.txt
  pip install torch_geometric
  pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
  pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
  ```
- Alternatively, a better way is to use the [Dockerfile](install/Dockerfile) and build your own docker. We found the code runs faster in the docker for some reasons.
- We use [WandB](https://wandb.ai/) for logging. You can register an account for free.
- **Be aware**
  - We use 8 *NVIDIA A100 (80GB)* for training and validation, the training and fine-tuning take a few days, whereas the validation and testing take a few hours.
  - We cannot share pre-trained models according to the [terms](https://waymo.com/open/terms) of the Waymo Open Motion Dataset.


## Dataset preparation
- Download the [Waymo Open Motion Dataset](https://waymo.com/open/download/). We use v1.2.1.
- Use [scripts/cache_womd.sh](scripts/cache_womd.sh) to preprocess the dataset into pickle files to accelerate data loading during the training and evaluation.
- You should pack three datasets: `training`, `validation` and `testing`.

### Vocabulary-only trajectory reconstruction

CatK model inputs and future labels remain the original WOMD/CatK data. The only
model-feature change is that `length/width/height` comes from the last history
frame (`current_time_index`) rather than an average that can include future
frames.

Trajectory reconstruction is isolated to offline vocabulary construction. The
bundled geometric filter reconstructs each training trajectory over all 91
available frames, then splits it into the same 0.5 s local segments consumed by
CatK's K-disk clustering. Because this copy is never used as per-scenario
history input or as a future target, the history/future boundary does not need a
causal two-pass reconstruction. Never substitute the reconstructed
vocabulary-source caches for the normal CatK training, validation, or testing
caches. The bundled filter is distributed under the PolyForm Noncommercial
License 1.0.0 in `src/smart/tokens`.

### Compare raw and reconstructed trajectory vocabularies

`src/smart/tokens/compare_trajectory_token_reconstruction.py` runs a matched
experiment on one WOMD TFRecord shard or a directory of training shards. It
writes agent-only CatK caches for the legacy and reconstructed branches,
extracts the same 0.5 s local trajectory segments used by
`traj_clustering.py`, learns K-disk vocabularies, and exports matched-scale plots
plus smoothness and quantization metrics. Maps are omitted from these comparison
caches because they are unchanged and are not consumed by agent-token
clustering.

```bash
python -m src.smart.tokens.compare_trajectory_token_reconstruction \
  --input-path /path/to/preprocessed_scenario/training \
  --output-dir outputs/trajectory_token_reconstruction_comparison \
  --vocab-output-dir src/smart/tokens \
  --vocab-output-name agent_vocab_reconstructed.pkl \
  --method filter \
  --filter-strength strong \
  --num-clusters 2048 \
  --num-workers 12 \
  --worker-backend process
```

The final CatK-compatible vocabulary is written directly to
`src/smart/tokens/agent_vocab_reconstructed.pkl`. Analysis vocabularies retain
the maximum supported token count per class, while `*_agent_vocab.pkl` files
trim every class to the same count for the existing `TokenProcessor`. Add
`--write-reconstructed-tfrecord` only when a serialized audit copy is useful;
that TFRecord is also vocabulary-only and must not be used as model inputs or
labels. `--reconstruction-root` is optional and only needed for the external
`batch` or `optimizer` methods.

### Optional causal history dynamics

The reconstructed vocabulary sources above remain training-only, offline
artifacts and are never used as model inputs.  A separate optional model branch
can estimate three motion-frame quantities directly from each scenario's first
11 observable 10 Hz states: signed longitudinal acceleration, course angular
speed, and lateral acceleration.  Each of CatK's two history tokens receives
its own feature triplet: frames 0--5 and frames 5--10 are reconstructed with
separate quadratic fits before their endpoint derivatives are calculated.  The
same original CatK caches work for training, validation, and testing without
future leakage or additional preprocessing.
Positive longitudinal acceleration means speeding up along the motion tangent;
positive angular speed/lateral acceleration means turning left.  Low-speed
turning values are zeroed with separate vehicle/pedestrian/cyclist thresholds.

The original CatK architecture remains the default.  Enable the dynamics
ablation with the dedicated experiment:

```bash
MY_EXPERIMENT=pre_bc_history_dynamics \
MY_TASK_NAME=pre_bc_history_dynamics_b200 \
bash scripts/train.sh
```

Continue with `experiment=clsft_history_dynamics` and evaluate a compatible
checkpoint with `experiment=inference_history_dynamics`.  A checkpoint trained
without this branch should be evaluated with the original experiment config.

## Run the code
In the scripts, we provide
- [scripts/train.sh](scripts/train.sh) for training and fine-tuning.
- [scripts/local_val.sh](scripts/local_val.sh) for local validation.
- [scripts/wosac_sub.sh](scripts/wosac_sub.sh) for packing submission files.

The training script runs single-node DDP and defaults to 8 GPUs. Its Conda installation, environment, cache root, GPU count, task name, and W&B mode can be overridden with environment variables. The cache root must contain CatK-preprocessed `training`, `validation`, and `validation_tfrecords_splitted` data. The default experiment configs follow the paper: BC pre-training uses 32 epochs and CAT-32 fine-tuning uses 10 epochs, both with a total batch size of 80 on 8 GPUs. Strict paper reproduction requires a cache generated from WOMD v1.2.1; override `CACHE_ROOT` if the server default points to another dataset version.

```
CONDA_ROOT=/root/anaconda3 \
CONDA_ENV=trajtok \
CACHE_ROOT=/path/to/catk_cache \
NUM_GPUS=8 \
MY_TASK_NAME=pre_bc_b200 \
bash scripts/train.sh
```

Additional Hydra overrides can be appended to the command. For example, after BC pre-training, run CAT-K fine-tuning with:

```
MY_EXPERIMENT=clsft \
MY_TASK_NAME=clsft_catk_b200 \
bash scripts/train.sh ckpt_path=/path/to/pretrained/last.ckpt
```

The local validation config follows the paper's 2% validation protocol (880 scenarios) with inference `K=40`. WOSAC submission generation keeps the leaderboard setting `K=48` and temperature `1.0`. The ego GMM configs likewise follow the paper with 32 BC epochs followed by 5 CAT-3 fine-tuning epochs.

### Fast WOSAC 2025 validation

CatK can use TrajTok's GPU-accelerated WOSAC 2025 evaluator directly from a sibling checkout. By default, the inference config expects CatK at `/root/workspace/catk`, TrajTok at `/root/workspace/TrajTok`, and the CatK cache at `/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario`. It evaluates the complete validation split with 32 rollouts per scenario and inference `K=48`. It first looks for TrajTok-preprocessed ground truth in `${CACHE_ROOT}/validation_gt`; when that directory or an individual scenario is unavailable, it falls back to CatK's per-scenario TFRecord.

Set the checkpoint through the environment for a TrajTok-style command:

```
CATK_CKPT=/path/to/model.ckpt \
python run.py experiment=inference task_name=eval
```

Equivalently, pass the checkpoint as a Hydra override:

```
python run.py \
  experiment=inference \
  task_name=eval \
  ckpt_path=/path/to/model.ckpt
```

The inference config uses all visible GPUs through single-node DDP and logs to W&B offline by default. Common optional overrides are:

```
CACHE_ROOT=/path/to/catk_cache \
TRAJTOK_ROOT=/path/to/TrajTok \
FAST_WOSAC_GT_DIR=/path/to/validation_gt \
python run.py \
  experiment=inference \
  task_name=eval \
  ckpt_path=/path/to/model.ckpt \
  trainer.limit_val_batches=10 \
  logger.wandb.offline=false \
  logger.wandb.entity=YOUR_WANDB_ENTITY
```

TrajTok Fast WOSAC is intended for rapid local evaluation. Use the official WOSAC evaluation server for final competition results.

#### Pre-BC + endpoint interpolation

CatK also includes TrajTok's inference-only `endpoint_interpolation` post-reconstruction. It rebuilds the 10 Hz points from CatK's generated 2 Hz token endpoints, with separate handling for moving, low-speed, and static agents. It does not change token selection, model weights, or training, and is disabled by default.

Evaluate the same pre-BC checkpoint with the post-reconstruction enabled:

```
CATK_CKPT=/path/to/pre_bc.ckpt \
python run.py experiment=inference_post_interp task_name=pre_bc_post_interp
```

For the moving-only ablation, disable low-speed/static reconstruction and
leave every non-moving agent and stopped segment unchanged:

```
CATK_CKPT=/path/to/pre_bc.ckpt \
python run.py \
  experiment=inference_post_interp_moving_only \
  task_name=pre_bc_post_interp_moving_only
```

This policy uses `0.5 m/s` as both the moving-agent and moving-segment
threshold. These thresholds can be overridden with
`model.model_config.decoder.endpoint_interpolation.moving_speed_threshold_mps`
and `moving_segment_speed_threshold_mps`.

For a paired comparison, run the baseline with the same checkpoint, validation data, and seed:

```
CATK_CKPT=/path/to/pre_bc.ckpt \
python run.py experiment=inference task_name=pre_bc_baseline
```

Both commands use Fast WOSAC 2025, the complete validation split, 32 rollouts, inference `K=48`, and all visible GPUs. The terminal and W&B run contain `val_closed/wosac/realism_meta_metric` and all component metrics. The endpoint method and thresholds can be overridden under `model.model_config.decoder.endpoint_interpolation`.

To inspect one validation scenario and one agent, use the CatK comparison tool. It generates one CatK token path, applies endpoint interpolation offline to that exact rollout, then reuses TrajTok's original six-panel plot for XY trajectory, heading, linear speed, linear acceleration, angular speed, and angular acceleration:

```
CATK_CKPT=/path/to/pre_bc.ckpt \
TRAJTOK_ROOT=/root/workspace/TrajTok \
python tools/compare_endpoint_interpolation.py \
  --split val \
  --scene-index 0 \
  --postprocess-policy moving_only
```

The default automatically selects the agent with the largest reduction in mean absolute angular acceleration. A specific scenario and agent can be selected instead:

```
CATK_CKPT=/path/to/pre_bc.ckpt \
python tools/compare_endpoint_interpolation.py \
  --split val \
  --scenario-id SCENARIO_ID \
  --agent-id AGENT_ID
```

Omit `--postprocess-policy moving_only` to inspect the full low-speed/static policy. Use `--select-motion-mode endpoint_interpolation`, `raw_token_expansion`, or `mixed` to automatically select an agent from one moving-only branch; the full policy also supports `low_speed_reconstruction` and `static_reconstruction`. Outputs are written under `outputs/catk_endpoint_interpolation_check`: a six-panel PNG, per-step CSV, raw/post-interpolation rollout PKLs, and a JSON summary. Moving-only filenames receive a `_moving_only` suffix, so they do not overwrite full-policy outputs. The default sampling `K=1` gives the clearest deterministic comparison; set `--sampling-num-k 48` to use the validation sampling width. Raw and post-interpolation outputs share one generated token path by construction.

To reproduce our final results, you should follow the following steps
1. Use [scripts/train.sh](scripts/train.sh) with the [BC pre-training config](configs/experiment/pre_bc.yaml) to pre-train the SMART-tiny 7M model.
2. Use [scripts/train.sh](scripts/train.sh) with the [CLSFT with CAT-K config](configs/experiment/clsft.yaml) to fine-tune the SMART-tiny model pre-trained in step 1.
3. Use [scripts/wosac_sub.sh](scripts/wosac_sub.sh) to pack the submission fille for `validate` or `test` split. Upload the `wosac_submission.tar.gz` file located in `logs` folder to the [WOSAC leaderboard](https://waymo.com/open/challenges/2024/sim-agents/) such that you can evaluate the model fine-tuned in step 2 on the WOSAC leaderboard.
4. Alternatively, you can do local validation with [scripts/local_val.sh](scripts/local_val.sh).

For Gaussian Mixture Model (GMM) based ego policy, the procedure is similar, just use the following configs
- [BC pre-training config for GMM-based ego policy](configs/experiment/ego_gmm_pre_bc.yaml)
- [CLSFT with CAT-K config for GMM-based ego policy](configs/experiment/ego_gmm_clsft.yaml)
- [Local validation config for GMM-based ego policy](configs/experiment/ego_gmm_local_val.yaml)
- There is no submission option for ego-policy.

## Performance

The submission of our CAT-K fine-tuned SMART to the [WOSAC Leaderboard](https://waymo.com/open/challenges/2024/sim-agents/) is found [here](https://waymo.com/open/challenges/sim-agents/results/5ea7a3eb-7337/1731338655639000/).
The submission of our reproduced SMART to the test split is found [here](https://waymo.com/open/challenges/sim-agents/results/5ea7a3eb-7337/1731391949275000/), note that it is not published to the leaderboard.

## Ablation configs

Please refer to [docs/ablation_models.md](docs/ablation_models.md) for the configurations of ablation models.
Specifically you will find the data augmentation methods used by [SMART](https://arxiv.org/abs/2207.05844) and [Trajeglish](https://arxiv.org/abs/2312.04535).

## Acknowledgement

Our code is based on [SMART](https://github.com/rainmaker22/SMART). We appreciate them for the valuable open-source code! Please don't forget to cite their amazing work as well!
