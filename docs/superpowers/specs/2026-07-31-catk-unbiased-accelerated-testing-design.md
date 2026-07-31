# CAT-K 无偏加速安全测试设计（Phase 1）

## 状态

本设计已经过交互式审查并获得用户确认。

本规范的实施范围仅包含 Phase 1：固定风险倾斜重要性采样
（fixed risk-tilted importance sampling）。AIS/CEM 和 D2RL/PPO 仅保留稳定扩展
接口，不属于首版实现范围。

## 1. 决策摘要

首版不依赖强化学习。CAT-K 原始闭环 token 策略定义基准分布 \(P\)，加速器只对
每条 rollout 中的一辆背景挑战车（principal other vehicle，POV）构造风险倾斜
提议分布 \(Q\)。每次被改变的采样都记录精确的 \(p/q\) 概率比，最终使用普通
重要性采样恢复基准 CAT-K 下的碰撞、碰撞类型和近碰撞概率。

核心约束如下：

- 当前主车和所有普通背景智能体继续按照原始 CAT-K 策略采样；
- 每条 8 秒 rollout 最多确定性选择并锁定一辆 POV；
- POV 只在关键状态按照 \(q\) 采样，其他状态仍按照 \(p\) 采样；
- 风险函数不奖励某一种碰撞类型；
- 正式评估不读取未来轨迹真值；
- 提议分布参数必须在正式评估之前冻结；
- 正式估计不裁剪权重，也不使用自归一化权重替代无偏估计；
- 碰撞视频可以在首次碰撞时结束，但仿真轨迹仍保存完整 8 秒；
- 一旦首次主车碰撞发生，后续步骤恢复 \(q=p\)，事件权重不再变化。

“无偏”的目标分布是当前配置下的 CAT-K 闭环生成分布，而不是现实世界分布。
未来如需相对于真实交通分布无偏，需要另外建立现实数据分布建模与校准层。

## 2. 目标与非目标

### 2.1 目标

Phase 1 必须实现：

1. 更高频地生成影响主车的碰撞和近碰撞；
2. 对 CAT-K 基准分布下的事件概率给出可审计的重要性采样估计；
3. 分别估计追尾、侧擦/侧碰、角度碰撞、正面碰撞及其他/不确定碰撞的绝对概率；
4. 输出置信区间、有效样本量和达到同一统计精度时的加速比；
5. 保留完整概率账本，使每条 rollout 的权重能够离线重建；
6. 不修改 CAT-K 已学习的 checkpoint、词表或 Fast WOSAC 数值后端。

### 2.2 非目标

Phase 1 不包含：

- D2RL、PPO 或其他强化学习训练；
- 边评估边更新 \(\epsilon\)、\(\beta\) 或关键性阈值；
- 同一 rollout 中同时控制多辆背景车；
- 指定追尾、侧碰或正面碰撞类型作为奖励；
- 取代现有 WOSAC realism 指标；
- 宣称对现实世界事故率无偏；
- 外部 ADS 主车的正式接入。

最后一项会预留接口，但首版仍由 CAT-K 生成主车和其他智能体。

## 3. 现有代码集成边界

现有递归闭环生成位于
`src/smart/modules/agent_decoder.py`。推理阶段执行 16 个 2 Hz token 步骤；
每个 token 内部展开为 5 个 10 Hz 未来帧。每一步重新构建地图到智能体和
智能体到智能体关系，计算 `next_token_logits`，随后由
`src/smart/utils/rollout.py::sample_next_token_traj` 采样 token 并更新下一步状态。

首版采用采样器注入，而不是把关键性、事件统计和文件输出全部写入
`AgentDecoder`：

```text
src/smart/accelerated_testing/
├── __init__.py
├── config.py                 # 配置和合法性校验
├── token_distribution.py     # 从 logits 构造精确基准 p
├── risk.py                   # token 对风险和 criticality
├── pov_selector.py           # 确定性选择与单 POV 锁定
├── proposal.py               # r、q 和 log(p/q)
├── controller.py             # 每条 rollout 的状态机
├── ledger.py                 # 逐步概率账本与 manifest
├── events.py                 # 主车碰撞、类型与近碰撞
├── estimators.py             # IS、CI、ESS 和加速比
└── runner.py                 # 正式评估入口与结果汇总
```

对现有文件的修改限定为：

- `src/smart/utils/rollout.py`：抽取“构造 Top-K 行为分布”和“从给定分布采样”
  的纯函数；默认路径必须保持现有行为；
- `src/smart/modules/agent_decoder.py`：增加可选 sampler/controller hook，
  向其提供当前状态、token 几何、智能体 ID/类型、形状和 logits；
- 模型评估入口：把 rollout 元数据、完整 10 Hz 轨迹和 controller 账本交给
  加速测试 runner；
- 新增独立配置和测试。

不得直接修改
`src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics/` 中嵌入的数值
实现。安全事件检测器可以调用其中稳定的几何函数，但不能改变其 WOSAC 行为，
以免破坏现有数值对齐和来源可追溯性。

## 4. 基准分布 \(p\)

### 4.1 精确定义

基准 \(p\) 不是未经处理的 2048 维 softmax，而是原始 CAT-K 闭环评估采样器
真正执行的行为分布。

对于每个智能体和时间步：

1. 从模型得到 2048 维 logits；
2. 按 `sampling_scheme.num_k` 选出 Top-K token；
3. 将 Top-K logits 除以 `sampling_scheme.temp`；
4. 只在 Top-K 支持集 \(\mathcal S_p\) 上做 softmax。

因此：

\[
p(a\mid s)=
\begin{cases}
\operatorname{softmax}(\ell_a/T), & a\in\mathcal S_p,\\
0, & a\notin\mathcal S_p.
\end{cases}
\]

默认正式评估配置与当前推理配置一致：

- `criterium=topk_prob`；
- `num_k=48`；
- `temp=1.0`。

正式加速评估只允许 `topk_prob`。以下模式因使用未来 GT 而必须立即拒绝：

- `topk_prob_sampled_with_dist`；
- `topk_dist_sampled_with_prob`。

若用户改变 K 或温度，该配置本身就定义了一个新的基准 \(P\)。结果报告必须记录
K、温度和完整配置哈希，不得把不同基准下的结果直接合并。

Phase 1 不能给 \(p=0\) 的支持集外 token 增加采样概率。若希望危险 proposal
覆盖完整 2048 词表，必须先把 baseline 和 proposal 同时设为 `num_k=2048`，
并把它作为新的基准重新运行 baseline MC；不能只扩大 \(q\) 而保持原 Top-48
\(p\)。

### 4.2 数值要求

- 使用 `log_softmax` 和 `logsumexp` 构造概率；
- 概率计算和累计 `log_weight` 使用 float64；
- 采样可以使用模型设备上的原生精度，但写入账本前必须转为 float64；
- 支持集外的 token 必须同时满足 \(p=0\) 和 \(q=0\)；
- 对支持集内所有 token 必须满足 \(p>0\) 且 \(q>0\)。

## 5. 单 POV 选择

### 5.1 候选集合

每个 0.5 秒 token 步骤，以每条 rollout 的当前状态建立候选集合。候选必须同时
满足：

- 不是主车；
- 类型为 vehicle；
- 当前状态有效且由 CAT-K 闭环更新；
- 与主车中心距离不超过 60 米；
- 具有稳定的 `track_id` 和有效车辆尺寸。

行人和骑行者在 Phase 1 中继续按 \(p\) 采样，不作为 POV。以后可在保持同一
概率接口的前提下扩展。

### 5.2 动作敏感型 criticality

对候选车 \(i\) 及其可采样 token \(a\in\mathcal S_{p_i}\)，定义：

\[
R_i(a)=
\mathbb E_{e\sim\tilde p_{\text{ego}}}
[\operatorname{risk}(e,a)],
\]

其中 \(\tilde p_{\text{ego}}\) 是主车 \(p\) 中 Top-8 token 重新归一化后的分布。
风险计算使用这个期望，不得读取同一步最终采到的主车 token，因此不存在同一步
“偷看主车动作”。

动作敏感型 criticality 为：

\[
C_i =
\max_{a\in\mathcal S_{p_i}}R_i(a)
-\mathbb E_{a\sim p_i}[R_i(a)].
\]

它衡量“改变该车 token 能够增加多少风险”，而不是简单把距离近等同于关键。
实现可以为诊断目的计算全部 2048 个 token 的风险，但支持集外 token 不得参与
\(C_i\)、\(r\) 或 \(q\)，否则会与实际基准采样器不一致。

### 5.3 锁定规则

- 在尚未选择 POV 时，找出当前所有 \(C_i\ge\tau_C\) 的候选；
- 选择 \(C_i\) 最大的候选；
- 数值相同时选择最小 `track_id`，保证确定性；
- 首次选择后锁定该 `track_id`，直到 rollout 结束；
- 锁定后每步只重算该 POV 的 \(C_i\)；当 \(C_i<\tau_C\) 时使用 \(q=p\)；
- 锁定车失效或离开场景后不选择替代车，剩余步骤全部使用 \(q=p\)；
- 若 8 秒内始终没有候选超过阈值，该 rollout 完全等于基准 rollout。

选择规则是当前状态的确定性函数，因此不需要额外的“选择哪辆车”概率比。每条
并行 rollout 独立锁定自己的 POV。

## 6. 风险函数

### 6.1 预测视界

对主车候选 token 与 POV 候选 token：

1. 使用 token 内的 0.1、0.2、0.3、0.4 和 0.5 秒车辆框；
2. 根据 token 最后两个内部点得到终端速度；
3. 保持终端航向和速度不变，以 0.1 秒间隔外推到 1.0 秒；
4. 在整个 1.0 秒短视界上取最大风险。

这只是构造 proposal 的短视界近似，不会替换 CAT-K 最终生成轨迹。

### 6.2 风险分量

每对候选轨迹使用与 Fast WOSAC 一致的车辆框尺寸和圆角几何，计算：

- \(d_{\min}\)：1 秒内最小有符号多边形距离；
- \(t_{\text{overlap}}\)：冻结航向、保持当前相对速度时第一次预测重叠的时间，
  搜索范围为 0 至 5 秒；不存在时记为 \(+\infty\)；
- \(v_{\text{close}}\)：沿两车中心连线的正向接近速度；
- \(a_{\text{req}} =
  v_{\text{close}}^2/(2\max(d_{\min},0.1))\)，仅在
  \(v_{\text{close}}>0\) 时有效，否则为 0。

默认无量纲风险为：

\[
\begin{aligned}
s_{\text{overlap}} &= \sigma(-d_{\min}/0.25),\\
s_{\text{gap}} &= \exp[-\max(d_{\min},0)/1.0],\\
s_{\text{tto}} &=
\begin{cases}
\exp(-t_{\text{overlap}}/1.5), & t_{\text{overlap}}\le5,\\
0, & \text{otherwise},
\end{cases}\\
s_{\text{brake}} &= \sigma[(a_{\text{req}}-3.0)/0.5],\\
\operatorname{risk} &= \tfrac14(
s_{\text{overlap}}+s_{\text{gap}}+s_{\text{tto}}+s_{\text{brake}}).
\end{aligned}
\]

所有尺度和四个权重都是显式配置项；Phase 1 默认等权。风险函数中不得出现
“追尾”“侧碰”“正面碰撞”等类型奖励，也不得使用未来 GT、未来红绿灯 GT 或
未来记录轨迹。

### 6.3 关键性阈值

\(\tau_C\) 只使用训练集上的基准 CAT-K rollout 校准：

1. 对首次主车碰撞前 3 秒定义 precursor window；
2. 一条碰撞 rollout 在该窗口内至少一次满足
   \(\max_i C_i\ge\tau_C\)，即视为被覆盖；
3. 选择仍能覆盖至少 99% 碰撞 rollout 的最高阈值；
4. 在满足召回率的前提下，目标是只保留约 1% 至 5% 的 token 步骤；
5. 若无法同时达到两者，优先保证 99% precursor 召回率并报告实际保留比例。

自动校准至少需要 100 条训练集主车碰撞 rollout。样本不足时不得使用验证集或
测试集补足；应增加训练集 pilot rollout。仍无法满足时使用保守回退
\(\tau_C=0\)，并明确标记“未完成稀疏阈值校准”。

## 7. 风险倾斜提议分布

对锁定 POV 的关键状态，在其基准支持集上计算 \(p\)-加权均值和标准差：

\[
\mu_R=\sum_a p(a)R(a),\qquad
\sigma_R^2=\sum_a p(a)[R(a)-\mu_R]^2.
\]

当 \(\sigma_R<10^{-6}\) 时令 \(r=p\)。否则：

\[
z_a=\operatorname{clip}\left(
\frac{R(a)-\mu_R}{\sigma_R},-5,5
\right),
\]

\[
\log r(a\mid s)=
\log p(a\mid s)+\beta z_a
-\operatorname{logsumexp}_{b\in\mathcal S_p}
[\log p(b\mid s)+\beta z_b].
\]

最终提议为：

\[
q(a\mid s)=(1-\epsilon)p(a\mid s)+\epsilon r(a\mid s).
\]

Phase 1 默认：

- \(\epsilon=0.05\)；
- \(\beta=1.0\)；
- `z_clip=5.0`。

三者在正式评估中固定。由于 \(\epsilon<1\)，有
\(q(a)\ge(1-\epsilon)p(a)\)，所以基准支持集始终被完整覆盖。风险标准化裁剪属于
proposal 定义的一部分；它不等于裁剪最终重要性权重。

以下情况强制 \(q=p\)：

- 尚未锁定 POV；
- 当前不是关键状态；
- 锁定 POV 当前无效；
- 风险方差过小；
- 已发生首次主车碰撞；
- 加速测试功能被关闭。

## 8. 每步数据流与权重

每个 0.5 秒步骤按以下顺序执行：

1. 使用上一步 token 更新所有智能体状态；
2. 重新构建时间、地图到智能体和智能体到智能体关系；
3. CAT-K 为所有智能体计算 logits；
4. 从 logits 构造精确基准 \(p\)；
5. 对尚未锁定 POV 的 rollout 计算候选 criticality 并确定性锁定；
6. 对锁定 POV 计算 \(r\) 和 \(q\)；
7. 主车与普通背景智能体从 \(p\) 采样，POV 从当前 \(q\) 采样；
8. 将 token 展开为 5 个 10 Hz 帧并更新闭环状态；
9. 在 10 Hz 帧上检测首次主车碰撞；
10. 写入逐步账本并累计权重。

对一条 rollout：

\[
\log W =
\sum_{t\in\mathcal T_q}
[\log p(a_t\mid s_t)-\log q(a_t\mid s_t)],
\]

其中 \(\mathcal T_q\) 只包含 POV 实际使用 \(q\ne p\) 的步骤。主车和其他智能体
在 \(P\) 与 \(Q\) 下使用同一条件分布，因此概率比为 1。

若首次主车碰撞发生在 token 内部 10 Hz 帧：

- 碰撞事件、时间戳和类型立即固定；
- 当前 token 的 \(p/q\) 已计入权重；
- 从下一个 token 步骤起强制 \(q=p\)；
- 视频允许在首次碰撞帧结束；
- 仿真继续到 8 秒并保存完整轨迹；
- 首次碰撞后的轨迹只用于导出，不参与首次事件或类型判定。

## 9. 事件定义

加速测试事件是独立的 ego-centric 安全统计，不改变现有 Fast WOSAC 指标。

### 9.1 主车碰撞

碰撞检测复用 Fast WOSAC 的车辆几何约定：

- `CORNER_ROUNDING_FACTOR=0.7`；
- 严格使用 signed distance \(<0\) 判定重叠；
- 在每个 token 的 5 个 10 Hz 内部帧检测；
- 只统计当前帧之后新发生且涉及主车的碰撞；
- 当前帧已经重叠的对象不计为新未来碰撞，并在结果中单独标记
  `initial_overlap=true`。

对初始已重叠的主车-对象配对，检测器先进入 suppressed 状态；只有该配对至少
一个 10 Hz 帧恢复到 signed distance \(\ge0\) 后，后续再次从非重叠变为
signed distance \(<0\) 才记为新碰撞。近碰撞检测同样忽略尚未解除的初始重叠
配对。这样无需静默删除整个场景，也不会把仿真开始前已经存在的接触当成未来
事件。

若首次碰撞帧同时涉及多辆车，碰撞对象选择 signed distance 最小者；仍相同时
选择最小 `track_id`。其他同时碰撞对象写入辅助列表，但不产生多个主事件。

### 9.2 碰撞类型

在首次碰撞帧，对主车与选定碰撞对象计算：

- 航向差
  \(\Delta\psi=|\operatorname{wrap}(\psi_o-\psi_e)|\in[0^\circ,180^\circ]\)；
- 将对象中心相对主车中心的位置转换到主车坐标系，得到
  \(x_{\text{rel}},y_{\text{rel}}\)；
- 尺寸归一化接触位置
  \(u_x=|x_{\text{rel}}|/[(L_e+L_o)/2]\) 和
  \(u_y=|y_{\text{rel}}|/[(W_e+W_o)/2]\)；
- 根据相邻 10 Hz 帧有限差分得到两车速度；
- 中心连线接近速度
  \(v_{\text{close}}=-\hat r^\top(v_o-v_e)\)。

轴向判定使用 `contact_axis_margin=0.15`：

- \(u_x-u_y\ge0.15\)：纵向接触；
- \(u_y-u_x\ge0.15\)：横向接触；
- 其他情况：轴向不确定。

默认分类规则：

1. \(\Delta\psi\le45^\circ\)、\(v_{\text{close}}>0\) 且纵向接触：
   `rear_end`；
2. \(\Delta\psi\le45^\circ\)、\(v_{\text{close}}>0\) 且横向接触：
   `sideswipe`；
3. \(45^\circ<\Delta\psi<135^\circ\) 且 \(v_{\text{close}}>0\)：
   `angle`；
4. \(\Delta\psi\ge135^\circ\)、\(v_{\text{close}}>0\) 且纵向接触：
   `head_on`；
5. 其余情况：
   `other_or_unknown`。

如果中心距离过小导致中心连线不稳定、速度无效或车辆尺寸无效，也必须进入
`other_or_unknown`，不得强制分到四个主要类别。

### 9.3 近碰撞

近碰撞与真实碰撞互斥：只有整条 8 秒 rollout 没有主车碰撞时才可成立。主车与
任一有效智能体至少满足以下一个条件即记为 `near_miss_union=true`：

1. 观测到的最小圆角框 signed distance 位于 \([0,1.0)\) 米；
2. 任一 10 Hz 帧上的通用 constant-velocity time-to-overlap 小于 1.5 秒；
3. 存在可识别的共享冲突区时，post-encroachment time（PET）小于 1.5 秒；
4. 广义所需避免碰撞减速度 \(a_{\text{req}}>3.0\,\mathrm{m/s^2}\)。

四个条件必须分别输出。PET 仅在两条 10 Hz 轨迹的车辆包络形成共享冲突区且两车
先后占用该区域时定义；否则输出 `pet_applicable=false`，不得把“不适用”当成
安全或危险。

现有 Fast WOSAC TTC 只针对同车道前车，因此不能直接作为通用近碰撞 TTC。
Phase 1 必须在独立事件模块中实现通用 time-to-overlap，同时保留原 WOSAC TTC
不变。

## 10. 无偏估计与统计报告

### 10.1 固定场景集上的目标

对 \(S\) 个正式评估场景，每个场景运行 \(M_s\) 条 rollout。事件 \(A\) 的目标是
对场景等权的 CAT-K 概率：

\[
\mu_A=\frac1S\sum_{s=1}^S
\mathbb E_{P_s}[I(A)].
\]

普通重要性采样估计为：

\[
\hat\mu_A=
\frac1S\sum_{s=1}^S
\frac1{M_s}\sum_{m=1}^{M_s}
W_{sm}I_{sm}(A).
\]

该写法即使各场景 rollout 数量不同也不会让某个场景被过度加权。若所有
\(M_s\) 相同，才可等价写成所有 rollout 的简单平均。

碰撞、每个碰撞类型和每个近碰撞子事件都使用相同公式分别估计。正式估计：

- 不裁剪 \(W\)；
- 不对 \(W\) 做样本内归一化；
- 不静默删除大权重或危险 rollout。

可额外报告 clipped 或 self-normalized 结果作为明确标注的“有偏诊断”，但它们
不能替代主结果。

### 10.2 类型概率

每个类型首先报告绝对概率：

\[
\hat\mu_c=\widehat P(\text{first ego collision type}=c).
\]

这是普通 IS 的无偏估计目标。条件类型构成
\(\hat\mu_c/\hat\mu_{\text{collision}}\) 是比值估计，有限样本下不保证严格无偏，
只能作为附加描述并通过 bootstrap 给出区间。报告不得把条件比值称为“严格
无偏”。

原始 \(Q\) 样本中的类型比例允许明显偏斜；“碰撞类型无偏”指加权后的绝对类型
概率以 CAT-K 基准 \(P\) 为目标，而不是要求未加权危险样本数量均匀。

### 10.3 置信区间与诊断

默认报告 90% 置信区间：

- 主区间使用场景级 cluster bootstrap；
- 每次先有放回抽样场景，再在抽中的场景内有放回抽样 rollout；
- 默认 `bootstrap_replicates=2000`；
- 同时输出基于 \(Y=W I(A)\) 的解析标准误作为快速诊断。

统计报告还必须包含：

\[
\operatorname{ESS}=
\frac{(\sum W)^2}{\sum W^2},
\]

以及：

- `ESS/N`；
- 权重均值和 \(\mathbb E_Q[W]=1\) 的偏差；
- 最大权重；
- 权重变异系数；
- 90% CI 相对半宽
  `RHW=(upper-lower)/(2*estimate)`；
- 原始 \(Q\) 事件率；
- 加权 \(P\) 事件率。

当点估计为 0 时 RHW 不定义，报告 `insufficient_events=true`。

### 10.4 加速比

统计加速比定义为达到同一 `RHW=0.3` 时：

\[
\text{acceleration ratio} =
\frac{\text{baseline MC 所需 rollout 数}}
{\text{fixed-IS 所需 rollout 数}}.
\]

所需样本量通过相同场景集上的 bootstrap 精度曲线估计。另行报告 wall-clock
加速比，其中必须包含 risk scoring、账本和事件检测开销。

实现正确不等于已经获得有效加速。只有同时满足下列条件时才能宣称“加速有效”：

- 目标事件 `RHW<=0.3`；
- `ESS/N>=0.10`；
- 统计加速比大于 1；
- wall-clock 加速比大于 1；
- 与独立大样本 baseline MC 的差异区间包含 0。

## 11. 输出与可审计性

每次正式评估创建不可覆写的运行目录：

```text
accelerated_testing/{run_id}/
├── manifest.json
├── step_ledger.jsonl.gz
├── rollout_summary.jsonl
├── trajectories.pt
├── report.json
└── failures.jsonl
```

`manifest.json` 至少包含：

- CAT-K checkpoint SHA-256；
- agent vocabulary SHA-256；
- Hydra 完整解析配置及 SHA-256；
- 代码 Git commit 和 dirty 状态；
- 场景列表及其哈希；
- RNG seed 规则；
- \(K,T,\epsilon,\beta,z_{\max},\tau_C\)；
- 事件阈值和碰撞分类阈值；
- proposal 是否已冻结；
- PyTorch、CUDA 和 GPU 信息。

每个实际使用 \(q\) 的步骤在 `step_ledger.jsonl.gz` 中保存：

- scenario ID、rollout ID、seed、2 Hz 步骤和对应 10 Hz 范围；
- POV `track_id`；
- baseline 支持集 token ID；
- 支持集上的 `log_p`、风险 \(R\)、`log_r` 和 `log_q`；
- 所选 token ID 和确定性的 per-rollout RNG key；
- \(C\)、\(\epsilon\)、\(\beta\)；
- 当前与累计 `log_weight`；
- 是否在该 token 中首次碰撞。

`rollout_summary.jsonl` 至少包含：

- 是否选择 POV 及其 ID；
- 首次主车碰撞时间、对象 ID 和类型；
- initial overlap 和同时碰撞对象；
- 四个 near-miss 子指标及 union；
- 最终 `log_weight`；
- 关键步骤数；
- 完整轨迹位置。

保存支持集概率向量而非只保存被选 token 的概率，确保可以离线检查归一化并完整
重建每一步 \(q\)。

## 12. 故障处理

采用 fail-closed 策略。以下任一情况使本次正式评估失败：

- logits、风险、概率或累计权重出现 NaN/Inf；
- \(p\)、\(r\) 或 \(q\) 未在容差 \(10^{-6}\) 内归一化；
- 支持集内存在 \(p>0,q=0\)；
- POV ID、token ID、车辆形状或主车角色缺失；
- 账本无法重建累计 `log_weight`；
- 正式评估中 proposal 参数发生变化；
- 正式配置启用了依赖未来 GT 的 sampler；
- 某条 rollout 因异常被静默排除。

故障 rollout 的元数据写入 `failures.jsonl`，但不能通过丢弃该样本后继续发布正式
估计。需要修复原因并重新运行整个受影响的正式批次。

`other_or_unknown` 碰撞类型、没有选出 POV、PET 不适用和没有发生危险事件都是
合法结果，不属于故障。

## 13. 默认配置

```yaml
accelerated_testing:
  enabled: true
  phase: fixed_is
  baseline:
    criterium: topk_prob
    num_k: 48
    temperature: 1.0
  pov:
    candidate_radius_m: 60.0
    vehicle_only: true
    max_per_rollout: 1
    lock_once_selected: true
    criticality_threshold: 0.0
    criticality_threshold_source: conservative_fallback
  risk:
    ego_top_k: 8
    horizon_s: 1.0
    internal_dt_s: 0.1
    tto_max_s: 5.0
    z_clip: 5.0
    component_weights: [0.25, 0.25, 0.25, 0.25]
  proposal:
    epsilon: 0.05
    beta: 1.0
    frozen: true
  events:
    near_gap_m: 1.0
    near_tto_s: 1.5
    near_pet_s: 1.5
    near_required_decel_mps2: 3.0
    same_direction_max_deg: 45.0
    opposing_direction_min_deg: 135.0
    contact_axis_margin: 0.15
  statistics:
    confidence_level: 0.90
    bootstrap_replicates: 2000
    target_rhw: 0.30
    minimum_ess_fraction: 0.10
```

这里的 `criticality_threshold=0.0` 是样本不足时的保守回退，不代表推荐的正式
阈值。完成训练集校准后，配置生成阶段必须把它替换为确定数值，将
`criticality_threshold_source` 改为训练校准 artifact 的 SHA-256。该字段不能在
正式运行时自动解析为验证集统计量。

每条 rollout 的采样 RNG key 由
`(global_seed, scenario_id, rollout_id, token_step, agent_id)` 经过稳定的
64 位哈希得到。相同 key 必须产生相同 token，且结果不能依赖 batch 排列或
distributed rank。加速功能关闭时仍沿用现有采样路径，避免改变普通推理回归
结果。

## 14. 测试与验收

### 14.1 单元测试

1. `token_distribution` 与现有 `topk_prob` 在同 logits、K、温度和随机数下采样
   一致；
2. \(\epsilon=0\) 时 \(q=p\) 且每步 `log_ratio=0`；
3. \(\beta=0\) 时 \(r=p,q=p\)；
4. 任意合法配置下 \(p,r,q\) 归一且满足支持集条件；
5. 风险函数只接收当前状态、token 库和模型输出，不存在未来 GT 参数；
6. POV 只锁定一次，平局按最小 `track_id` 解决；
7. 锁定 POV 无效后不会换车；
8. synthetic 几何覆盖五个碰撞类型和阈值边界；
9. 近碰撞四个子条件、互斥规则和 PET 不适用均有测试；
10. 账本离线重放能在 \(10^{-10}\) 容差内重建 `log_weight`。

### 14.2 解析测试

构造事件概率可解析的离散 Markov toy 环境，验证：

- 原始 MC 收敛到解析值；
- 风险倾斜后的未加权事件率发生改变；
- 普通 IS 的碰撞和各类型绝对概率落入 90% CI；
- clipped/self-normalized 结果只出现在诊断字段；
- 改变 proposal 不改变加权估计目标。

### 14.3 CatK 回归测试

- `accelerated_testing.enabled=false` 时，默认推理结果与当前代码在相同 seed 下
  一致；
- 现有 Fast WOSAC 指标键和值不受影响；
- 现有 pre_bc、clsft 和普通 inference 配置无需新增参数即可运行；
- GT-conditioned training rollout sampler 仍可用于原训练流程，但正式加速评估
  会拒绝它；
- batched scenarios 和每场景多 rollout 的 POV 状态互不串扰。

### 14.4 统计验收

在独立 held-out 场景上同时运行冻结的 baseline MC 与 fixed IS：

- 对有足够事件数量的碰撞、类型和近碰撞指标，二者差值的 90% cluster-bootstrap
  CI 包含 0；
- fixed IS 的原始危险事件率应高于 baseline；若没有提高，说明 proposal 无效，
  但不能据此更改同一正式批次的 proposal；
- `mean(W)` 与 1 相容；
- `ESS/N>=0.10`；
- 没有无法解释的失败 rollout；
- 只有达到第 10.4 节的全部“加速有效”条件后，报告中才显示加速成功。

稀有类型事件数不足时必须报告 `underpowered=true`，不得用 CI 重叠声称已验证
该类型无偏。

## 15. 分阶段扩展

### 15.1 Phase 2：AIS/CEM

可以在训练或独立 pilot 场景上用 adaptive importance sampling 或 cross-entropy
method 调整 \(\epsilon\)、\(\beta\) 和 \(\tau_C\)。调参结束后必须：

- 固化为普通 Phase 1 配置；
- 保存调参数据集和 artifact 哈希；
- 在未参与调参的正式场景上重新运行；
- 沿用相同 \(p/q\) 账本和估计器。

### 15.2 Phase 3：D2RL/PPO

D2RL 可复用冻结的 CAT-K hidden state、相对运动和风险摘要，学习状态相关
\(\epsilon(s)\)。它只替换 proposal 参数生成器，不改变：

- 基准 \(p\)；
- \(q=(1-\epsilon)p+\epsilon r\) 的支持集保护；
- 概率账本；
- 事件定义；
- 普通 IS 估计器；
- 正式评估冻结原则。

强化学习的作用是进一步降低方差和学习“何时介入”，不是赋予估计无偏性。

### 15.3 外部 ADS 主车

未来把主车替换为外部 ADS 时，引入：

```text
EgoPolicy.step(observation, rng) -> ego action or trajectory
EgoPolicy.predict_candidates(observation) -> risk-scoring candidates
```

同一个外部 EgoPolicy 必须同时用于 baseline 和 proposal rollout，因此其条件
概率在 \(P/Q\) 比值中抵消。若黑盒 ADS 无法返回候选分布，风险评分可以使用其
确定性规划轨迹或多个短时闭环查询；背景车的精确 \(p/q\) 逻辑不变。此时估计
目标变为“CAT-K 背景交通条件于该外部 ADS 行为”的混合系统风险。

## 16. 参考

设计借鉴了 *Dense reinforcement learning for safety validation of
autonomous vehicles* 中“自然分布、加速分布、重要性采样恢复和 dense critical
states”的总体思想。CAT-K Phase 1 的关键差异是利用离散 motion-token logits
直接构造可审计的固定 proposal，并将 D2RL 延后为可选优化层。
