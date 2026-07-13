# LAMP 复现逻辑框架

本文档用于指导复现论文 *Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making* 中提出的 **LAMP** 框架。

当前复现目标不是完整复现实验结果，也不包含消融实验，而是先实现 LAMP 的核心机制，使代码具备如下完整流程：

```text
经济环境观测
→ 语言化推理
→ 智能体交流
→ 反思与信念更新
→ 语言增强状态
→ MARL 策略决策
```

LAMP 的核心思想是：

```text
LLM 负责理解经济状态、生成推理、交流和反思；
MARL 负责根据数值信息和语言信息学习长期最优策略。
```

## 1. 整体流程

LAMP 的整体训练循环可以表示为：

```text
for each episode:
    reset environment
    reset short-term experience

    for each time step t:
        1. 获取环境数值观测
        2. Think 判断是否生成短期新闻或长期新闻
        3. 每个 agent 基于新闻和自身状态生成 reasoning
        4. 在长期 checkpoint 时调用 Speak
        5. agent 生成 public statement
        6. 其他 agent 读取 statement 并生成 reflection
        7. Decide 将数值观测和语言信息编码成增强状态
        8. RL actor 输出动作
        9. 环境执行动作并返回 reward 和 next state
        10. 存储 RL transition
        11. 更新 actor-critic 网络
        12. 保存高 reward reasoning 到经验池
```

整体上，LAMP 由三个主要模块组成：

```text
Think
Speak
Decide
```

此外还需要两个辅助组件：

```text
Experience Pool
Text Encoder / Language Embedding
```

## 2. 环境接口

为了复现 LAMP，需要先抽象一个经济环境接口。该接口不必一开始完全复现 TaxAI，但必须提供与 LAMP 交互所需的基本信息。

环境需要支持：

```python
obs = env.reset()
next_obs, rewards, done, info = env.step(actions)
```

其中 `obs` 应包含：

```text
global observation O_t^g
private observations O_t^{h,i}
```

全局观测可以包括：

```text
wage
average asset
average income
average efficiency
GDP
social welfare
wealth gini
```

家庭私有观测可以包括：

```text
asset of household i
efficiency of household i
income of household i
previous action of household i
```

家庭 agent 的动作是：

```text
a_t^i = {p_t^i, h_t^i}
```

其中：

```text
p_t^i: savings rate
h_t^i: labor supply
```

## 3. Think 模块

Think 模块负责把数值经济状态转化为语言化经济解释。

它包含两条路径：

```text
short-term reasoning
long-term reasoning
```

### 3.1 短期推理

短期推理用于捕捉突发经济冲击。

触发条件：

```text
如果某个关键指标变化超过阈值 sigma，则触发 short-term news
```

关键指标包括：

```text
wealth gini
social welfare
GDP
```

短期新闻输入：

```text
current global observation
previous global observation
latest long-term news
```

输出：

```text
short-term news R_t^short
agent private reasoning ψ_t^i
agent economic status κ_t^i
```

其中经济状态可以离散化为：

```text
0: bad
1: neutral
2: good
```

### 3.2 长期推理

长期推理用于捕捉结构性趋势。

触发条件：

```text
当 t 到达长期 checkpoint L_i 时触发
```

长期新闻输入：

```text
global observations over a time window
```

输出：

```text
long-term news R_Li^long
long-term reasoning
```

长期推理更关注：

```text
经济增长趋势
贫富差距变化
福利变化
劳动供给变化
消费趋势
长期稳定性
```

## 4. Experience Pool

Experience Pool 用来保存历史上高 reward 的 reasoning trajectory。

它不是 RL replay buffer，而是语言推理经验池。

需要维护两个池子：

```text
ShortTermExperiencePool
LongTermExperiencePool
```

每条 reasoning experience 至少包含：

```python
{
    "agent_id": int,
    "time_step": int,
    "observation": dict,
    "reasoning": str,
    "reflection": str,
    "action": list,
    "reward": float,
    "embedding": vector
}
```

短期经验池保存最近一段时间内表现好的 reasoning。

长期经验池在长期 checkpoint 时更新，将高 reward reasoning 存入长期记忆。

检索方式：

```text
当前 observation / reasoning
→ text embedding
→ kNN search
→ 返回相似历史经验
```

论文中使用 FAISS。代码复现时可以先设计统一接口：

```python
experience_pool.add(experience)
experience_pool.retrieve(query, top_k)
```

后续可以选择：

```text
numpy cosine similarity
FAISS
sklearn NearestNeighbors
```

## 5. Speak 模块

Speak 模块负责智能体之间的语言交流和反思。

它包含三个步骤：

```text
candidate statement generation
statement selection
reflection
```

### 5.1 生成候选发言

对于每个 agent，LLM 根据以下信息生成多条候选 public statements：

```text
private observation
short-term reasoning
long-term reasoning
retrieved experience
current economic status
```

例如生成 3 条候选发言：

```python
candidate_statements = [
    statement_1,
    statement_2,
    statement_3,
]
```

### 5.2 选择发言

论文中使用 self-attention selector 选择一条发言。

复现时可以设计为：

```python
selected_statement = statement_selector(candidate_statements, context)
```

早期实现可以用简化版本：

```text
rule-based scoring
embedding similarity scoring
small neural network selector
```

后续再替换成 attention-based selector。

输出：

```text
public statement v_t^i
```

所有 agent 的发言组成：

```text
V_t = {v_t^1, v_t^2, ..., v_t^N}
```

### 5.3 Reflection

每个 agent 接收其他 agent 的发言，并更新对其他人的判断。

Reflection 输入：

```text
agent private observation
all public statements
agent private reasoning
```

Reflection 输出：

```text
belief
trust
self-reflection
```

形式为：

```python
{
    "wealth_belief": [low/mid/high for each peer],
    "trust": [0-10 for each peer],
    "self_reflection": str,
}
```

论文中的形式是：

```text
w_t^{i→j}: 对其他 agent 财富层级的判断
τ_t^{i→j}: 对判断的信心
α_t^i: 自我反思
```

## 6. Text Encoder

LAMP 中所有语言信息最终都需要转成向量，才能输入 RL 网络。

语言信息包括：

```text
short-term news
long-term news
private reasoning
public statement
reflection
retrieved experience
```

编码流程：

```text
text
→ text encoder E_text
→ pooled vector
→ projection layer P
→ language embedding m_t^i
```

代码中可以封装为：

```python
language_embedding = text_encoder.encode(texts)
projected_embedding = projection(language_embedding)
```

论文中冻结 `E_text`，只训练投影层 `P`。

因此复现时建议：

```text
text encoder: frozen
projection layer: trainable
```

如果暂时不接入真实大模型，可以先用 mock encoder：

```text
hash embedding
sentence-transformer
随机但固定的 embedding
```

但接口要保持不变，方便后续替换。

## 7. Decide 模块

Decide 是真正的 RL 决策模块。

它接收：

```text
numeric observation
language embedding
reflection embedding
```

组成增强状态：

```text
x_t = (O_t^g, O_t^{h,1:N}, m_t^{1:N})
```

对每个 agent，actor 输出动作：

```python
action_i = actor_i(local_obs_i, language_embedding_i)
```

动作包括：

```text
savings rate p_t^i
labor supply h_t^i
```

可以用连续动作空间：

```text
p_t^i ∈ [p_min, p_max]
h_t^i ∈ [0, h_max]
```

## 8. RL 部分：MADDPG

LAMP 的 RL 部分可以基于 MADDPG 实现。

需要包含：

```text
Actor network
Critic network
Target actor
Target critic
Replay buffer
Policy update
Critic update
Soft target update
```

每个 agent 有自己的 actor：

```python
actor_i(local_obs_i, language_embedding_i) -> action_i
```

集中式 critic 输入联合状态和联合动作：

```python
critic(global_state, joint_actions) -> Q_value
```

训练过程：

```text
1. actor 输出所有 agent 的动作
2. 环境执行 joint action
3. 得到 rewards 和 next observations
4. 存入 replay buffer
5. 从 replay buffer 采样 batch
6. 更新 critic
7. 更新 actor
8. soft update target networks
```

Replay buffer 中保存的是标准 RL transition：

```python
{
    "state": x_t,
    "actions": a_t,
    "rewards": r_t,
    "next_state": x_t+1,
    "done": done,
}
```

注意区分：

```text
RL replay buffer: 训练 actor-critic
Experience Pool: 保存高 reward 语言推理
```

## 9. 推荐代码目录结构

后续复现代码可以设计为：

```text
lamp/
    __init__.py

    envs/
        base_env.py
        taxai_wrapper.py
        toy_economy_env.py

    modules/
        think.py
        speak.py
        decide.py
        experience_pool.py
        text_encoder.py
        statement_selector.py

    rl/
        maddpg.py
        actor.py
        critic.py
        replay_buffer.py
        noise.py

    llm/
        base_llm.py
        mock_llm.py
        openai_llm.py
        qwen_llm.py
        prompts.py

    utils/
        config.py
        logging.py
        serialization.py
        metrics.py

    train_lamp.py
    run_lamp.py
```

其中核心职责是：

```text
think.py: 生成 short-term / long-term reasoning
speak.py: 生成 statement 和 reflection
decide.py: 拼接数值状态和语言向量，调用 actor
experience_pool.py: 存储和检索高 reward reasoning
text_encoder.py: 文本向量化
maddpg.py: 多智能体强化学习训练逻辑
```

## 10. 最小可复现版本

第一阶段不建议直接实现完整论文系统，而是先实现一个最小 LAMP 原型。

最小版本包括：

```text
1. 一个 toy economy environment
2. 多个 household agents
3. Think 模块生成简单文本 reasoning
4. Speak 模块生成简单 public statement
5. Reflection 生成 belief/trust/self-reflection
6. Text encoder 将文本转成向量
7. MADDPG 根据数值观测 + 语言向量输出动作
8. Experience Pool 保存高 reward reasoning
```

这个版本的目标是跑通：

```text
Think → Speak → Decide → Env Step → RL Update → Experience Update
```

而不是追求论文指标。

## 11. 复现优先级

建议按以下顺序实现：

```text
Step 1: 定义环境接口和 toy economy environment
Step 2: 实现 ReplayBuffer
Step 3: 实现基础 MADDPG，只使用数值观测跑通训练
Step 4: 实现 TextEncoder 和 language embedding 拼接
Step 5: 实现 mock LLM
Step 6: 实现 Think 模块
Step 7: 实现 Experience Pool
Step 8: 实现 Speak 和 Reflection 模块
Step 9: 将 Think-Speak-Decide 接入 LAMP 主循环
Step 10: 替换 mock LLM 为真实 LLM 接口
Step 11: 替换 toy environment 为 TaxAI 或 TaxAI-like wrapper
```

核心原则：

```text
先保证 RL 主干能运行，
再逐步增加语言模块，
最后再接真实经济环境和真实 LLM。
```

## 12. 当前复现边界

本阶段只复现 LAMP 框架本身，因此暂时不做：

```text
完整 TaxAI 实验复现
三种经济场景对比
baseline 对比
消融实验
论文数值结果复现
```

当前目标是实现一个结构正确的 LAMP：

```text
有 Think
有 Speak
有 Decide
有 Experience Pool
有语言 embedding
有 MARL actor-critic
能完成一次完整训练循环
```

## 13. 最终闭环

LAMP 的代码复现应围绕一个核心闭环展开：

```text
数值经济状态
→ Think 生成经济解释和推理
→ Experience Pool 检索历史高价值推理
→ Speak 生成交流消息
→ Reflection 更新信念和信任
→ Text Encoder 编码语言信息
→ Decide / MADDPG 输出储蓄率和劳动供给
→ 环境返回 reward
→ 更新 RL 网络和语言经验池
```

也就是说，LAMP 不是“LLM 直接做动作”，而是：

```text
LLM 提供语言理解、推理、交流和反思；
RL 策略网络负责长期优化和动作决策。
```

