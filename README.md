# O1-Pruner
Official repository for paper: O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning
# [we plan to release the code, checkpoints in Feb 2nd, 2025]
<img src="image/pipeline.png"></img>

# Abstract
Recently, long-thought reasoning LLMs, such as OpenAI's O1, have adopted extended reasoning processes similar to how humans ponder over complex problems. This reasoning paradigm significantly enhances the model's problem-solving abilities and has achieved promising results. However, long-thought reasoning process leads to a substantial increase in inference time. A pressing challenge is reducing the inference overhead of long-thought LLMs while ensuring accuracy. 
In this paper, we experimentally demonstrate that long-thought reasoning models struggle to effectively allocate token budgets based on problem difficulty and reasoning redundancies. To address this, we propose Length-Harmonizing Fine-Tuning (**O1-Pruner**), aiming at minimizing reasoning overhead while maintaining accuracy. This effective fine-tuning method first estimates the LLM's baseline performance through pre-sampling and then uses RL-style fine-tuning to encourage the model to generate shorter reasoning processes under accuracy constraints. This allows the model to achieve efficient reasoning with lower redundancy while maintaining accuracy. Experiments on various mathematical reasoning benchmarks show that **O1-Pruner** not only significantly reduces inference overhead but also achieves higher accuracy, providing a novel and promising solution to this challenge.
