================================================================================
          MINIMAL-HARDWARE GRPO (2-GRPO) TRAINING & EXECUTION PLAN
================================================================================

--------------------------------------------------------------------------------
0. ACADEMIC REFERENCES & BENCHMARK METRICS
--------------------------------------------------------------------------------
Recent research shows that GRPO functions as an implicit contrastive optimizer 
(similar to DPO). Large group sizes (G=8 or G=16) are not strictly required 
for advantage estimation stability if compensated by mini-batch size.

- Key Reference: "It Takes Two: Your GRPO Is Secretly DPO" (2025).
- Performance Retention: 2-GRPO retains 97.6%–98.1% of full G=16 GRPO 
  performance on reasoning benchmarks.
- Efficiency Gains: Requires only 12.5% of sequence rollouts and cuts training 
  runtime by over 70%–79%.
- Hardware Impact: Reduces peak generation VRAM significantly while preserving 
  gradient unbiasedness.


--------------------------------------------------------------------------------
1. SETUP & CORE ARCHITECTURE
--------------------------------------------------------------------------------
Select a lightweight instruction-tuned model combined with Parameter-Efficient 
Fine-Tuning (PEFT) to keep memory footprint minimal during backpropagation.

Model Candidates:
  - Qwen/Qwen2.5-0.5B-Instruct (about 1 GB)
  - SmolLM2-135M-Instruct
  - Llama-3.2-1B-Instruct

Python Setup Snippet:
---------------------
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
model = get_peft_model(model, lora_config)


--------------------------------------------------------------------------------
2. DATASET CURRICULUM FILTERING
--------------------------------------------------------------------------------
With G=2, prompts where both generated completions succeed or both fail yield an 
advantage signal of 0 (std(R) = 0), causing zero policy update.

1. Offline Pre-Run: Generate 2 candidate rollouts per prompt on target dataset.
2. Filter Criteria: Retain ONLY prompts that produce a score differential 
   (e.g., 1 pass and 1 fail).
3. Objective: Guarantees nearly 100% of training steps generate a non-zero 
   contrastive advantage signal.


--------------------------------------------------------------------------------
3. DENSE COMPOSITE REWARD SYSTEM
--------------------------------------------------------------------------------
Dense rewards prevent binary 0/1 sparse bottlenecks. They ensure that even 
when both completions fail or pass, subtle quality differences generate a 
usable gradient.

Reward Logic:
-------------
def dense_reward_function(completions, ground_truths, **kwargs):
    rewards = []
    for completion, target in zip(completions, ground_truths):
        score = 0.0
        
        # 1. Structural / Formatting Check
        if "<think>" in completion and "</think>" in completion:
            score += 0.2
            
        # 2. Sequence Length Control (Prevents run-away generations)
        if len(completion) < 1000:
            score += 0.1
            
        # 3. Target Correctness & Partial Credit
        target_str = str(target).strip()
        if target_str in completion:
            score += 1.0
        elif any(digit in completion for digit in target_str if digit.isdigit()):
            score += 0.2  # Partial credit signal
            
        rewards.append(score)
    return rewards


--------------------------------------------------------------------------------
4. HARDWARE ESTIMATES & HYPERPARAMETERS
--------------------------------------------------------------------------------
Expected VRAM Usage (for ~0.5B to 1B Parameters):
  - BF16 / FP16 + QLoRA: ~4 GB - 6 GB VRAM
    (Compatible GPUs: RTX 3060 12GB, RTX 4060 8GB, T4)
  - 8-bit Quantized + LoRA: ~3 GB - 4 GB VRAM
    (Compatible GPUs: Consumer Laptops, GTX 1080 Ti, Free Colab)

Key Hyperparameters:
  - Group Size (num_generations / G): 2 (Minimal pairwise rollout)
  - Per-Device Train Batch Size: 4 to 8 (Mitigates G=2 gradient variance)
  - Gradient Accumulation Steps: 4 (Effective batch size = 16 to 32 prompts)
  - Max Completion Length: 256 to 512 tokens
  - KL Coefficient (beta): 0.01
  - Gradient Checkpointing: True


--------------------------------------------------------------------------------
5. HUGGING FACE TRL IMPLEMENTATION SCRIPT
--------------------------------------------------------------------------------
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir="./grpo_g2_optimized",
    learning_rate=1e-5,
    
    # 2-GRPO Parameters
    num_generations=2,                 # Minimal Pairwise Rollout (G=2)
    per_device_train_batch_size=4,     # Compensates variance with prompt batch size
    gradient_accumulation_steps=4,     # Simulates large global batch size
    
    max_completion_length=256,
    beta=0.01,
    
    logging_steps=5,
    save_strategy="steps",
    save_steps=100,
    gradient_checkpointing=True,
    bf16=True,                         # Set to fp16=True if GPU doesn't support bf16
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=dense_reward_function,
    args=training_args,
    train_dataset=filtered_hard_dataset,
)

trainer.train()
================================================================================


import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force CPU execution
device = "cpu"

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map=device
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("Model successfully loaded on CPU!")

The strategy outlined in the previous plan stems directly from the research paper "It Takes Two: Your GRPO Is Secretly DPO".The theoretical papers and benchmark results behind this strategy illustrate how $2$-GRPO performs and how to measure its output:📚 1. Key References & Theoretical FoundationPrimary Paper: "It Takes Two: Your GRPO Is Secretly DPO" (2025/2026)Core Finding: GRPO doesn't work simply because a large sample size ($G=16$) provides accurate population statistics. Rather, its policy updates operate like an online, token-weighted variant of Direct Preference Optimization (DPO). The primary driver of gradient stability is the total mini-batch size of prompts, not the rollout count $G$ per prompt.The Math Shift: Setting $G=2$ turns GRPO into an optimal online contrastive sampler ($A^+ \text{ vs } A^-$).📊 2. Published Empirical Benchmarks ($G=16$ vs. $G=2$)In evaluations across math and reasoning LLMs (such as RL post-trained Qwen-1.5B and Qwen-7B models), researchers benchmarked performance using standard metric suites:Core Performance MetricsBenchmark TaskStandard GRPO (G=16)Minimal 2-GRPO (G=2)Retained PerformanceCompute SavingsGSM8K (Grade School Math)Baseline ($100\%$)$98.1\%$ of Baseline~98% Accuracy$87.5\%$ fewer rolloutsMATH (Hard Mathematics)Baseline ($100\%$)$97.6\%$ of Baseline~97.6% Accuracy$79\%$ faster training timeHumanEval / MBPP (Code)Baseline ($100\%$)$97.2\%$ of Baseline~97% Accuracy$70\%+$ VRAM reductionNote: The study also showed that combining $2$-GRPO with light prompt resampling ($2\text{-GRPO+RS}$) actually matched or exceeded $16\text{-GRPO}$ performance while remaining significantly faster.🧪 3. Proposed Evaluation & Benchmark Plan for Your ModelTo verify that your $2$-GRPO implementation is training correctly without running out of memory, evaluate your model against this recommended evaluation suite:Recommended Benchmark DatasetsMath & Logic:GSM8K (Test Split): Standard metric for short step-by-step math reasoning.MATH-500: A subset of harder competition-level math problems to test deep multi-step deduction.Instruction Following & Format Adherence:IFEval (Instruction Following Evaluation): To strictly benchmark whether the model adheres to structural constraints (e.g., <think> tags, specific word limits, or output formats).Key Metrics to Monitor During TrainingPass@1 vs. Mean@32 Accuracy: Track how often a single attempt gets the right answer vs. sampling multiple trajectories.Format Adherence Rate (%): The percentage of generations that cleanly output formatting tags without truncation.Rollout Waste Rate: The proportion of training steps where both $G=2$ samples score identically ($0/0$ or $1/1$). If this exceeds 30%, your prompt filtering curriculum needs to be tightened.