# ugrip24-ling

## Project Description
The "Can LLMs Ace the Linguistic Olympiad" project is part of the 2024 Undergraduate Research Internship Program (UGRIP) at MBZUAI. The objectives are:

### Week 1 (06/03 - 06/09)
- Build the experimental pipeline
- Benchmark several state-of-the-art large language models (LLMs) such as Llama-3 and Chat-GPT4 on the problem-solving capabilities of the International Lingistic Olympiad (IOL) problems, especially Rosetta-Stone
- Design custom prompts for phonology problems, syntax (Rosetta Stone), and more
- Evaluate performances with prompt-tuning
- Research on data contamination detection and problem-content encryption methods
- Experiment with evaluation metrics

### Week 2 (06/10 - 06/16)
- Artificial language generation and code-simulation recovery
- Tree of thought prompting setup
- Multi-turn interactions (does giving hints improve the performance? Human in the loop)

## Required Packages
- `pip install transformers`
- `pip install vllm`
- `pip install openai, AzureOpenAI`
- 
  
## Evaluation
- Automated evaluation scripts of LLM performances are in the `evaluation_pipeline` directory.
- This script uses the "PuzzLing" dataset's scoring programs for various accuracy metrics, such as exact matching (EM), BLEU scores, etc.
- Prepare the input data as follows:
![Step01_eval_file_dependency](images/eval_tutorial/step01_eval_file_dependency.png)

- Then, run the `llm_evaluate.py` script for automated LLM report generation. Follow instructions on the block comments.
![Step01_run_eval_code](images/eval_tutorial/step02_run_eval_code.png)

- Finally, observe the outputs:
![Step03_eval_outputs_01](images/eval_tutorial/step03_eval_outputs_01.png)
![Step04_eval_outputs_02](images/eval_tutorial/step04_eval_outputs_02.png)

