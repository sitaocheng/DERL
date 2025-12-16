# DERL

We release the code for the paper

[**Differentiable Evolutionary Reinforcement Learning**](https://arxiv.org/abs/2512.13399).

Models are available at [huggingface_repo](https://huggingface.co/DifferentiableEvolutionaryRL).


Here is the main idea and results of DERL.
![DERL](figures/figure1.jpg)

Our bi-level evolutionary training is illustrated as follows.
![Bi-level Evolutionary Training](figures/figure2.jpg)


## Quick Start

1. Modify the path and settings in grpo_coordinator.py and run_qwen2_5_3b_llm_math_meta_llm_rl.sh based on your environments.

2. Download the base model as inner-loop policy model (e.g., Qwen/Qwen-2.5-3B) and modify the model_path in the script accordingly. 

3. Download our provided initialized outer-loop Meta-Optimizer from our [huggingface_repo](https://huggingface.co/DifferentiableEvolutionaryRL) modify  INITIAL_MODEL_PATH in grpo_coordinator.py accordingly.

4. Download and prepare your training and testing data into `./dataset/`. Modify the data path in eval_llm.py. We recommend using scripts in `./examples/data_preprocess` or based on your own environments and tasks.

5. Run python grpo_coordinator.py.


## File Descriptions

- `grpo_coordinator.py`, `grpo_coordinator_population.py`: The main code coordinates the outer-loop and inner-loop training for DERL and DERL-pop., respectively.
- `eval_llm.py`: The code for evaluating validation performance and testing performance.
- `./verl/utils/reward_score/meta_reward_grpo.py`: The code to modify the atomic primitives.
- `./examples/grpo_trainer/run_qwen2_5_3b_llm_math_meta_llm_rl.sh`: script for inner-loop evolution.
- `./examples/grpo_trainer/run_qwen2_5_3b_llm_math_meta_llm_rl_population.sh`: script for DERL-pop. inner-loop evolution.
- `./examples/grpo_trainer/run_qwen2_5_05b_instr_math_meta_rl.sh`: script for outer-loop evolution.




## Citation

If you find this repository useful, please cite our paper:

```bibtex
@misc{cheng2025differentiableevolutionaryreinforcementlearning,
      title={Differentiable Evolutionary Reinforcement Learning}, 
      author={Sitao Cheng and Tianle Li and Xuhan Huang and Xunjian Yin and Difan Zou},
      year={2025},
      eprint={2512.13399},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.13399}, 
}
``` 