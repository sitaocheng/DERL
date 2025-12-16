# DERL

We release the code for the paper

[**Differentiable Evolutionary Reinforcement Learning**](https://arxiv.org/abs/2512.13399).

Models are available at [huggingface_repo](https://huggingface.co/DifferentiableEvolutionaryRL).


Here is the main idea and results of DERL.
![DERL](figures/figure1.jpg)

Our bi-level evolutionary training is illustrated as follows.
![Bi-level Evolutionary Training](figures/figure2.jpg)


## File Descriptions

- `grpo_coordinator.py.py`, `grpo_coordinator_population.py`: The main code coordinates the outer-loop and inner-loop training.
- `data_split_by_portion.py`: Splits the dataset into different portions of subsets for SFT and RL.
- `datagen_profile.py`: The main script for generating synthetic human biographies (profiles) with rich attributes and relationships.
- `relations.py`: Defines the relationship templates and logic for generating various combinations of relationships.
- `reshape_q_template.py`: Processes and reshapes question templates to create the final question-answer pairs for each relation combinations.

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