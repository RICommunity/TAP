# TAP: A Query-Efficient Method for Jailbreaking Black-Box LLMs
[![arXiv](https://img.shields.io/badge/arXiv-2312.02119-b31b1b.svg)](https://arxiv.org/abs/2312.02119)  

**Abstract.**  While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thoughts reasoning until one of the generated prompts jailbreaks the target. Crucially, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning reduces the total number of queries sent to the target. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo) for more than 80% of the prompts using only a small number of queries. This significantly improves upon the previous state-of-the-art black-box method for generating jailbreaks.

## Overview of Tree of Attacks with Pruning (TAP)
TAP is an automatic query-efficient black-box method for jailbreaking LLMs using interpretable prompts. TAP utilizes three LLMs: an *attacker* whose task is to generate the jailbreaking prompts using tree-of-thoughts reasoning, an *evaluator* that assesses the generated prompts and evaluates whether the jailbreaking attempt was successful or not, and a *target*, which is the LLM that we are trying to jailbreak. 

We start with a single empty prompt as our initial set of attack attempts, and, at each iteration of our method, we execute the following steps:

1. *(Branch)* The attacker generates improved prompts (using tree-of-thought reasoning).   
2. *(Prune: Phase 1)* The evaluator eliminates any off-topic prompts from our improved ones.   
3. *(Attack and Assess)* We query the target with each remaining prompt and use the evaluator to score its responses. If a successful jailbreak is found, we return its corresponding prompt. 
4. *(Prune: Phase 2)* Otherwise, we retain the evaluator’s highest-scoring prompts as the attack attempts for the next iteration.    


Apart from the attacker, evaluator, and target LLMs, TAP is parameterized by the maximum depth $d\geq 1$, the maximum width $w\geq 1$, and the branching factor $b \geq 1$ of the tree-of-thought constructed by the method.

<img src="figures/tap.png" alt="An illustration of Tree of Attacks with Pruning (TAP)" style="width:auto;"/>

**Figure.** *Illustration of the four steps of Tree of Attacks with Pruning (TAP) and the use of the three LLMs (attacker, evaluator, and target) in each of the steps. This procedure is repeated until we find a jailbreak for our target or until a maximum number of repetitions is reached.*


## Getting Started
Please ensure that the API keys for the models you want to use are stored in the appropriate variables. (For GPT Models, store the key in `OPENAI_API_KEY` and for PaLM models in `PALM_API_KEY`). If you would like to use local versions of Vicuna or Lamma, please add paths to them in `config.py`. Alternatively, if you have API access to Vicuna or Llama from a third party, please update the API links in `config.py`. (Depending on the interface of the API, you may have to update `class APIModel` in `language_models.py`.)

Run `wandb login` to set up `wandb` before using TAP.

## Running Experiment
`main_TAP.py` contains an implementation of TAP. To run an experiment, you can execute `main_TAP.py` as follows
```
python main_TAP.py --attack-model [[ATTACKER]] --target-model [[TARGET]] --evaluator-model [[EVALUATOR]] --goal [[GOAL]] --target-str [[TARGET_STR]] --store-folder [[FOLDER]]
```
Where `[[GOAL]]` is the prompt asking for restricted information (e.g., "how to build a bomb") and `[[TARGET_STR]]` is a desired prefix in the response of the target model (e.g., "Sure, here is how to build a bomb").

You can modify the default parameters of TAP by passing the arguments `branching-factor`, `width`, and `depth` with the desired values to `main_TAP.py`.

For ease of use, we illustrate how to run TAP against several standard closed-source and open-source LLMs in `Demo.ipynb`.

## Acknowledgments
TAP builds upon another recent black-box jailbreaking method: Prompt Automatic Iterative Refinement (PAIR). This repository is based on an implementation of PAIR available here: [JailbreakingLLMs](https://github.com/patrickrchao/JailbreakingLLMs).

## Citation
If you use this implementation in your research, please cite the following paper:
```
   @misc{mehrotra2023treeOfAttacks,
      title={Tree of Attacks: Jailbreaking Black-Box LLMs Automatically},
      author={Anay Mehrotra and Manolis Zampetakis and Paul Kassianik and Blaine Nelson and Hyrum Anderson and Yaron Singer and Amin Karbasi},
      year={2023},
      archivePrefix={arXiv},
      primaryClass={cs.LG} 
   }
```   

*(Have you used this implementation? Let us know!)*

## Contact 
Please feel free to email `anaymehrotra1@gmail.com`.
