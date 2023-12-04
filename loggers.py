import os
import pytz
import wandb
import pandas as pd
from datetime import datetime

from os import listdir
from os.path import isfile, join

import common 

class WandBLogger:
    """WandB logger."""

    def __init__(self, args, system_prompt):
        self.logger = wandb.init(
            project = "jailbreak-llms",
            config = {
                "attack_model" : args.attack_model,
                "target_model" : args.target_model,
                "evaluator_model": args.evaluator_model,
                "keep_last_n": args.keep_last_n,
                "system_prompt": system_prompt,
                "index": args.index,
                "category": args.category,
                "goal": args.goal,
                "depth": args.depth,
                "width": args.width,
                "branching_factor": args.branching_factor,
                "target_str": args.target_str,
                "n_streams": args.n_streams,

            }
        )
        self.is_jailbroken = False
        self.query_to_jailbreak = 0
        self.table = pd.DataFrame()
        self.batch_size = args.n_streams
        self.index = args.index
        self.goal = args.goal
        self.jailbreak_prompt = None
        self.jailbreak_response = None

    def log(self, iteration: int, attack_list: list, response_list: list, judge_scores: list, on_topic_scores: list = None, conv_ids: list = None, parent_conv_ids: list = None):
        
        df = pd.DataFrame(attack_list)
        df["target_response"] = response_list
        df["judge_scores"] = judge_scores
        df["on_topic_scores"] = on_topic_scores if on_topic_scores is not None else [-1 for a in judge_scores]
        df["iter"] = iteration
        
        df["conv_id"] = conv_ids if conv_ids is not None else [i+1 for i in range(len(response_list))]
        df["parent_conv_id"] = parent_conv_ids if parent_conv_ids is not None else [i+1 for i in range(len(response_list))]
        
        self.table = pd.concat([self.table, df])

        self.query_to_jailbreak += len(response_list)

        if any([score == 10 for score in judge_scores]):
            jailbreak_ind = judge_scores.index(10)
            if not self.is_jailbroken:
                self.logger.log({"queries_to_jailbreak": self.query_to_jailbreak})
                self.is_jailbroken = True

            self.jailbreak_prompt = attack_list[jailbreak_ind]["prompt"]
            self.jailbreak_response = response_list[jailbreak_ind]

        self.logger.log({
            "iteration":iteration,
            "judge_scores":judge_scores,
            "mean_judge_score_iter": sum(judge_scores) / len(judge_scores),
            "is_jailbroken":self.is_jailbroken,
            "max_judge_score":self.table["judge_scores"].max(),
            "jailbreak_prompt":self.jailbreak_prompt,
            "jailbreak_response":self.jailbreak_response,
            "data": wandb.Table(data = self.table)})

        self.print_summary_stats(iteration)

    def finish(self):
        self.print_final_summary_stats()
        self.logger.finish()

    def print_summary_stats(self, iter):
        bs = self.batch_size
        df = self.table 
        mean_score_for_iter = df[df['iter'] == iter]['judge_scores'].mean()
        max_score_for_iter = df[df['iter'] == iter]['judge_scores'].max()
        
        num_total_jailbreaks = df[df['judge_scores'] == 10]['conv_id'].nunique()
        
        jailbreaks_at_iter = df[(df['iter'] == iter) & (df['judge_scores'] == 10)]['conv_id'].unique()
        prev_jailbreaks = df[(df['iter'] < iter) & (df['judge_scores'] == 10)]['conv_id'].unique()

        num_new_jailbreaks = len([cn for cn in jailbreaks_at_iter if cn not in prev_jailbreaks])

        print(f"{'='*14} SUMMARY STATISTICS {'='*14}")
        print(f"Mean/Max Score for iteration: {mean_score_for_iter:.1f}, {max_score_for_iter}")
        print(f"Number of New Jailbreaks: {num_new_jailbreaks}/{bs}")
        print(f"Total Number of Conv. Jailbroken: {num_total_jailbreaks}/{bs} ({num_total_jailbreaks/bs*100:2.1f}%)\n")

    def print_final_summary_stats(self):
        print(f"{'='*8} FINAL SUMMARY STATISTICS {'='*8}")
        print(f"Index: {self.index}")
        print(f"Goal: {self.goal}")
        df = self.table
        if self.is_jailbroken:
            num_total_jailbreaks = df[df['judge_scores'] == 10]['conv_id'].nunique()
            print(f"First Jailbreak: {self.query_to_jailbreak} Queries")
            print(f"Total Number of Conv. Jailbroken: {num_total_jailbreaks}/{self.batch_size} ({num_total_jailbreaks/self.batch_size*100:2.1f}%)")
            print(f"Example Jailbreak PROMPT:\n\n{self.jailbreak_prompt}\n\n")
            print(f"Example Jailbreak RESPONSE:\n\n{self.jailbreak_response}\n\n\n")            
            
        else:
            print("No jailbreaks achieved.")
            max_score = df['judge_scores'].max()
            print(f"Max Score: {max_score}")

        self.table.to_parquet(common.STORE_FOLDER + '/' + f'iter_{common.ITER_INDEX}_df')
