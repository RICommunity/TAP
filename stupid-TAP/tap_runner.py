import os
import json
import copy

import numpy as np

from system_prompts import get_attacker_system_prompt, get_attacker_system_prompt_with_memory
from loggers import WandBLogger
from evaluators import load_evaluator
from conversers import load_attack_and_target_models
from common import process_target_response, get_init_msg, conv_template, random_string

import common


def clean_attacks_and_convs(attack_list, convs_list):
    """
        Remove any failed attacks (which appear as None) and corresponding conversations
    """
    tmp = [(a, c) for (a, c) in zip(attack_list, convs_list) if a is not None]
    tmp = [*zip(*tmp)]
    attack_list, convs_list = list(tmp[0]), list(tmp[1])

    return attack_list, convs_list


def prune(on_topic_scores=None,
            judge_scores=None,
            adv_prompt_list=None,
            improv_list=None,
            convs_list=None,
            target_response_list=None,
            extracted_attack_list=None,
            sorting_score=None,
            attack_params=None):
    """
        This function takes
            1. various lists containing metadata related to the attacks as input,
            2. a list with `sorting_score`
        It prunes all attacks (and correspondng metadata)
            1. whose `sorting_score` is 0;
            2. which exceed the `attack_params['width']` when arranged
               in decreasing order of `sorting_score`.

        In Phase 1 of pruning, `sorting_score` is a list of `on-topic` values.
        In Phase 2 of pruning, `sorting_score` is a list of `judge` values.
    """
    # Shuffle the brances and sort them according to judge scores
    shuffled_scores = enumerate(sorting_score)
    shuffled_scores = [(s, i) for (i, s) in shuffled_scores]
    # Ensures that elements with the same score are randomly permuted
    np.random.shuffle(shuffled_scores)
    shuffled_scores.sort(reverse=True)

    def get_first_k(list_):
        width = min(attack_params['width'], len(list_))

        truncated_list = [list_[shuffled_scores[i][1]] for i in range(width) if shuffled_scores[i][0] > 0]

        # Ensure that the truncated list has at least two elements
        if len(truncated_list ) == 0:
            truncated_list = [list_[shuffled_scores[0][0]], list_[shuffled_scores[0][1]]]

        return truncated_list

    # Prune the brances to keep
    # 1) the first attack_params['width']-parameters
    # 2) only attacks whose score is positive

    if judge_scores is not None:
        judge_scores = get_first_k(judge_scores)

    if target_response_list is not None:
        target_response_list = get_first_k(target_response_list)

    on_topic_scores = get_first_k(on_topic_scores)
    adv_prompt_list = get_first_k(adv_prompt_list)
    improv_list = get_first_k(improv_list)
    convs_list = get_first_k(convs_list)
    extracted_attack_list = get_first_k(extracted_attack_list)

    return on_topic_scores,\
            judge_scores,\
            adv_prompt_list,\
            improv_list,\
            convs_list,\
            target_response_list,\
            extracted_attack_list


def run_tap(args):
    original_prompt = args.goal

    common.ITER_INDEX = args.iter_index
    common.STORE_FOLDER = args.store_folder
    common.DEBUG = args.debug

    # Initialize attack parameters
    attack_params = {
         'width': args.width,
         'branching_factor': args.branching_factor,
         'depth': args.depth
    }

    # ------------------------------------------------------------------
    # Phase-2 READ PATH: retrieve past successful attacks and inject them
    # into the attacker's system prompt as few-shot examples. When memory is
    # OFF this branch is skipped and the system prompt is byte-identical to the
    # stateless baseline.
    # ------------------------------------------------------------------
    memory = None
    retrieved = []
    memory_size_before = 0
    if args.use_memory:
        from memory import AttackMemory
        memory = AttackMemory(
            persist_dir=args.memory_dir,
            collection_name=args.memory_collection,
            embedding_model=args.embedding_model,
        )
        memory_size_before = memory.count()
        # Optional exact-match filters on stored metadata (attacker / target / judge).
        filter_target = args.memory_filter_target or (
            None if args.memory_any_target else args.target_model
        )
        filter_attacker = args.memory_filter_attacker or None
        filter_evaluator = args.memory_filter_evaluator or None
        retrieved = memory.retrieve(
            goal=args.goal,
            k=args.retrieval_k,
            mode=args.retrieval_mode,
            target_model=filter_target,
            attacker_model=filter_attacker,
            evaluator_model=filter_evaluator,
            exclude_goal=(args.goal if args.memory_exclude_same_goal else None),
        )
        system_prompt = get_attacker_system_prompt_with_memory(
            args.goal, args.target_str, retrieved
        )
        print(
            f"[memory] size_before={memory_size_before} "
            f"retrieved={len(retrieved)} mode={args.retrieval_mode} "
            f"k={args.retrieval_k} exclude_same_goal={args.memory_exclude_same_goal} "
            f"filters(target={filter_target}, attacker={filter_attacker}, "
            f"evaluator={filter_evaluator})",
            flush=True,
        )
    else:
        system_prompt = get_attacker_system_prompt(
            args.goal,
            args.target_str
        )

    # Initialize models and logger
    attack_llm, target_llm = load_attack_and_target_models(args)
    print('Done loading attacker and target!', flush=True)

    evaluator_llm = load_evaluator(args)
    print('Done loading evaluator!', flush=True)

    logger = WandBLogger(args, system_prompt)
    print('Done logging!', flush=True)

    if args.debug:
        print("\n" + "=" * 24 + " DEBUG attacker system prompt " + "=" * 24, flush=True)
        print(system_prompt, flush=True)
        if args.use_memory and retrieved:
            print("\n" + "-" * 24 + " retrieved memory (top-k) " + "-" * 24, flush=True)
            for i, r in enumerate(retrieved, 1):
                print(
                    f"[{i}] score={r.get('score')} similarity={r.get('similarity')} "
                    f"goal={r.get('goal')!r}",
                    flush=True,
                )
                print(f"    stored_prompt: {r.get('prompt')}", flush=True)
        print("=" * 72 + "\n", flush=True)

    # Initialize conversations
    batchsize = args.n_streams
    init_msg = get_init_msg(args.goal, args.target_str)
    processed_response_list = [init_msg for _ in range(batchsize)]
    convs_list = [conv_template(attack_llm.template,
                                self_id='NA',
                                parent_id='NA') for _ in range(batchsize)]

    for conv in convs_list:
        conv.set_system_message(system_prompt)

    # Begin TAP

    print('Beginning TAP!', flush=True)

    for iteration in range(1, attack_params['depth'] + 1):
        print(f"""\n{'='*36}\nTree-depth is: {iteration}\n{'='*36}\n""", flush=True)

        ############################################################
        #   BRANCH
        ############################################################
        extracted_attack_list = []
        convs_list_new = []

        for _ in range(attack_params['branching_factor']):
            print(f'Entering branch number {_}', flush=True)
            convs_list_copy = copy.deepcopy(convs_list)

            for c_new, c_old in zip(convs_list_copy, convs_list):
                c_new.self_id = random_string(32)
                c_new.parent_id = c_old.self_id

            extracted_attack_list.extend(
                    attack_llm.get_attack(convs_list_copy, processed_response_list)
                )
            convs_list_new.extend(convs_list_copy)

        # Remove any failed attacks and corresponding conversations
        convs_list = copy.deepcopy(convs_list_new)
        extracted_attack_list, convs_list = clean_attacks_and_convs(extracted_attack_list, convs_list)


        adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
        improv_list = [attack["improvement"] for attack in extracted_attack_list]

        ############################################################
        #   PRUNE: PHASE 1
        ############################################################
        # Get on-topic-scores (does the adv_prompt asks for same info as original prompt)
        on_topic_scores = evaluator_llm.on_topic_score(adv_prompt_list, original_prompt)

        # Prune attacks which are irrelevant
        (on_topic_scores,
        _,
        adv_prompt_list,
        improv_list,
        convs_list,
        _,
        extracted_attack_list) = prune(
            on_topic_scores,
            None, # judge_scores
            adv_prompt_list,
            improv_list,
            convs_list,
            None, # target_response_list
            extracted_attack_list,
            sorting_score=on_topic_scores,
            attack_params=attack_params)


        print(f'Total number of prompts (after pruning phase 1) are {len(adv_prompt_list)}')


        ############################################################
        #   QUERY AND ASSESS
        ############################################################
        target_response_list = target_llm.get_response(adv_prompt_list)
        print("Finished getting target responses.")

        # Get judge-scores (i.e., likelihood of jailbreak) from Evaluator
        judge_scores = evaluator_llm.judge_score(adv_prompt_list, target_response_list)
        print("Finished getting judge scores from evaluator.")

        ############################################################
        #   PRUNE: PHASE 2
        ############################################################
        # Prune attacks which to be fewer than attack_params['width']
        (on_topic_scores,
        judge_scores,
        adv_prompt_list,
        improv_list,
        convs_list,
        target_response_list,
        extracted_attack_list) = prune(
            on_topic_scores,
            judge_scores,
            adv_prompt_list,
            improv_list,
            convs_list,
            target_response_list,
            extracted_attack_list,
            sorting_score=judge_scores,
            attack_params=attack_params)

        # WandB log values
        logger.log(iteration,
                extracted_attack_list,
                target_response_list,
                judge_scores,
                on_topic_scores,
                conv_ids=[c.self_id for c in convs_list],
                parent_conv_ids=[c.parent_id for c in convs_list])

        # Truncate conversation to avoid context length issues
        for conv in convs_list:
            # Note that this does not delete the conv.role (i.e., the system prompt)
            conv.messages = conv.messages[-2*(args.keep_last_n):]

        # Early stopping criterion
        if any([score == 10 for score in judge_scores]):
            print("Found a jailbreak. Exiting.")
            break

        # `process_target_response` concatenates the target response, goal, and score
        #   -- while adding appropriate labels to each
        processed_response_list = [
                process_target_response(
                        target_response=target_response,
                        score=score,
                        goal=args.goal,
                        target_str=args.target_str
                    )
                    for target_response, score in zip(target_response_list, judge_scores)
            ]

    # ------------------------------------------------------------------
    # Phase-2 WRITE PATH: persist high-scoring prompts from this run back into
    # memory so future runs can retrieve them ("learning"). Runs before
    # logger.finish() so the in-memory results table is still populated.
    # ------------------------------------------------------------------
    n_written = 0
    if args.use_memory and memory is not None and args.memory_write:
        table = logger.table
        items = []
        if len(table) > 0 and 'prompt' in table.columns and 'judge_scores' in table.columns:
            for _, row in table.iterrows():
                items.append({
                    "prompt": row.get("prompt"),
                    "score": row.get("judge_scores"),
                })
        n_written = memory.add_successes(
            items,
            goal=args.goal,
            target_model=args.target_model,
            attacker_model=args.attack_model,
            evaluator_model=args.evaluator_model,
            goal_category=(args.goal_category or None),
            write_threshold=args.memory_write_threshold,
            pass_index=args.pass_index,
        )
        print(
            f"[memory] wrote {n_written} new prompts "
            f"(threshold>={args.memory_write_threshold}); size now {memory.count()}",
            flush=True,
        )

    # Sidecar with retrieval/write metadata for Phase-3 analysis of the
    # learning curve (memory size vs. queries-to-success, etc.).
    if args.store_folder:
        meta = {
            "iter_index": args.iter_index,
            "pass_index": args.pass_index,
            "goal": args.goal,
            "goal_category": args.goal_category,
            "attack_model": args.attack_model,
            "target_model": args.target_model,
            "evaluator_model": args.evaluator_model,
            "use_memory": bool(args.use_memory),
            "retrieval_mode": args.retrieval_mode,
            "retrieval_k": args.retrieval_k,
            "exclude_same_goal": bool(args.memory_exclude_same_goal),
            "memory_filter_target": args.memory_filter_target or None,
            "memory_filter_attacker": args.memory_filter_attacker or None,
            "memory_filter_evaluator": args.memory_filter_evaluator or None,
            "memory_size_before": memory_size_before,
            "n_retrieved": len(retrieved),
            "n_written": n_written,
            "memory_write_threshold": args.memory_write_threshold,
            "retrieved": [
                {
                    "goal": r.get("goal"),
                    "score": r.get("score"),
                    "similarity": r.get("similarity"),
                    "weighted_score": r.get("weighted_score"),
                    "attacker_model": r.get("attacker_model"),
                    "target_model": r.get("target_model"),
                    "evaluator_model": r.get("evaluator_model"),
                }
                for r in retrieved
            ],
        }
        try:
            meta_path = os.path.join(
                args.store_folder, f"memory_meta_iter_{args.iter_index}.json"
            )
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:  # noqa: BLE001 - metadata is best-effort
            print(f"[memory] failed to write metadata sidecar: {e}", flush=True)

    logger.finish()
