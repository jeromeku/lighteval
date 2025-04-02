#ruff: noqa
import torch
import argparse
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available
from lighteval.utils.utils import EnvConfig
from lighteval.tasks.registry import Registry
from transformers.utils.logging import set_verbosity
import time
from contextlib import contextmanager

import logging

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs
    from datetime import timedelta
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def create_task_def_single(task, few_shot_k, truncate_few_shot):
    return f"{task}|{few_shot_k}|{1 if truncate_few_shot else 0}"

def create_task_def(tasks: list[str], few_shot_k: int, truncate_few_shot: bool):
    task_defs = [create_task_def_single(task, few_shot_k, truncate_few_shot) for task in tasks]
    return ",".join(task_defs)
    

def format_time(time: float):
    if time < 1:
        return f"{time * 1000:.2f} ms"
    if time < 60:
        return f"{time:.2f} s"
    return f"{time / 60:.2f} min"

@contextmanager
def timer_context(name: str):
    delim = "=" * 100
    start_msg = f"{delim} Starting {name} {delim}"
    print(f"\n{start_msg}\n", flush=True)
    start = time.time()
    yield
    end = time.time()
    time_str = format_time(end - start)
    end_msg = f"{delim} Finished {name} in {time_str} seconds {delim}"
    print(f"\n{end_msg}\n", flush=True)

def main(args):
    with timer_context("EvaluationTracker"):
        evaluation_tracker = EvaluationTracker(
            output_dir="./results",
            save_details=True,
            push_to_hub=False,
        )

    with timer_context("PipelineParameters"):
        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.ACCELERATE,
            env_config=EnvConfig(cache_dir="tmp/"),
            override_batch_size=args.override_batch_size,
            max_samples=args.max_samples,
        )

    model_config = TransformersModelConfig(
        pretrained=args.model,
        dtype=args.dtype,
        use_chat_template=args.use_chat_template,
        accelerator=accelerator,
    )

    with timer_context("Registry"):
        registry = Registry()

    tasks = registry.expand_task_definition(args.task)
    tasks = tasks[:args.num_tasks]

    print("Running tasks: ", tasks)
    task_def = create_task_def(tasks, few_shot_k=args.num_few_shot_k, truncate_few_shot=args.truncate_few_shot)
    print("task_def: ", task_def)
    
    with timer_context("Pipeline"):
        pipeline = Pipeline(
            tasks=task_def,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
        )

    with timer_context("Pipeline.evaluate"):
        pipeline.evaluate()

    with timer_context("Pipeline.save_and_push_results"):
        pipeline.save_and_push_results()

    with timer_context("Pipeline.show_results"):
        pipeline.show_results()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task", type=str, default="helm|mmlu")
    args.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args.add_argument("--dtype", type=str, default="bfloat16")
    args.add_argument("--use_chat_template", type=bool, default=True)
    args.add_argument("--launcher_type", type=str, default="accelerate")
    args.add_argument("--override_batch_size", type=int, default=1)
    args.add_argument("--max_samples", type=int, default=10)
    args.add_argument("--num_tasks", type=int, default=2)
    args.add_argument("--num_few_shot_k", type=int, default=5)
    args.add_argument("--truncate_few_shot", type=bool, default=True)
    args.add_argument("--log_level", type=str, default="INFO", choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"])
    args = args.parse_args()
    log_level = getattr(logging, args.log_level)
    set_verbosity(log_level)
    args.dtype = getattr(torch, args.dtype)
    main(args)