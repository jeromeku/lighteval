#ruff: noqa
import os
import torch
import argparse
import json
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.requests import Request, RequestType
from lighteval.utils.imports import is_accelerate_available
from lighteval.utils.utils import EnvConfig
from lighteval.tasks.registry import Registry
from transformers.utils.logging import set_verbosity
import time
from contextlib import contextmanager
from dataclasses import asdict
import logging
from tabulate import tabulate

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs
    from datetime import timedelta
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

# Global dict to track timing for each section
TIMING_STATS = {}

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
    elapsed = end - start
    time_str = format_time(elapsed)
    end_msg = f"{delim} Finished {name} in {time_str} {delim}"
    print(f"\n{end_msg}\n", flush=True)
    
    # Store timing in global dict
    TIMING_STATS[name] = elapsed

def print_timing_summary():
    """Print a formatted table summarizing all timed sections."""
    if not TIMING_STATS:
        return
        
    # Convert times to formatted strings
    summary = [
        [name, format_time(elapsed)] 
        for name, elapsed in TIMING_STATS.items()
    ]
    
    # Add total time
    total_time = sum(TIMING_STATS.values())
    summary.append(["Total", format_time(total_time)])
    
    # Print table
    print("\nTiming Summary:")
    print(tabulate(
        summary,
        headers=["Section", "Time"],
        tablefmt="grid",
        colalign=("left", "right")
    ))

PARALLELISM_MANAGER_MAP = {
    "accelerate": ParallelismManager.ACCELERATE,
    "nanotron": ParallelismManager.NANOTRON,
    "tgi": ParallelismManager.TGI,
    "openai": ParallelismManager.OPENAI,
    "vllm": ParallelismManager.VLLM,
    "none": ParallelismManager.NONE,
    "sglang": ParallelismManager.SGLANG
}

def get_parallelism_manager(launcher_type: str) -> ParallelismManager:
    """Convert string launcher type to ParallelismManager enum.
    
    Args:
        launcher_type (str): String representation of launcher type
        
    Returns:
        ParallelismManager: Corresponding enum value
        
    Raises:
        ValueError: If invalid launcher type provided
    """
    
    launcher_type = launcher_type.lower()
    if launcher_type not in PARALLELISM_MANAGER_MAP:
        valid_types = ", ".join(PARALLELISM_MANAGER_MAP.keys())
        raise ValueError(f"Invalid launcher type '{launcher_type}'. Must be one of: {valid_types}")
        
    return PARALLELISM_MANAGER_MAP[launcher_type]

def create_save_dir(args) -> str:
    """
    Add subdir by model name and accelerator config
    """
    model_name = args.model.split("/")[-1]
    launcher_type = args.launcher_type
    save_dir = os.path.join(model_name, launcher_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")
    return save_dir

@timer_context("Main")
def main(args):
    with timer_context("EvaluationTracker"):
        evaluation_tracker = EvaluationTracker(
            output_dir=args.save_dir,
            save_details=True,
            push_to_hub=args.push_to_hub,
            hub_results_org=args.hub_results_org,
        )

    with timer_context("PipelineParameters"):
        env_config = EnvConfig(cache_dir=args.cache_dir)
        dataset_loading_processes = args.dataset_loading_processes or os.cpu_count()
        launcher_type = get_parallelism_manager(args.launcher_type)
        pp_params = {
            "launcher_type": launcher_type,
            "env_config": env_config,
            "max_samples": args.max_samples,
            "dataset_loading_processes": dataset_loading_processes,
            "override_batch_size": args.override_batch_size,
        }
        pipeline_params = PipelineParameters(**pp_params)

        pp_dict = asdict(pipeline_params)
        env_config = pp_dict.pop("env_config")
        env_config.pop("token")
        pp_dict["env_config"] = env_config
        print(f"PipelineParameters:\n{pp_dict}")
        with open(os.path.join(args.save_dir, "pipeline_params.json"), "w") as f:
            # Convert all values to strings
            pp_dict = {k: str(v) for k, v in pp_dict.items()}
            json.dump(pp_dict, f)

    model_config = TransformersModelConfig(
        pretrained=args.model,
        dtype=args.dtype,
        use_chat_template=args.use_chat_template,
        accelerator=accelerator,
    )
    print(f"ModelConfig:\n{model_config}")

    with timer_context("Registry"):
        registry = Registry()
    all_tasks = []
    for task in args.tasks:
        tasks = registry.expand_task_definition(task)
        all_tasks.extend(tasks)
    num_tasks = args.num_tasks if args.num_tasks is not None else len(all_tasks)
    tasks = all_tasks[:num_tasks]

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
        requests: dict[RequestType, list[Request]] = pipeline.requests
        req_type, reqs = next(iter(requests.items()))
        for req_type, reqs in requests.items():
            print(f"Request type: {req_type} {len(reqs)}")
            print(f"-> sample request: {reqs[0]}")
        
    with timer_context("Pipeline.evaluate"):
        pipeline.evaluate()

    with timer_context("Pipeline.save_and_push_results"):
        pipeline.save_and_push_results()

    with timer_context("Pipeline.show_results"):
        pipeline.show_results()

    # Print timing summary at the end
    print_timing_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["helm|mmlu"], help="Tasks to run in format suite|benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--launcher_type", type=str, default="accelerate")
    parser.add_argument("--override_batch_size", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--num_few_shot_k", type=int, default=5)
    parser.add_argument("--truncate_few_shot", type=bool, default=True)
    parser.add_argument("--dataset_loading_processes", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_results_org", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=os.getenv("HF_HOME"))
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"])
    
    args = parser.parse_args()
    log_level = getattr(logging, args.log_level)
    set_verbosity(log_level)
    args.dtype = getattr(torch, args.dtype)
    args.save_dir = args.save_dir or create_save_dir(args)
    print(f"Args:\n{args}")
    main(args)
