#!/usr/bin/env python3
import json
import random
import os
from typing import List, Dict, Any
from pathlib import Path

MODULES = ["reflection", "planner", "executor", "memory"]

def generate_task_description(task_id: int) -> str:
    tasks = [
        "Implement a function to calculate the Fibonacci sequence",
        "Debug a memory leak in a web application",
        "Optimize database query performance",
        "Create a REST API for user authentication",
        "Refactor legacy code to use modern design patterns",
        "Fix concurrency issues in a multi-threaded application",
        "Implement a caching mechanism for frequently accessed data",
        "Create unit tests for a payment processing module",
        "Migrate a monolithic application to microservices",
        "Implement rate limiting for API endpoints",
        "Debug a race condition in async code",
        "Optimize image processing pipeline for better performance",
        "Implement a real-time notification system",
        "Create a data validation framework",
        "Fix security vulnerabilities in authentication flow",
        "Implement a distributed task queue system"
    ]
    return tasks[task_id % len(tasks)]

def generate_module_output(module: str, round_num: int, quality: str = "good") -> str:
    outputs = {
        "reflection": {
            "good": f"Round {round_num}: Analyzing the current approach... The implementation shows good separation of concerns. Previous attempts revealed potential optimization opportunities. Need to consider edge cases for boundary conditions.",
            "bad": f"Round {round_num}: Looks fine. No issues. Continue."
        },
        "planner": {
            "good": f"Round {round_num}: Step 1: Analyze current codebase structure. Step 2: Identify optimization points. Step 3: Implement incremental improvements. Step 4: Validate changes with tests. Step 5: Document modifications.",
            "bad": f"Round {round_num}: Just implement it. Write code. Test. Done."
        },
        "executor": {
            "good": f"Round {round_num}: Using tool: code_analyzer with params: {{file: 'main.py', metrics: true}}. Results show 85% coverage. Implementing optimization using caching strategy. Applied refactoring pattern successfully.",
            "bad": f"Round {round_num}: Run code. Error occurred. Try again. Fixed maybe."
        },
        "memory": {
            "good": f"Round {round_num}: Query: 'previous optimization attempts'. Result: Found 3 relevant entries about caching strategies. Storing: 'successful_cache_implementation' with context and performance metrics.",
            "bad": f"Round {round_num}: Query: 'stuff'. Result: Nothing found. Store: 'data'."
        }
    }
    return outputs[module][quality]

def generate_trajectory(task: str, num_rounds: int, module_qualities: Dict[str, str]) -> str:
    trajectory_parts = []
    for round_num in range(1, num_rounds + 1):
        round_content = f"Round {round_num}:\n"
        round_content += "<think>\n<reflection>\n"
        round_content += generate_module_output("reflection", round_num, module_qualities.get("reflection", "good"))
        round_content += "\n</reflection>\n"
        
        round_content += "<memory query>\n"
        round_content += f"Searching for relevant context from round {round_num}"
        round_content += "\n</memory query>\n"
        
        round_content += "<memory result>\n"
        round_content += f"Found {random.randint(0, 5)} relevant entries"
        round_content += "\n</memory result>\n</think>\n"
        
        round_content += "<plan>\n"
        round_content += generate_module_output("planner", round_num, module_qualities.get("planner", "good"))
        round_content += "\n</plan>\n"
        
        round_content += "<action>\n"
        round_content += generate_module_output("executor", round_num, module_qualities.get("executor", "good"))[:50]
        round_content += "\n</action>\n"
        
        round_content += "<tool results>\n"
        round_content += f"Tool execution successful with output..."
        round_content += "\n</tool results>\n"
        
        round_content += "<memory store>\n"
        round_content += f"Storing results from round {round_num}"
        round_content += "\n</memory store>\n"
        
        trajectory_parts.append(round_content)
    
    return "\n".join(trajectory_parts)

def extract_module_outputs(trajectory: str, module: str) -> str:
    if module == "reflection":
        import re
        pattern = r'<reflection>(.*?)</reflection>'
        matches = re.findall(pattern, trajectory, re.DOTALL)
        return " | ".join([m.strip() for m in matches])
    
    elif module == "planner":
        import re
        pattern = r'<plan>(.*?)</plan>'
        matches = re.findall(pattern, trajectory, re.DOTALL)
        return " | ".join([m.strip() for m in matches])
    
    elif module == "executor":
        import re
        action_pattern = r'<action>(.*?)</action>'
        tool_pattern = r'<tool results>(.*?)</tool results>'
        actions = re.findall(action_pattern, trajectory, re.DOTALL)
        tools = re.findall(tool_pattern, trajectory, re.DOTALL)
        combined = []
        for a, t in zip(actions, tools):
            combined.append(f"Action: {a.strip()} | Result: {t.strip()}")
        return " | ".join(combined)
    
    elif module == "memory":
        import re
        query_pattern = r'<memory query>(.*?)</memory query>'
        result_pattern = r'<memory result>(.*?)</memory result>'
        store_pattern = r'<memory store>(.*?)</memory store>'
        queries = re.findall(query_pattern, trajectory, re.DOTALL)
        results = re.findall(result_pattern, trajectory, re.DOTALL)
        stores = re.findall(store_pattern, trajectory, re.DOTALL)
        combined = []
        for q, r, s in zip(queries, results, stores):
            combined.append(f"Query: {q.strip()} | Result: {r.strip()} | Store: {s.strip()}")
        return " | ".join(combined)

def generate_dataset() -> List[Dict[str, Any]]:
    dataset = []
    num_pairs_per_module = 8
    task_counter = 0
    
    for module in MODULES:
        for pair_idx in range(num_pairs_per_module):
            task_counter += 1
            task = generate_task_description(task_counter)
            num_rounds = random.randint(2, 4)
            
            # Generate positive trajectory
            positive_qualities = {m: "good" for m in MODULES}
            positive_trajectory = generate_trajectory(task, num_rounds, positive_qualities)
            
            # Generate negative trajectory (only target module is bad)
            negative_qualities = {m: "good" for m in MODULES}
            negative_qualities[module] = "bad"
            negative_trajectory = generate_trajectory(task, num_rounds, negative_qualities)
            
            # Create positive sample
            positive_sample = {
                "task": task,
                "trajectory_full_context": positive_trajectory,
                "module_k_all_rounds": extract_module_outputs(positive_trajectory, module),
                "outcome": 1,
                "preference": 1,
                "target_module": module,
                "pair_id": f"{module}_{pair_idx}",
                "is_positive": True
            }
            
            # Create negative sample
            negative_sample = {
                "task": task,
                "trajectory_full_context": negative_trajectory,
                "module_k_all_rounds": extract_module_outputs(negative_trajectory, module),
                "outcome": 0,
                "preference": 0,
                "target_module": module,
                "pair_id": f"{module}_{pair_idx}",
                "is_positive": False
            }
            
            dataset.append(positive_sample)
            dataset.append(negative_sample)
    
    return dataset

def save_dataset(dataset: List[Dict[str, Any]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full dataset
    output_path = Path(output_dir) / "synthetic_dataset.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Save pairs for training
    pairs = []
    for i in range(0, len(dataset), 2):
        if i + 1 < len(dataset):
            positive = dataset[i]
            negative = dataset[i + 1]
            if positive["pair_id"] == negative["pair_id"]:
                pair = {
                    "pair_id": positive["pair_id"],
                    "target_module": positive["target_module"],
                    "task": positive["task"],
                    "positive": {
                        "trajectory_full_context": positive["trajectory_full_context"],
                        "module_k_all_rounds": positive["module_k_all_rounds"],
                        "outcome": positive["outcome"]
                    },
                    "negative": {
                        "trajectory_full_context": negative["trajectory_full_context"],
                        "module_k_all_rounds": negative["module_k_all_rounds"],
                        "outcome": negative["outcome"]
                    }
                }
                pairs.append(pair)
    
    pairs_path = Path(output_dir) / "training_pairs.json"
    with open(pairs_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(dataset)} samples ({len(pairs)} pairs)")
    print(f"Saved to {output_path} and {pairs_path}")
    
    # Print statistics
    module_counts = {}
    for sample in dataset:
        module = sample["target_module"]
        if module not in module_counts:
            module_counts[module] = {"positive": 0, "negative": 0}
        if sample["is_positive"]:
            module_counts[module]["positive"] += 1
        else:
            module_counts[module]["negative"] += 1
    
    print("\nDataset statistics:")
    for module, counts in module_counts.items():
        print(f"  {module}: {counts['positive']} positive, {counts['negative']} negative")

if __name__ == "__main__":
    import os
    # Use relative path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(os.path.dirname(script_dir), "dataset")
    
    dataset = generate_dataset()
    save_dataset(dataset, dataset_dir)