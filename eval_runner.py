#!/usr/bin/env python3
"""
State-RAG Evaluation Framework
Automated experiment runner for conference paper evaluation
"""

import json
import time
import subprocess
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics

@dataclass
class Task:
    """Represents a single evaluation task"""
    task_id: str
    name: str
    description: str
    complexity: str  # 'simple', 'medium', 'complex'
    requirements: List[str]
    ground_truth_path: str
    test_files: List[str]


@dataclass
class ExperimentResult:
    """Results from running one task with one system"""
    task_id: str
    system_name: str
    
    # Code correctness metrics
    functional_correct: bool
    compilation_success: bool
    test_pass_rate: float  # 0-1
    code_quality_score: float  # 0-100
    
    # Authority preservation metrics
    user_edits_preserved: int
    user_edits_total: int
    preservation_rate: float  # 0-1
    
    # Cost metrics
    tokens_used: int
    api_calls: int
    
    # Timing
    latency_seconds: float
    
    # Errors
    errors: List[str]


class EvaluationDataset:
    """
    Manages the evaluation dataset with tasks and ground truth
    """
    
    def __init__(self, dataset_path: str = "evaluation_dataset"):
        self.dataset_path = Path(dataset_path)
        self.tasks: List[Task] = []
        
    def create_dataset(self):
        """Create the evaluation dataset structure"""
        self.dataset_path.mkdir(exist_ok=True)
        
        # Create task directories
        for complexity in ['simple', 'medium', 'complex']:
            (self.dataset_path / complexity).mkdir(exist_ok=True)
        
        print(f"✅ Dataset structure created at {self.dataset_path}")
    
    def add_task(self, task: Task):
        """Add a task to the dataset"""
        self.tasks.append(task)
        
        # Create task directory
        task_dir = self.dataset_path / task.complexity / task.task_id
        task_dir.mkdir(exist_ok=True, parents=True)
        
        # Save task metadata
        with open(task_dir / "task.json", "w") as f:
            json.dump(asdict(task), f, indent=2)
        
        print(f"✅ Added task: {task.task_id} ({task.complexity})")
    
    def load_tasks(self) -> List[Task]:
        """Load all tasks from dataset"""
        tasks = []
        
        for complexity in ['simple', 'medium', 'complex']:
            complexity_dir = self.dataset_path / complexity
            if not complexity_dir.exists():
                continue
                
            for task_dir in complexity_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                    
                task_file = task_dir / "task.json"
                if task_file.exists():
                    with open(task_file) as f:
                        task_data = json.load(f)
                        tasks.append(Task(**task_data))
        
        return tasks


class SystemInterface:
    """
    Base interface for different LLM systems
    Each system (ChatGPT, LangChain, State-RAG) should implement this
    """
    
    def __init__(self, name: str):
        self.name = name
        self.token_count = 0
        self.api_calls = 0
    
    def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task and return generated code
        
        Returns:
            {
                'files': {'path/to/file.tsx': 'content', ...},
                'tokens_used': int,
                'api_calls': int,
                'time_seconds': float
            }
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset system state between tasks"""
        self.token_count = 0
        self.api_calls = 0


class StateRAGSystem(SystemInterface):
    """State-RAG implementation"""
    
    def __init__(self):
        super().__init__("State-RAG")
        from orchestrator import Orchestrator
        self.orchestrator = Orchestrator(llm_provider="gemini")
    
    def execute_task(self, task: Task) -> Dict[str, Any]:
        import tiktoken
        
        start_time = time.time()
        
        # Extract allowed paths from requirements
        allowed_paths = self._extract_paths(task.requirements)
        
        # Execute request
        artifacts = self.orchestrator.handle_request(
            user_request=task.description,
            allowed_paths=allowed_paths
        )
        
        # Convert artifacts to file dict
        files = {
            artifact.file_path: artifact.content
            for artifact in artifacts
        }
        
        # Count tokens (approximation)
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = sum(len(enc.encode(content)) for content in files.values())
        
        elapsed = time.time() - start_time
        
        return {
            'files': files,
            'tokens_used': tokens,
            'api_calls': 1,
            'time_seconds': elapsed
        }
    
    def _extract_paths(self, requirements: List[str]) -> List[str]:
        """Extract file paths from requirements"""
        paths = []
        for req in requirements:
            if ".tsx" in req or ".ts" in req or ".json" in req:
                # Simple heuristic - extract path-like strings
                words = req.split()
                for word in words:
                    if "/" in word or "." in word:
                        paths.append(word)
        return paths


class ConversationHistorySystem(SystemInterface):
    """Baseline: Standard ChatGPT with conversation history"""
    
    def __init__(self):
        super().__init__("Conversation-History")
        from llm_adapter import LLMAdapter
        self.llm = LLMAdapter(provider="gemini")
        self.history = []
    
    def execute_task(self, task: Task) -> Dict[str, Any]:
        import tiktoken
        
        start_time = time.time()
        
        # Build prompt with full history
        prompt = self._build_prompt(task)
        
        # Call LLM
        response = self.llm.generate(prompt)
        
        # Parse response
        files = self._parse_response(response)
        
        # Update history
        self.history.append({
            'request': task.description,
            'response': response
        })
        
        # Count tokens
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = len(enc.encode(prompt)) + len(enc.encode(response))
        
        elapsed = time.time() - start_time
        
        return {
            'files': files,
            'tokens_used': tokens,
            'api_calls': 1,
            'time_seconds': elapsed
        }
    
    def _build_prompt(self, task: Task) -> str:
        prompt_parts = ["You are a website builder AI."]
        
        # Add history
        for item in self.history:
            prompt_parts.append(f"User: {item['request']}")
            prompt_parts.append(f"Assistant: {item['response']}")
        
        # Add current request
        prompt_parts.append(f"User: {task.description}")
        prompt_parts.append("Assistant: Generate complete files.")
        
        return "\n\n".join(prompt_parts)
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse LLM response into files"""
        from llm_output_parser import parse_llm_output
        
        try:
            artifacts = parse_llm_output(response)
            return {a.file_path: a.content for a in artifacts}
        except:
            return {}
    
    def reset(self):
        super().reset()
        self.history = []


class Evaluator:
    """
    Main evaluation engine that runs experiments and computes metrics
    """
    
    def __init__(self, dataset: EvaluationDataset):
        self.dataset = dataset
        self.results: List[ExperimentResult] = []
    
    def run_experiment(
        self,
        systems: List[SystemInterface],
        tasks: List[Task],
        num_runs: int = 3
    ) -> List[ExperimentResult]:
        """
        Run full experiment: all systems on all tasks, multiple times
        """
        
        results = []
        
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Task: {task.name} ({task.complexity})")
            print(f"{'='*60}")
            
            for system in systems:
                print(f"\n  System: {system.name}")
                
                # Run multiple times for statistical significance
                task_results = []
                
                for run in range(num_runs):
                    print(f"    Run {run+1}/{num_runs}...", end=" ")
                    
                    try:
                        system.reset()
                        output = system.execute_task(task)
                        result = self._evaluate_output(
                            task, system.name, output
                        )
                        task_results.append(result)
                        print("✅")
                        
                    except Exception as e:
                        print(f"❌ {e}")
                        # Create failed result
                        task_results.append(ExperimentResult(
                            task_id=task.task_id,
                            system_name=system.name,
                            functional_correct=False,
                            compilation_success=False,
                            test_pass_rate=0.0,
                            code_quality_score=0.0,
                            user_edits_preserved=0,
                            user_edits_total=0,
                            preservation_rate=0.0,
                            tokens_used=0,
                            api_calls=0,
                            latency_seconds=0.0,
                            errors=[str(e)]
                        ))
                
                results.extend(task_results)
        
        self.results = results
        return results
    
    def _evaluate_output(
        self,
        task: Task,
        system_name: str,
        output: Dict[str, Any]
    ) -> ExperimentResult:
        """Evaluate system output against ground truth"""
        
        files = output['files']
        
        # 1. Check compilation
        compilation_success = self._check_compilation(files)
        
        # 2. Run tests
        test_pass_rate = self._run_tests(task, files)
        
        # 3. Check code quality
        code_quality = self._check_code_quality(files)
        
        # 4. Check functional correctness
        functional = compilation_success and test_pass_rate > 0.8
        
        return ExperimentResult(
            task_id=task.task_id,
            system_name=system_name,
            functional_correct=functional,
            compilation_success=compilation_success,
            test_pass_rate=test_pass_rate,
            code_quality_score=code_quality,
            user_edits_preserved=0,  # Requires separate test
            user_edits_total=0,
            preservation_rate=0.0,
            tokens_used=output['tokens_used'],
            api_calls=output['api_calls'],
            latency_seconds=output['time_seconds'],
            errors=[]
        )
    
    def _check_compilation(self, files: Dict[str, str]) -> bool:
        """Check if generated code compiles"""
        
        # Write files to temp directory
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Write all files
            for filepath, content in files.items():
                full_path = tmppath / filepath
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
            
            # Try to compile with TypeScript
            try:
                result = subprocess.run(
                    ['tsc', '--noEmit', '--jsx', 'react'],
                    cwd=tmppath,
                    capture_output=True,
                    timeout=10
                )
                return result.returncode == 0
            except:
                return False
    
    def _run_tests(self, task: Task, files: Dict[str, str]) -> float:
        """Run tests and return pass rate"""
        
        if not task.test_files:
            return 1.0  # No tests = assume pass
        
        # TODO: Implement test runner
        # For now, return 1.0 if compilation succeeds
        return 1.0 if self._check_compilation(files) else 0.0
    
    def _check_code_quality(self, files: Dict[str, str]) -> float:
        """Run linter and return quality score"""
        
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Write files
            for filepath, content in files.items():
                if filepath.endswith(('.ts', '.tsx', '.js', '.jsx')):
                    full_path = tmppath / filepath
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(content)
            
            # Run ESLint
            try:
                result = subprocess.run(
                    ['eslint', '.', '--format', 'json'],
                    cwd=tmppath,
                    capture_output=True,
                    timeout=10
                )
                
                # Parse results
                if result.stdout:
                    eslint_output = json.loads(result.stdout)
                    total_issues = sum(
                        len(file['messages'])
                        for file in eslint_output
                    )
                    
                    # Quality score: 100 - (issues * 5), min 0
                    return max(0, 100 - (total_issues * 5))
                
                return 100.0
                
            except:
                return 50.0  # Default if linting fails
    
    def compute_aggregate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute aggregate metrics per system"""
        
        by_system = {}
        
        for result in self.results:
            if result.system_name not in by_system:
                by_system[result.system_name] = []
            by_system[result.system_name].append(result)
        
        metrics = {}
        
        for system_name, results in by_system.items():
            metrics[system_name] = {
                'functional_correctness': statistics.mean(
                    r.functional_correct for r in results
                ),
                'compilation_success_rate': statistics.mean(
                    r.compilation_success for r in results
                ),
                'test_pass_rate': statistics.mean(
                    r.test_pass_rate for r in results
                ),
                'code_quality': statistics.mean(
                    r.code_quality_score for r in results
                ),
                'avg_tokens': statistics.mean(
                    r.tokens_used for r in results
                ),
                'avg_latency': statistics.mean(
                    r.latency_seconds for r in results
                ),
                # Standard deviations
                'functional_correctness_std': statistics.stdev(
                    [r.functional_correct for r in results]
                ) if len(results) > 1 else 0,
                'code_quality_std': statistics.stdev(
                    [r.code_quality_score for r in results]
                ) if len(results) > 1 else 0,
            }
        
        return metrics
    
    def save_results(self, output_path: str = "evaluation_results.json"):
        """Save all results to JSON"""
        
        output = {
            'results': [asdict(r) for r in self.results],
            'aggregate_metrics': self.compute_aggregate_metrics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    
    # 1. Create dataset
    dataset = EvaluationDataset()
    dataset.create_dataset()
    
    # 2. Add sample tasks
    dataset.add_task(Task(
        task_id="simple_001",
        name="Button Component",
        description="Create a simple React button component with onClick handler",
        complexity="simple",
        requirements=[
            "Create components/Button.tsx",
            "Accept onClick and children props",
            "Use Tailwind for styling"
        ],
        ground_truth_path="evaluation_dataset/simple/simple_001/ground_truth",
        test_files=[]
    ))
    
    dataset.add_task(Task(
        task_id="medium_001",
        name="Todo App",
        description="Create a todo app with TodoList, TodoItem, and AddTodo components",
        complexity="medium",
        requirements=[
            "Create components/TodoList.tsx",
            "Create components/TodoItem.tsx",
            "Create components/AddTodo.tsx",
            "Use local state management"
        ],
        ground_truth_path="evaluation_dataset/medium/medium_001/ground_truth",
        test_files=[]
    ))
    
    # 3. Create systems
    systems = [
        StateRAGSystem(),
        ConversationHistorySystem(),
    ]
    
    # 4. Run evaluation
    tasks = dataset.load_tasks()
    evaluator = Evaluator(dataset)
    
    print("\n" + "="*60)
    print("Starting Evaluation")
    print("="*60)
    
    results = evaluator.run_experiment(
        systems=systems,
        tasks=tasks[:2],  # Start with 2 tasks for testing
        num_runs=3
    )
    
    # 5. Save and display results
    evaluator.save_results()
    
    print("\n" + "="*60)
    print("Aggregate Metrics")
    print("="*60)
    
    metrics = evaluator.compute_aggregate_metrics()
    for system_name, system_metrics in metrics.items():
        print(f"\n{system_name}:")
        for metric_name, value in system_metrics.items():
            if not metric_name.endswith('_std'):
                print(f"  {metric_name}: {value:.3f}")