"""LLM-assisted task decomposition, PDDL generation, and planning pipeline."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import openai
import ai2thor
import ai2thor.controller
from ai2thor.platform import CloudRendering

from security_utils import (
    resolve_within,
    resolve_within_any,
    sanitize_log_value,
    validate_filename,
    validate_identifier,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import resources.robots as robots

# Constants
DEFAULT_MAX_COMPLETION_TOKENS = 3000
DEFAULT_RETRY_DELAY = 20
MAX_RETRIES = 3

class PDDLError(Exception):
    """Base exception class for PDDL-related errors."""
    pass

class ValidationError(PDDLError):
    """Exception raised for PDDL validation errors."""
    pass

class PlanningError(PDDLError):
    """Exception raised for PDDL planning errors."""
    pass

class LLMError(Exception):
    """Exception raised for Language Model related errors."""
    pass

class PDDLUtils:
    """Utility functions for PDDL operations."""
    
    @staticmethod
    def convert_to_dict_objprop(objs: List[str], obj_mass: List[float]) -> List[Dict[str, Union[str, float]]]:
        """Convert object list to dictionary format with mass.
        
        Args:
            objs (List[str]): List of object names
            obj_mass (List[float]): List of object masses
            
        Returns:
            List[Dict[str, Union[str, float]]]: List of dictionaries containing object properties
        """
        return [{'name': obj, 'mass': mass} for obj, mass in zip(objs, obj_mass)]
    
    @staticmethod
    def get_ai2_thor_objects(floor_plan: int) -> List[Dict[str, Any]]:
        """Get objects from AI2Thor environment.
        
        Args:
            floor_plan (int): Floor plan number
            
        Returns:
            List[Dict[str, Any]]: List of objects with their properties
        """
        controller = None
        try:
            controller = ai2thor.controller.Controller(scene=f"FloorPlan{floor_plan}", platform=CloudRendering)
            obj = [obj["objectType"] for obj in controller.last_event.metadata["objects"]]
            obj_mass = [obj["mass"] for obj in controller.last_event.metadata["objects"]]
            return PDDLUtils.convert_to_dict_objprop(obj, obj_mass)
        finally:
            if controller:
                controller.stop()

class FileProcessor:
    """Handles file operations and text processing for PDDL files.
    
    This class manages reading, writing, and processing of PDDL files and related
    text content. It provides methods for file operations and text manipulation
    specific to PDDL task processing.
    """
    
    def __init__(self, base_path: str):
        """Initialize the file processor.
        
        Args:
            base_path (str): Base path for file operations
        """
        self.base_path = str(Path(base_path).expanduser().resolve())
        self.each_run_path = os.path.join(self.base_path, "resources", "each_run")
        os.makedirs(self.each_run_path, exist_ok=True)
    
    def read_file(self, file_path: str) -> str:
        """Read contents of a file.
        
        Args:
            file_path (str): Path to the file to read
            
        Returns:
            str: Contents of the file
        """
        try:
            safe_path = resolve_within(self.base_path, file_path, must_exist=True, file_only=True)
            return safe_path.read_text(encoding="utf-8")
        except OSError as error:
            raise PDDLError(f"Could not read file {file_path}: {error}") from error
    
    def write_file(self, file_path: str, content: str) -> None:
        """Write content to a file.
        
        Args:
            file_path (str): Path to write to
            content (str): Content to write
        """
        try:
            safe_path = resolve_within(self.base_path, file_path)
            safe_path.write_text(content, encoding="utf-8")
        except OSError as error:
            raise PDDLError(f"Could not write file {file_path}: {error}") from error
    
    def split_pddl_tasks(self, task_id: int, log_folder: str, code_plan: Union[str, List[str]], isValidated: bool, problem_file_name: str) -> None:
        """Split PDDL tasks and save them to files.
        
        Args:
            code_plan (List[str]): List of PDDL plans to split
        """
        try:
            # Convert single plan to list
            if isinstance(code_plan, str):
                code_plan = [code_plan]
            
            # Create timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp_directory = os.path.join(self.each_run_path, timestamp)
            os.makedirs(timestamp_directory, exist_ok=True)
            
            for i, plan in enumerate(code_plan):
                tasks = re.split(r"\s*\(define\s*\(problem", plan)
                
                for j, task in enumerate(tasks[1:]):
                    task = "(define (problem" + task
                    task = self.balance_parentheses(task)

                    match = re.search(r'\(\s*problem\s+([^\s)]+)\)', task, re.IGNORECASE)
                    if match:
                        task_name = validate_filename(
                            match.group(1), field="PDDL problem name"
                        )
                        filename = f"{i+1}_{j+1}_{task_name}.pddl"
                    else:
                        print("Warning: Could not extract problem name, using timestamped filename.")
                        print("Task content:", task)
                        # avoid blocking in non-interactive runs:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"{i+1}_{j+1}_{timestamp}.pddl"
                    
                    # Save to both timestamped directory and generated_subtask
                    filepath = os.path.join(timestamp_directory, filename)
                    self.write_file(filepath, task)
                    
                    # Also save to generated_subtask for compatibility
                    if isValidated:
                        if problem_file_name:
                            filename = validate_filename(
                                problem_file_name,
                                suffix=".pddl",
                                field="problem filename",
                            )
                        
                        validated_subtask_dir = os.path.join(log_folder, "validated_subtask")
                        os.makedirs(validated_subtask_dir, exist_ok=True)
                        subtask_filepath = os.path.join(validated_subtask_dir, filename)
                        self.write_file(subtask_filepath, task)
                    else:
                        subtask_path = os.path.join(log_folder, "generated_subtask") 
                        os.makedirs(subtask_path, exist_ok=True)
                        subtask_filepath = os.path.join(subtask_path, filename)
                        self.write_file(subtask_filepath, task)
                    
            
        except Exception as e:
            raise PDDLError(f"Error splitting PDDL tasks: {str(e)}")
    
    def balance_parentheses(self, content: str) -> str:
        """Balance parentheses in PDDL content.
        
        Args:
            content (str): PDDL content to process
            
        Returns:
            str: Processed PDDL content with balanced parentheses
        """
        open_count = 0
        start_index = -1
        end_index = -1
        
        for i, char in enumerate(content):
            if char == '(':
                if open_count == 0:
                    start_index = i
                open_count += 1
            elif char == ')':
                open_count -= 1
                if open_count == 0:
                    end_index = i
                    break
        
        if start_index != -1 and end_index != -1:
            return content[start_index:end_index+1]
        return ""

    def split_and_store_tasks(
        self,
        content: str,
        llm: Optional['LLMHandler'] = None,
        gpt_version: Optional[str] = None
    ) -> Tuple[List[str], str]:
        """Split and store tasks from content.

        Returns:
            Tuple[List[str], str]: (subtasks list, sequence operations)
        """
        sequence_operations = ""

        # 1) Extract problem_summary and sequence_operations
        summary_match = re.search(
            r'(?:#?\s*)?Problem\s*content\s*summary\s*:?(.*?)(?=(?:#?\s*)?Sequence\s*of\s*Operations?\s*:?)',
            content, re.DOTALL | re.IGNORECASE
        )
        if summary_match:
            problem_summary = summary_match.group(1).strip()
        else:
            print("Warning: Could not extract problem content summary, attempting fallback.")
            split_marker = re.search(r'(?:#?\s*)?Sequence\s*of\s*Operations?\s*:', content, re.IGNORECASE)
            if split_marker:
                problem_summary = content[:split_marker.start()].strip()
                sequence_operations = content[split_marker.end():].strip()
            else:
                problem_summary = content.strip()
                sequence_operations = "failed to extract2"

        # 2) FIX: robust subtask extraction — only capture blocks that start with a SubTask header
        # Supports "#SubTask 1:" / "# SubTask 2:" / "SubTask 3:" (case-insensitive)
        subtask_block_re = re.compile(
            r'(?im)^\s*#?\s*Sub\s*Task\s*\d+\s*:\s*.*?(?=^\s*#?\s*Sub\s*Task\s*\d+\s*:\s*|\Z)',
            re.DOTALL
        )
        subtasks = [m.group(0).strip() for m in subtask_block_re.finditer(problem_summary)]
        #if subtasks:
        # 2a) If no SubTask header found, treat the whole thing as one subtask (safe fallback)
        if not subtasks:
            subtasks = [problem_summary.strip()] if problem_summary.strip() else []


        # 3) Verify/repair structure (unchanged except for using a robust header matcher)
        fixed_subtasks = []
        structure_ok = re.compile(
            r'\*\*Assigned\s*Robots?\*\*\s*:\s*.*?\n\*\*Objects\s*Involved\*\*\s*:\s*.*',
            re.DOTALL | re.IGNORECASE
        )
        fallback_plain = re.compile(
            r'\bAssigned\s*Robots?\b\s*:\s*.*?\n\bObjects\s*Involved\b\s*:\s*.*',
            re.DOTALL | re.IGNORECASE
        )

        for subtask in subtasks:

            if structure_ok.search(subtask) or fallback_plain.search(subtask):

                fixed_subtasks.append(subtask)
                continue

            if llm and gpt_version:
                fix_prompt = (
                    "The following subtask description needs to be reformatted. Please reformat it to strictly follow this structure:\n\n"
                    "#SubTask [number]: [Task Name]\n\n"
                    "# Initial Precondition analyze due to previous subtask:\n"
                    "#1. [precondition description]\n\n"
                    "[Action descriptions with Parameters, Preconditions, and Effects]\n\n"
                    "**Assigned Robot**: [list of robot numbers]\n"
                    "**Objects Involved**: [list of objects]\n\n"
                    "Important formatting rules:\n"
                    "1. Each section must start with the exact headers shown above\n"
                    "2. The order must be: SubTask header, Preconditions, Action descriptions, Assigned Robot, Objects Involved\n"
                    "3. Use '**Assigned Robot**:' and '**Objects Involved**:' exactly as shown with double asterisks\n"
                    "4. Include all action descriptions with their Parameters, Preconditions, and Effects\n"
                    "5. Keep the original action descriptions if they exist\n\n"
                    "Original subtask:\n" + subtask + "\n\n"
                    "Please provide ONLY the reformatted version following the structure above. Do not add any explanations or additional text."
                )

                if "gpt" not in gpt_version:
                    _, fixed_subtask = llm.query_model(
                        fix_prompt, gpt_version, stop=["def"])
                else:
                    messages = [
                        {"role": "system", "content": "You are a Robot PDDL problem Expert. Your task is to reformat subtask descriptions to match a specific structure. Do not add any explanations or additional text."},
                        {"role": "user", "content": fix_prompt}
                    ]
                    _, fixed_subtask = llm.query_model(messages, gpt_version)


                if structure_ok.search(fixed_subtask) or fallback_plain.search(fixed_subtask):
                    fixed_subtasks.append(fixed_subtask)
                else:
                    print("LLM structure fix failed, using original subtask")
                    fixed_subtasks.append(subtask)
            else:
                print("LLM handler not provided, using original subtask")
                fixed_subtasks.append(subtask)

        return fixed_subtasks, sequence_operations
   
    def extract_domain_name(self, problem_file_path: str) -> Optional[str]:
        """Extract the domain name from a problem PDDL file.
        
        Args:
            problem_file_path (str): Path to the problem PDDL file
            
        Returns:
            Optional[str]: Domain name if found, None otherwise
        """
        try:
            domain_pattern = re.compile(r'\(\s*:domain\s+(\S+)\s*\)')
            content = self.read_file(problem_file_path)
            match = domain_pattern.search(content)
            return match.group(1) if match else None
        except Exception as e:
            print(f"Error extracting domain name from {problem_file_path}: {str(e)}")
            return None

    def find_domain_file(self, domain_name: str) -> Optional[str]:
        """Find the domain file for a given domain name.
        
        Args:
            domain_name (str): Name of the domain to find
            
        Returns:
            Optional[str]: Path to domain file if found, None otherwise
        """
        try:
            safe_domain = validate_identifier(domain_name, field="PDDL domain name")
            domain_path = resolve_within(
                Path(self.base_path) / "resources",
                f"{safe_domain}.pddl",
            )
            return str(domain_path) if domain_path.is_file() else None
        except Exception as e:
            print(f"Error finding domain file for {domain_name}: {str(e)}")
            return None

    def clean_directory(self, directory_path: str) -> None:
        """Clean a directory by removing all files and subdirectories.
        
        Args:
            directory_path (str): Path to directory to clean
        """
        safe_directory = resolve_within(self.base_path, directory_path)
        if safe_directory.exists():
            for filename in os.listdir(safe_directory):
                file_path = safe_directory / filename
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    def extract_plan_from_output(self, content: str) -> str:
            if not content or not isinstance(content, str):
                raise ValueError("Invalid content provided to extract_plan_from_output")

            line_pattern = re.compile(
                r"""
                ^\s*
                (?:\d+\:\s*)?                 # optional step index like "0:"
                (?:[A-Za-z_][\w-]*)           # action name
                (?:\s+[^\s()]+)*              # zero or more arguments
                \s+\(\d+\)\s*$                # trailing cost in parentheses
                """,
                re.MULTILINE | re.VERBOSE
            )
            matches = line_pattern.findall(content)
            return "\n".join(m.strip() for m in matches) if matches else ""
    
class LLMHandler:
    """Handles interactions with Language Models (LLMs).
    

    """
    
    def __init__(self, api_key_file: str):
        """Initialize the LLM handler.
        
        Args:
            api_key_file (str): Path to the API key file
        """
        self.setup_api(api_key_file)
    
    def setup_api(self, api_key_file: str) -> None:
        """Configure OpenAI from the environment or a local key file."""
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        candidates = (Path(api_key_file), Path(f"{api_key_file}.txt"))

        if not api_key:
            for candidate in candidates:
                try:
                    api_key = candidate.read_text(encoding="utf-8").strip()
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise LLMError(f"Could not read API key file {candidate}: {error}") from error
                if api_key:
                    break

        if not api_key:
            raise LLMError(
                "Set OPENAI_API_KEY or provide a non-empty API key file with "
                "--openai-api-key-file."
            )
        openai.api_key = api_key
    
    def query_model(
        self, 
        prompt: Union[str, List[Dict]], 
        gpt_version: str, 
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        stop: Optional[List[str]] = None,
        logprobs: Optional[int] = 1
    ) -> Tuple[dict, str]:
        """the language model 
        
        Args:
            prompt: Either a string or a list of message dicts
            gpt_version: The model version to use
            max_completion_tokens: Maximum number of tokens in the response
            stop: Optional list of stop sequences
            logprobs: Optional number of logprobs to return
        
        Returns:
            Tuple of (full response object, generated text)
            
        """
        retry_delay = DEFAULT_RETRY_DELAY
        
        for attempt in range(MAX_RETRIES):
            try:
                if "gpt" not in gpt_version:
                    response = openai.completions.create(
                        model=gpt_version, 
                        prompt=prompt, 
                        max_completion_tokens=max_completion_tokens, 
                        stop=stop, 
                        logprobs=logprobs
                    )
                    return response, response.choices[0].text.strip()
                else:
                    response = openai.chat.completions.create(
                        model=gpt_version,
                        messages=prompt,
                        max_completion_tokens=max_completion_tokens,
                        reasoning_effort="low",
                    )
                    return response, response.choices[0].message.content.strip()
                    
            except openai.RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                raise LLMError("Rate limit exceeded")
                
            except (openai.APIError, openai.APITimeoutError) as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_delay)
                    continue
                raise LLMError(f"API Error after all retries: {str(e)}")
                
            except Exception as e:
                raise LLMError(f"Unexpected error in LLM query: {str(e)}")

class PDDLValidator:
    """Handles PDDL validation operations"""
    
    def __init__(self, llm_handler: LLMHandler, file_processor: FileProcessor):
        """Initialize the PDDL validator.
        
        Args:
            llm_handler (LLMHandler)
            file_processor (FileProcessor)
        """
        self.llm = llm_handler
        self.file_processor = file_processor
    
    def validate_problem(self, domain_file: str, problem_file: str, gpt_version: str) -> None:
        """Validate a PDDL problem file against its domain.


        """
        try:
            domain_content = self.file_processor.read_file(domain_file)
            problem_content = self.file_processor.read_file(problem_file)
            
            prompt = (
                f"Domain Description:\n{domain_content}\n\n"
                f"Problem Description:\n{problem_content}\n\n"
                "Validate the preconditions in problem file to ensure all precondition listed object "

            )
            
            if "gpt" not in gpt_version:
                _, validated_text = self.llm.query_model(
                    prompt=prompt,
                    gpt_version=gpt_version,
                    stop=["def"]
                )
            else:
                messages = [
                    {"role": "system", "content": "You are a Robot PDDL problem Expert"},
                    {"role": "user", "content": prompt}
                ]
                _, validated_text = self.llm.query_model(messages, gpt_version)
            
            # Save the validated content back to the problem file
            self.file_processor.write_file(problem_file, validated_text)
            
        except Exception as e:
            raise ValidationError(f"Error validating PDDL problem: {str(e)}")

class PDDLPlanner:
    
    def __init__(self, base_path: str, file_processor: FileProcessor):
        """Initialize the PDDL planner.

        """
        self.base_path = str(Path(base_path).expanduser().resolve())
        self.file_processor = file_processor
        configured_planner = os.environ.get(
            "FAST_DOWNWARD_PATH",
            os.path.join(self.base_path, "downward", "fast-downward.py"),
        )
        planner_path = Path(configured_planner).expanduser().resolve()
        if planner_path.name != "fast-downward.py":
            raise PlanningError("FAST_DOWNWARD_PATH must select fast-downward.py")
        if not planner_path.is_file():
            raise PlanningError(f"Fast Downward executable not found: {planner_path}")
        self.planner_path = str(planner_path)

    def _execute(self, domain_file: str, problem_file: str) -> subprocess.CompletedProcess:
        """Run Fast Downward with fixed input names in an isolated directory."""
        with tempfile.TemporaryDirectory(prefix="fast-downward-") as work_directory:
            work_path = Path(work_directory)
            shutil.copyfile(domain_file, work_path / "domain.pddl")
            shutil.copyfile(problem_file, work_path / "problem.pddl")
            return subprocess.run(
                [
                    sys.executable,
                    self.planner_path,
                    "--alias",
                    "seq-opt-lmcut",
                    "domain.pddl",
                    "problem.pddl",
                ],
                cwd=work_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
                check=False,
            )
    
    def run_plan(self, domain_file: str, problem_file: str) -> None:
        """
        
        Args:
            domain_file (str)
            problem_file (str)
    
        """
        try:
            domain_path = str(resolve_within(
                Path(self.base_path) / "resources",
                domain_file,
                must_exist=True,
                file_only=True,
            ))
            problem_path = str(resolve_within_any(
                [Path(self.base_path) / "logs", Path(self.base_path) / "resources" / "each_run"],
                problem_file,
                must_exist=True,
                file_only=True,
            ))
            result = self._execute(domain_path, problem_path)
            
            # Save the plan output
            output_file = str(Path(problem_path).with_suffix("")) + "_plan.txt"
            self.file_processor.write_file(output_file, result.stdout)
            
            if result.stderr:
                print(f"Warnings/Errors for {problem_file}:", result.stderr)
                
        except Exception as e:
            raise PlanningError(f"Error running PDDL planner: {str(e)}")
    
    def calculate_completion_rate(self, log_folder: str, task_id: int, plan_folder: str) -> Tuple[int, int]:
        """
        
        Returns:
            Tuple[int, int]: (number of completed tasks, total number of tasks)
        """
        TC = 0
        total_subtasks = 0
        
        try:
            for file_path in glob.glob(os.path.join(log_folder, plan_folder, '*_plan.txt')):
                total_subtasks += 1
                content = self.file_processor.read_file(file_path)
                TC += content.count('Solution found!')
                
            return TC, total_subtasks
            
        except Exception as e:
            raise PlanningError(f"Error calculating completion rate: {str(e)}")

class TaskManager:
    """Manages task processing and coordination.
    
    This is the main orchestrator class that coordinates all operations
 result logging.
    """
    
    def __init__(self, base_path: str, gpt_version: str, api_key_file: str, prompt_decompse_set: str = "pddl_train_task_decomposesep", prompt_allocation_set: str = "pddl_train_task_allocationsep", verbose: bool = False, task_type: str = "simple_tasks", method: str = "lammap"):
        """Initialize the task manager.
        
        Args:
            base_path (str): Base path for all operations
            gpt_version (str): Version of GPT to use
            api_key_file (str): Path to the API key file
            prompt_decompse_set (str): Name of the decomposition prompt set
            prompt_allocation_set (str): Name of the allocation prompt set
        """
        
        self.base_path = str(Path(base_path).expanduser().resolve())
        self.verbose = verbose
        self.gpt_version = gpt_version
        self.prompt_decompse_set = prompt_decompse_set
        self.prompt_allocation_set = prompt_allocation_set
        self.task_type = task_type
        self.method = method

        # Initialize paths
        self.resources_path = os.path.join(self.base_path, "resources")
        logs_root = os.path.join(self.base_path, "logs", f"logs_{method}")

        self.logs_path_simple = os.path.join(logs_root, "simple_tasks")
        self.logs_path_complex = os.path.join(logs_root, "complex_tasks")
        self.logs_path_vague = os.path.join(logs_root, "vague_tasks")
                
        os.makedirs(self.logs_path_simple, exist_ok=True)
        os.makedirs(self.logs_path_complex, exist_ok=True)
        os.makedirs(self.logs_path_vague, exist_ok=True)
        
        # Initialize components
        self.llm = LLMHandler(api_key_file)
        self.file_processor = FileProcessor(self.base_path)
        self.validator = PDDLValidator(self.llm, self.file_processor)
        self.planner = PDDLPlanner(self.base_path, self.file_processor)
        

        
        # Clean generated subtask directory
        self.clean_generated_subtask_directory(False)
        self.clean_generated_subtask_directory(True)
        
        # Initialize result storage
        self.decomposed_plan: List[str] = []
        self.allocated_plan: List[str] = []
        self.code_plan: List[str] = []
        self.validated_plan: List[str] = []
        self.combined_plan: List[str] = []
        self.code_planpddl: List[str] = []
        self.sequence_operations: str = ""  # Initialize sequence_operations
        self.tc: List[int] = []
        self.total_subtasks: List[int] = []
        self.planning_times: List[float] = []
        
        # Get action mapping from actions module
        
        # Initialize objects_ai as None, will be set in process_tasks
        self.objects_ai = None

    def clean_generated_subtask_directory(self, isValidated: bool = False) -> None:
        """Clean validated or generated subtask directories with numbered suffixes."""
        try:
            # Decide the prefix based on validation flag
            prefix = "validated_subtask_" if isValidated else "generated_subtask_"

            # Look inside resources_path and find all dirs starting with the prefix
            for dirname in os.listdir(self.resources_path):
                dirpath = os.path.join(self.resources_path, dirname)
                if os.path.isdir(dirpath) and dirname.startswith(prefix):
                    try:
                        shutil.rmtree(dirpath)
                        print(f"Deleted directory: {dirpath}")
                    except Exception as e:
                        print(f"Error deleting {dirpath}: {e}")
        except Exception as e:
            print(f"Error accessing {self.resources_path}: {e}")
        
    def load_dataset(self, test_file: str) -> Tuple[List[str], List[List[dict]], List[str], List[int], List[int]]:
        """Load dataset from JSON file.
        
        Args:
            test_file (str): Path to the test file
            
        """
        test_tasks = []
        robots_test_tasks = []
        gt_test_tasks = []
        
        try:
            with open(test_file, "r", encoding="utf-8") as file:
                for line in file:
                    values = list(json.loads(line).values())
                    test_tasks.append(values[0])
                    robots_test_tasks.append(values[1])
                    gt_test_tasks.append(values[2])
            

            # Prepare robot configurations
            available_robots = []
            #all_robots_names = [rob['name'] for rob in robots.robots]
            for robots_list in robots_test_tasks:
                task_robots = []
                for i, r_id in enumerate(robots_list):
                    rob = copy.deepcopy(robots.robots[r_id-1])  # Direct reference like original
                    rob['name'] = f'robot{i+1}'  # Use f-string for consistency
                    task_robots.append(rob)
                available_robots.append(task_robots)
            
            return test_tasks, available_robots, gt_test_tasks
            
        except FileNotFoundError:
            raise PDDLError(f"Test file not found: {test_file}")
        except json.JSONDecodeError as e:
            raise PDDLError(f"Error parsing JSON in test file: {str(e)}")
        except Exception as e:
            raise PDDLError(f"Error loading dataset: {str(e)}")
    
    def create_log_folders(self, tasks: List[str]) -> str:
        self.log_folders = []
        for task in tasks:
            """Create a log folder for a given task."""
            date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            task_name = re.sub(r"[^A-Za-z0-9_-]+", "_", "_".join(task.split()))
            task_name = task_name.strip("_")[:160] or "task"
            folder_name = f"{task_name}_plans_{date_time}"
            if self.task_type == "simple_tasks":
                log_folder = os.path.join(self.logs_path_simple, folder_name)
            elif self.task_type == "complex_tasks":
                log_folder = os.path.join(self.logs_path_complex, folder_name)
            elif self.task_type == "vague_tasks":
                log_folder = os.path.join(self.logs_path_vague, folder_name)
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")
            os.makedirs(log_folder, exist_ok=True)
            self.log_folders.append(log_folder)
        return
    
    def log_results(self, task: str, idx: int, available_robots: List[dict], 
                   gt_test_tasks: List[str], floor_plan: str, objects_ai: str):
        
        
        try:
                
            if self.code_planpddl and idx < len(self.code_planpddl):
                self._write_plan(self.log_folders[idx], "code_planpddl.py", self.code_planpddl[idx])
            if self.combined_plan and idx < len(self.combined_plan):
                self._write_plan(self.log_folders[idx], "combined_plan.py", self.combined_plan[idx])
            if self.decomposed_plan and idx < len(self.decomposed_plan):
                self._write_plan(self.log_folders[idx], "decomposed_plan.py", self.decomposed_plan[idx])
            if self.allocated_plan and idx < len(self.allocated_plan):
                self._write_plan(self.log_folders[idx], "allocated_plan.py", self.allocated_plan[idx])
            if self.code_plan and idx < len(self.code_plan):
                code_plan_content = self.code_plan[idx]
                code_plan = ""
                for c_plan in code_plan_content:
                    code_plan += c_plan + "\n\n"    
                self._write_plan(self.log_folders[idx], "code_plan.py", code_plan)
            if self.validated_plan and idx < len(self.validated_plan):
                self._write_plan(self.log_folders[idx], "validated_plan.py", self.validated_plan[idx])
            
            planning_time = self.planning_times[idx]
            if self.verbose:
                print(f"Writing plans for task {idx + 1}")


            log_path = resolve_within(
                Path(self.base_path) / "logs",
                Path(self.log_folders[idx]) / "log.txt",
            )
            with log_path.open('w', encoding="utf-8") as f:
                f.write(sanitize_log_value(task))
                f.write(f"\n\nGPT Version: {sanitize_log_value(self.gpt_version)}")
                f.write(f"\n{sanitize_log_value(objects_ai)}")
                f.write(f"\nrobots = {sanitize_log_value(available_robots[idx])}")
                f.write(f"\nground_truth = {sanitize_log_value(gt_test_tasks[idx])}")
                f.write(f"\nfloor_plan = {sanitize_log_value(floor_plan)}")
                # f.write(f"\nTotalsuccesssubtask = {TC}")
                # f.write(f"\nTotalsubtask = {total_subtasks}")
                f.write(f"\nPlanningtime = {sanitize_log_value(planning_time)}")
            
            
        except Exception as e:
            print(f"Error writing plans for task {idx + 1}: {str(e)}")

    def _write_plan(self, folder: str, filename: str, content: Union[str, List]):
        """Write a plan to a file."""
        safe_folder = resolve_within(
            Path(self.base_path) / "logs", folder, directory_only=True
        )
        validate_filename(filename, field="plan filename")
        if isinstance(content, list):
            for i, item in enumerate(content):
                item_name = validate_filename(f"{filename}.{i}", field="plan filename")
                with (safe_folder / item_name).open('w', encoding="utf-8") as f:
                    f.write(str(item))
        else:
            with (safe_folder / filename).open('w', encoding="utf-8") as f:
                f.write(content)

    def process_lammap_tasks(self, test_tasks: List[str], available_robots: List[dict], objects_ai: str) -> None:
        """Process a list of tasks."""
        try:
            # Initial task count
            
            # Store objects_ai for use in other methods
            self.objects_ai = objects_ai
            
            # Initialize or reset result lists
            self.decomposed_plan = []
            self.allocated_plan = []
            self.code_plan = []
            self.validated_plan = []
            self.combined_plan = []
            self.code_planpddl = []
            
            # Get domain content
            allaction_domain_path = os.path.join(self.resources_path, "allactionrobot.pddl")
            self.domain_content = self.file_processor.read_file(allaction_domain_path)
            # Create log folders for each task
            self.create_log_folders(test_tasks)
            
            # Process each task
            for task_idx, (task, task_robots) in enumerate(zip(test_tasks, available_robots)):
                print(f"\n{'='*50}")
                print(f"Processing Task: {task}: {task_idx + 1}/{len(test_tasks)}")
                print(f"{'='*50}")
                
                # # Clean generated subtask directory before starting new task
                # self.clean_generated_subtask_directory()
            
                # Generate and store decomposed plan
                start_time = time.time()
                decomposed_plan = self._generate_decomposed_plan(
                    task, self.domain_content, task_robots, objects_ai
                )
                self.decomposed_plan.append(decomposed_plan)
                if self.verbose:
                    if decomposed_plan:
                        print("✓ Decomposed plan generated")
                    else:
                        print("✗ Failed to generate decomposed plan")

                # Generate and store allocation plan
                allocated_plan = self._generate_allocation_plan(
                    decomposed_plan, task_robots, objects_ai
                )
                self.allocated_plan.append(allocated_plan)
                
                if self.verbose:
                    if allocated_plan:
                        print("✓ Allocation plan generated")
                    else:
                        print("✗ Failed to generate allocation plan")
                
                # Generate problem summary
                problem_summary = self._generate_problem_summary(
                    decomposed_plan, allocated_plan, task_robots
                )
                if self.verbose:
                    if problem_summary:
                        print("✓ Problem summary generated")
                    else:
                        print("✗ Failed to generate problem summary")

                
                # Generate and store problem files
                code_plan = self._generate_problem_files(
                    task_robots, problem_summary, self.objects_ai
                )
                self.code_plan.append(code_plan) 
                if self.verbose:
                    if code_plan:
                        print("✓ Problem files generated")
                    else:
                        print("✗ Failed to generate problem files")

                # Split into subtasks
                self.file_processor.split_pddl_tasks(task_idx, self.log_folders[task_idx], code_plan, False, None)
                
                # Validate and plan
                self._validate_and_plan(task_idx)
                
                
                # Combine and process plans
                if self.method == "lammap_only_PDDL":
                    combined_plan = self._combine_all_plans(task_idx, "", "validated_subtask")
                else:
                    combined_plan = self._combine_all_plans(task_idx, allocated_plan, "validated_subtask")
                self.combined_plan.append(combined_plan)
                
                if self.verbose:
                    if combined_plan:
                        print("✓ Combined plan generated")
                    else:
                        print("✗ Failed to generate combined plan")

                # Match references and store final PDDL plan
                matched_plan = self._match_references_for_plan(combined_plan, objects_ai)
                self.code_planpddl.append(matched_plan)
                if self.verbose:
                    print("✓ References matched")

                end_time = time.time()
                self.planning_times.append(end_time - start_time)

                # Calculate completion rate
                fast_downward_tc, total = self.planner.calculate_completion_rate(self.log_folders[task_idx], task_idx, "validated_subtask") 
                self.tc.append(fast_downward_tc)
                self.total_subtasks.append(total)
                if self.verbose:
                    print(f"Task {task_idx + 1} Fast Downward completion rate: {fast_downward_tc}/{total}")
                
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"All {len(test_tasks)} tasks processed")
                print(f"{'='*50}")
            
        except Exception as e:
            print("\n[ERROR] Task Processing Failed:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Current task index: {task_idx if 'task_idx' in locals() else 'Not started'}")
            raise
    def _generate_llm_as_planner_plan(self, task: str, domain_content: str, robots: List[dict], objects_ai: str) -> str:
        """Generate decomposed plan for a task."""
        try:
            
            # Construct the prompt incrementally like the original
            prompt = f"from pddl domain file with all possible actions: \n{domain_content}\n\n"
            prompt += objects_ai
            prompt += f"\nrobots = {robots}\n\n"
            prompt += "robot initiate 'as not inaction robot '(which defaults location too)\n\n"
            prompt += "Generate a plan for the task using the Task description and the domain file in the time durative actions format, where parallel, tasks are performed at the same time. \
                IMPORTANT: all 'variablelocation' should be corrected to variable itself, since variable itself includes location. and result must be in PDDL plan format. Make sure to allocate robots based on their skills and mass capacity. \
                If the task can be decomposed into subtasks, decompose it and allocate robots to the subtasks. If the subtasks can be performed in parallel, indicate that in the plan. \
                The plan should be in a format that can be easily converted to PDDL later. Make sure to include all necessary details for each step of the plan, including which robot is performing which action and any relevant parameters for the actions."
            prompt += "\n Example: Task Description: Put Apple in the fridge and switch off the lights \n"
            prompt += "Plan:\n"
            prompt += "0.000: (gotoobject robot1 lightswitch) [1.000]\n"
            prompt += "1.000: (switchoff robot1 lightswitch) [1.000]\n"
            prompt += "2.000: (pickupobject robot1 apple) [1.000]\n"
            prompt += "3.000: (gotoobject robot1 fridge) [1.000]\n"
            prompt += "4.000: (putobject robot1 apple fridge) [1.000]\n"

            prompt += f"# Task Description: {task}"
            
            if "gpt" not in self.gpt_version:
                _, text = self.llm.query_model(prompt, self.gpt_version, stop=["def"])
            else:
                messages = [{"role": "user", "content": prompt}]
                _, text = self.llm.query_model(messages, self.gpt_version)
    
            return text
            
        except Exception as e:
            raise PDDLError(f"Error generating decomposed plan: {str(e)}")

    def _generate_decomposed_plan(self, task: str, domain_content: str, robots: List[dict], objects_ai: str) -> str:
        """Generate decomposed plan for a task."""
        try:
            # Read decomposition prompt file
            decompose_prompt_path = os.path.join(self.base_path, "scripts", "llm_prompts", f"{self.prompt_decompse_set}.txt")
            decompose_prompt = self.file_processor.read_file(decompose_prompt_path)
            
            # Construct the prompt incrementally like the original
            prompt = f"from pddl domain file with all possible actions: \n{domain_content}\n\n"
            prompt += objects_ai
            prompt += f"\nrobots = {robots}\n\n"
            prompt += "robot initiate 'as not inaction robot '(which defaults location too)\n\n"
            prompt += decompose_prompt
            prompt += "# GENERAL TASK DECOMPOSITION \n"
            prompt += "Decompose and parallel subtasks where ever possible.\n\n"
            prompt += f"# Task Description: {task}"
            
            if "gpt" not in self.gpt_version:
                _, text = self.llm.query_model(prompt, self.gpt_version, stop=["def"])
            else:
                messages = [{"role": "user", "content": prompt}]
                _, text = self.llm.query_model(messages, self.gpt_version)
    
            return text
            
        except Exception as e:
            raise PDDLError(f"Error generating decomposed plan: {str(e)}")
    
    def _generate_allocation_plan(self, decomposed_plan: str, robots: List[dict], objects_ai: str) -> str:
        """Generate allocation plan for decomposed tasks.
        
        """
        try:
            # Read allocation prompt file
            prompt_file = os.path.join(self.base_path, "scripts", "llm_prompts", f"{self.prompt_allocation_set}_solution.txt")
            with open(prompt_file, "r") as allocated_prompt_file:
                allocated_prompt = allocated_prompt_file.read()
            
            # Build prompt incrementally like the original
            prompt = "\n"
            prompt += allocated_prompt
            prompt += decomposed_plan
            prompt += "\n# TASK ALLOCATION"
            prompt += f"\n# Scenario: There are {len(robots)} robots available. The task should be performed using the minimum number of robots necessary. Robot should be assigned to subtasks that match its skills and mass capacity. Using your reasoning come up with a solution to satisfy all constraints."
            prompt += f"\n\nrobots = {robots}"
            prompt += f"\n{objects_ai}"
            prompt += "\n\n# IMPORTANT: The AI should ensure that the robots assigned to the tasks have all the necessary skills to perform the tasks. IMPORTANT: Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both and allocate robots based on availability."
            prompt += "\n\n#Important: Avoid ambiguous robot assignments such as: Assigned Robots: 'Any one of robot1 or robot2', 'team', 'group', 'any available robot', 'robot two'. Instead, specify exact robot assignments such as Assigned Robots: robot1, robot2"
            prompt += "\n# SOLUTION\n"
            
            # Handle different GPT versions like the original
            if "gpt" not in self.gpt_version:
                # older versions of GPT
                _, text = self.llm.query_model(prompt, self.gpt_version, stop=["def"])
            else:          
                messages = [{"role": "user", "content": prompt}]
                _, text = self.llm.query_model(messages, self.gpt_version)
            
            return text
            
        except Exception as e:
            raise PDDLError(f"Error generating allocation plan: {str(e)}")

    def _generate_problem_summary(self, decomposed_plans: Union[str, List[str]], allocated_plans: Union[str, List[str]], available_robots: Union[List[dict], List[List[dict]]]) -> List[str]:
        """Generate problem summaries from decomposed and allocated plans.
        

            
        Returns:
            List[str]: List of generated problem summaries
        """
        try:
            
            # Convert single items to lists
            if isinstance(decomposed_plans, str):
                decomposed_plans = [decomposed_plans]
            if isinstance(allocated_plans, str):
                allocated_plans = [allocated_plans]
            if not isinstance(available_robots[0], list):
                available_robots = [available_robots]
            
            # Read summary prompt file
            prompt_file = os.path.join(self.base_path, "scripts", "llm_prompts", f"{self.prompt_allocation_set}_summary.txt")
            with open(prompt_file, "r") as code_prompt_file:
                code_prompt = code_prompt_file.read()
            
            # Build base prompt once
            base_prompt = " finish the problem content summary strictly following the example format"
            base_prompt += "\n\n" + code_prompt + "\n\n"
            
            code_plan = []
            # Process each plan
            for i, (plan, solution) in enumerate(zip(decomposed_plans, allocated_plans)):
                # Build prompt for this plan
                prompt = base_prompt + plan
                prompt += "\n# TASK ALLOCATION"
                prompt += f"\n\nrobots = {available_robots[i]}"
                prompt += solution
                prompt += "\n# problem content summary  \n"
                
                if "gpt" not in self.gpt_version:
                    # older versions of GPT
                    _, text = self.llm.query_model(prompt, self.gpt_version, stop=["def"])
                else:            
                    # using variants of gpt 4 or 3.5
                    messages = [
                        {"role": "system", "content": "You are a Robot PDDL problem expert.\n"
                        "You MUST produce perfectly formatted, syntactically correct text.\n"
                        "Do NOT introduce typos.\n"
                        "Do NOT insert extra spaces inside words or identifiers.\n"
                        "All action names, parameters, predicates, and objects must be spelled correctly.\n"},
                        {"role": "user", "content": prompt}
                    ]
                    _, text = self.llm.query_model(messages, self.gpt_version)
                
                code_plan.append(text)
            
            return code_plan
            
        except Exception as e:
            raise PDDLError(f"Error generating problem summary: {str(e)}")

    def _generate_problem_files(self, robots: List[Dict], problem_summary: Union[str, List[str]], objects_ai) -> List[str]:
        """Generate PDDL problem files from plans.
        
        """
        # Handle list input by taking first summary
        if isinstance(problem_summary, list):
            problem_summary = problem_summary[0]
        
        problem_pddl = []
        # Split into subtasks and sequence operations
        subtasks, sequence_operations = self.file_processor.split_and_store_tasks(
            problem_summary,
            llm=self.llm,
            gpt_version=self.gpt_version
        )



        # Store sequence operations for later use
        self.sequence_operations = sequence_operations
        
        # Process each subtask using the class method#
        problem_pddl = self.problemextracting(
            robots=robots,
            subtasks=subtasks,
            llm=self.llm,
            gpt_version=self.gpt_version,
            file_processor=self.file_processor,
            objects_ai=objects_ai,
            prompt_allocation_set=self.prompt_allocation_set,
            problem_summary=problem_summary
        )
        
        return problem_pddl
    
    def problemextracting(
            self,
            robots: List[Dict],
            subtasks: List[str],
            llm: 'LLMHandler',
            gpt_version: str,
            file_processor: 'FileProcessor',
            objects_ai: str,
            prompt_allocation_set: str,
            problem_summary: str
        ) -> List[str]:

        """Extract problem files from subtasks."""
        problem_pddl: List[str] = []
        # Robust matchers for the Assigned Robots block
        assigned_block_re = re.compile(
            r'(?is)\*\*Assigned\s*Robots?\*\*\s*:\s*(.*?)\n\*\*Objects\s*Involved\*\*\s*:',
            re.IGNORECASE | re.DOTALL
        )

        assigned_block_fallback = re.compile(
            r'(?is)\bAssigned\s*Robots?\b\s*:\s*(.*?)\n\bObjects\s*Involved\b\s*:',
            re.IGNORECASE | re.DOTALL
        )

        # Match "robot 1" and "robots 1"
        robot_num_re = re.compile(r'\brobots?\s*(\d+)\b', re.IGNORECASE)

        # Phrases that imply "pick a single robot among the listed ones"
        single_choice_phrase_re = re.compile(
            r'\b(any\s+one|either|one\s+of|choose\s+one|pick\s+one|any\s+robot\s+among|any\s+of|any\s+single|or)\b',
            re.IGNORECASE
        )

        for subtask in subtasks:
            m = assigned_block_re.search(subtask) or assigned_block_fallback.search(subtask)
            if not m:
                print("Invalid subtask structure, skipping")
                print("Subtask content:", subtask)
                print("✗ Failed to generate pddl plans for this subtask")
                continue
            # else:
            #     #print("m:", m)

            assigned_robots = m.group(1).strip()

            # Extract robot numbers
            robot_numbers = robot_num_re.findall(assigned_robots)

            # Fallback: capture bare numbers if "robot(s)" isn't repeated before each number
            if not robot_numbers:
                robot_numbers = re.findall(r'\b\d+\b', assigned_robots)

            if not robot_numbers:
                print(f"No robot numbers found in Assigned Robots: {assigned_robots}; subtask: {subtask} , attempting fallback extraction.")
                robot_numbers = []
            normalized_robot_numbers = [f"robot{num}" for num in robot_numbers]
            # Detect phrasing that indicates a single-choice (not team) among listed robots
            single_choice = bool(single_choice_phrase_re.search(assigned_robots))

            # Team detection
            is_team = ("team" in assigned_robots.lower()) or ("allactionrobot" in assigned_robots.lower())

            # Single-choice wording indicates a single-robot assignment.
            if single_choice and normalized_robot_numbers:
                # Deterministic selection policy: smallest robot number
                try:
                    smallest = min(int(n[len("robot"):]) for n in normalized_robot_numbers)
                    normalized_robot_numbers = [f"robot{smallest}"]
                except ValueError:
                    # Fallback to the first listed if parsing failed
                    normalized_robot_numbers = [normalized_robot_numbers[0]]
                is_team = False


            if (not single_choice) and (len(normalized_robot_numbers) > 1):
                is_team = True

            if is_team:
                # Team task: concatenate all team domains
                all_domain_contents = ""
                for robot in normalized_robot_numbers:
                    domain_path = os.path.join(self.base_path, "resources", f"{robot}.pddl")
                    domain_text = file_processor.read_file(domain_path)
                    if domain_text:
                        all_domain_contents += domain_text

                if not all_domain_contents:
                    print("✗ Failed to generate decomposed plan")
                    continue

                problem_fileexamplepath = os.path.join(
                    self.base_path, "scripts", "llm_prompts", f"{prompt_allocation_set}_teamproblem.txt"
                )
                problem_examplecontent = file_processor.read_file(problem_fileexamplepath) or ""

                prompt = (
                    "\n" + problem_examplecontent +
                    "Strictly follow the structure and finish the tasks like example\n"
                    "Subtask examination from action perspective:" + subtask +
                    "\nDomain file content:" + all_domain_contents +
                    "\n based on the objects available below." + objects_ai +
                    "Task description: Generate the problem files, based on the objects above, "
                    "the preconditions, actions, and subtask examination.\n"
                    "#IMPORTANT, strictly follow the structure, stop generating after the Problem file generation is done."
                )


                if "gpt" not in gpt_version:
                    _, text = llm.query_model(prompt, gpt_version, stop=["def"])
                else:
                    messages = [
                        {"role": "system", "content": "You are a Robot PDDL problem expert"},
                        {"role": "user", "content": prompt}
                    ]
                    _, text = llm.query_model(messages, gpt_version)

                problem_pddl.append(text)

            else:
                # Single-robot case
                if not normalized_robot_numbers:
                    print("No robot number found in Assigned Robot; skipping.")
                    print("Assigned Robots content:", assigned_robots)
                    print("✗ Failed to generate pddl plans for this subtask")
                    continue

                # Deterministic single-robot choice: if multiple remain, take smallest
                if len(normalized_robot_numbers) > 1:
                    try:
                        smallest = min(int(n[len("robot"):]) for n in normalized_robot_numbers)
                        normalized_robot_numbers = [f"robot{smallest}"]
                    except ValueError:
                        print("Warning: Could not parse robot numbers for single choice; defaulting to first listed.")
                        normalized_robot_numbers = [normalized_robot_numbers[0]]

                robotassignnumber = f"{normalized_robot_numbers[0].replace(' ', '')}.pddl"
                domain_path = os.path.join(self.base_path, "resources", robotassignnumber)


                domain_content = file_processor.read_file(domain_path) or ""
                if not domain_content:
                    print(f"Domain file not found or empty: {domain_path}")
                    print("✗ Failed to generate pddl plans for this subtask")
                    continue

                problem_fileexamplepath = os.path.join(
                    self.base_path, "scripts", "llm_prompts", f"{prompt_allocation_set}_problem.txt"
                )
                problem_examplecontent = file_processor.read_file(problem_fileexamplepath) or ""

                prompt = (
                    "\n" + problem_examplecontent +
                    " Finish the tasks like example\n"
                    "Subtask examination from action perspective:" + subtask +
                    "\nDomain file content:" + domain_content +
                    "\n based on the objects available for potential usage below.\n" + objects_ai +
                    "\nTask description: generate the problem file. Based on the objects above, "
                    "the domain file preconditions, actions, and subtask examination. "
                    "IMPORTANT the robot initiates strictly as not inaction and robot "
                    "(which includes location)\n"
                    "#IMPORTANT, strictly follow the structure, stop generating after the Problem file generation is done."
                )
                

                if "gpt" not in gpt_version:
                    _, text = llm.query_model(prompt, gpt_version, stop=["def"])
                else:
                    messages = [
                        {"role": "system", "content": "You are a Robot PDDL problem Expert"},
                        {"role": "user", "content": prompt}
                    ]
                    _, text = llm.query_model(messages, gpt_version)

                problem_pddl.append(text)

        return problem_pddl
    
    def _validate_and_plan(self, task_idx: int) -> None:
        """Validate and plan all problem files."""
        try:
            # First run LLM validator
            self.run_llmvalidator(task_idx)
            # Wait for validation to complete
            
            # Then run planners
            self.run_planners(task_idx, "validated_subtask") 
            
        except Exception as e:
            raise PDDLError(f"Error in validation and planning: {str(e)}")

    def run_llmvalidator(self, task_idx: int) -> None:
        """Run LLM validation on problem files."""
        try:
            problem_path = os.path.join(self.log_folders[task_idx], "generated_subtask")
            problem_files = [f for f in os.listdir(problem_path) if f.endswith('.pddl')]
            sorted_problem_files = sorted(problem_files)  # Ensure consistent order
            for problem_file in sorted_problem_files:
                try:
                    problem_file_full = os.path.join(problem_path, problem_file)
                    domain_name = self.file_processor.extract_domain_name(problem_file_full)
                    if not domain_name:
                        print(f"No domain specified in {problem_file}")
                        continue

                    domain_file = self.file_processor.find_domain_file(domain_name)
                    if not domain_file:
                        print(f"No domain file found for domain {domain_name}")
                        continue

                    domain_content = self.file_processor.read_file(domain_file)
                    problem_content = self.file_processor.read_file(problem_file_full)

                    prompt = (f"Domain Description:\n"
                            f"{domain_content}\n\n"
                            f"Problem Description:\n"
                            f"{problem_content}\n\n"
                            "Validate the preconditions and effects in problem file to ensure all precondition listed object "
                            "is included and also in domain file, and go over structure to check the parenthesis "
                            "and syntext. Also ensure correct number of predicate arguments as per the domain file. Check and return only the validated problem file. Important: In :init, ensure there is no (inaction robot)")

                    if "gpt" not in self.gpt_version:
                        _, text = self.llm.query_model(prompt, self.gpt_version, stop=["def"])
                    else:
                        messages = [{"role": "system", "content": "You are a Robot PDDL problem Expert"},
                                {"role": "user", "content": prompt}]
                        _, text = self.llm.query_model(messages, self.gpt_version)
                    self.validated_plan.append(text)
                    code_plan = [text]
                    self.file_processor.split_pddl_tasks(task_idx, self.log_folders[task_idx], code_plan, True, problem_file)
                    
                except Exception as e:
                    print(f"Error processing file {problem_file}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in run_llmvalidator: {str(e)}")
            raise

    def run_planners(self, task_idx: int, pddl_folder: str) -> None:
        """Run PDDL planners on problem files."""
        try:
            validate_filename(pddl_folder, field="PDDL folder")
            problem_path = resolve_within(
                Path(self.base_path) / "logs",
                Path(self.log_folders[task_idx]) / pddl_folder,
                must_exist=True,
                directory_only=True,
            )
            problem_files = [f for f in os.listdir(problem_path) if f.endswith('.pddl')]
            sorted_problem_files = sorted(problem_files)  # Ensure consistent order
            for problem_file in sorted_problem_files:
                try:
                    safe_problem_name = validate_filename(
                        problem_file,
                        suffix=".pddl",
                        field="problem filename",
                    )
                    problem_file_full = str(
                        resolve_within(
                            problem_path,
                            safe_problem_name,
                            must_exist=True,
                            file_only=True,
                        )
                    )
                    domain_name = self.file_processor.extract_domain_name(problem_file_full)
                    if not domain_name:
                        print(f"No domain specified in {problem_file}")
                        continue

                    domain_file = self.file_processor.find_domain_file(domain_name)
                    if not domain_file:
                        print(f"No domain file found for domain {domain_name}")
                        continue
                    domain_file = str(resolve_within(
                        self.resources_path,
                        domain_file,
                        must_exist=True,
                        file_only=True,
                    ))

                    result = self.planner._execute(domain_file, problem_file_full)
                    output_file = str(
                        problem_path / safe_problem_name.replace('.pddl', '_plan.txt')
                    )
                    self.file_processor.write_file(output_file, result.stdout)
                    if self.verbose:
                        if ("Solution found!" in result.stdout) or ("Plan found with cost" in result.stdout):
                            print(f"✓ Plan found for {problem_file}")
                        else:
                            print(f"✗ No plan found for {problem_file}, check {output_file} for details")

                    if result.stderr:
                        print(f"Warnings/Errors for {problem_file}:", result.stderr)
                        
                except subprocess.TimeoutExpired:
                    print(f"Planner timed out for {problem_file}")
                except Exception as e:
                    print(f"Error processing file {problem_file}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in run_planners: {str(e)}")
            raise

    def _combine_all_plans(self, task_id: str, allocated_plan: Union[str, List[str]], plan_folder: str) -> str:
        """Combine all generated plan files into a single plan.
 
        """
        # Handle list input by taking first plan
        if isinstance(allocated_plan, list):
            allocated_plan = allocated_plan[0]

        prompt = ""

        if plan_folder!= "":
            base_path = os.path.join(self.log_folders[task_id], plan_folder)
            plan_files = [f for f in os.listdir(base_path) if f.endswith('_plan.txt')]
            sorted_plan_files = sorted(plan_files)  # Ensure consistent order
            # Add plans from files if they exist
            if sorted_plan_files:
                for idx, filename in enumerate(sorted_plan_files):
                    filepath = os.path.join(base_path, filename)
                    content = self.file_processor.read_file(filepath)
                    plan = self.file_processor.extract_plan_from_output(content)
                    if plan:  # Only add non-empty plans
                        prompt += f"\nPlan {idx + 1}: \n{plan}\n"
                    else:
                        print(f"No valid plan found in {filename}, skipping.")
            else:
                print(f"No plan files found in {base_path}, using allocated plan directly.")

            if self.verbose:
                print("Plan from Fast Downward:\n", prompt)


        # Add allocation examination and initial plan
        if allocated_plan!= "":
            prompt += "\nallocation examination\n"
            prompt += self.sequence_operations if hasattr(self, 'sequence_operations') else ""
            prompt += "\ninitial plan examination\n"
            prompt += allocated_plan
        prompt += ("\nyou are robot allocation expert, Your task is, based on inital plan examination "
                  "and allocation examination correct the subplans. Then based on your understanding "
                  "merge the subtasks together by using timed durative actions format, where parallel "
                  "tasks are performed at the same time. IMPORTANT: all 'variablelocation' should be "
                  "corrected to variable itself, since variable itself includes location. and result "
                  "must be in PDDL plan format.")
        

        if "gpt" not in self.gpt_version:
            _, text = self.llm.query_model(prompt, self.gpt_version, stop=["def"])
        else:
            messages = [{"role": "user", "content": prompt}]
            _, text = self.llm.query_model(messages, self.gpt_version)
             
        return text

    def _match_references_for_plan(self, plan: str, objects_ai: str) -> str:

        """Match and correct variable locations in plan."""
        prompt = (
            f"{objects_ai}\n"
            "IMPORTANT: Your TASK is based on the provided pddl plan provided in the passage below "
            "and the object list above, modify and only modify the plan so that all 'variablelocation' "
            "should be corrected to variable itself, since variable itself includes location. and similarly variable names should be corrected to variable itself. "
            "IMPORTANT: the only parenthesis usage should be for the correct PDDL plan, no exception.\n\n"
            f"{plan}"
        )
        
        if "gpt" not in self.gpt_version:
            _, text = self.llm.query_model(prompt, self.gpt_version, stop=["def"])
        else:
            messages = [{"role": "user", "content": prompt}]
            _, text = self.llm.query_model(messages, self.gpt_version)
        
        return text


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options shared with the Scale Plan entry point."""
    parser = argparse.ArgumentParser(
        description="Generate and validate PDDL plans with an LLM."
    )
    parser.add_argument(
        "--floor-plan",
        type=int,
        required=True,
        help="AI2-THOR floor-plan number.",
    )
    parser.add_argument(
        "--file-suffix",
        default="",
        help="Suffix appended to the FloorPlan dataset filename.",
    )
    parser.add_argument("--openai-api-key-file", default="api_key")
    parser.add_argument(
        "--gpt-version",
        type=str,
        default="gpt-5.2",
    )
    parser.add_argument(
        "--prompt-decompse-set",
        type=str,
        default="pddl_train_task_decomposesep",
    )
    parser.add_argument(
        "--prompt-allocation-set",
        type=str,
        default="pddl_train_task_allocationsep",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="MAT2THOR",
        choices=['MAT2THOR']
    )
    parser.add_argument(
        "--log-results",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--method",
        default="LLM_as_Planner",
        choices=["Scale_Plan", "LLM_as_Planner", "LLM_PDDL", "lammap_only_PDDL", "lammap"],
        help="Planning method to use.",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]*", args.file_suffix):
        parser.error("--file-suffix contains unsupported characters")
    for field, value in (
        ("--prompt-decompse-set", args.prompt_decompse_set),
        ("--prompt-allocation-set", args.prompt_allocation_set),
    ):
        try:
            validate_identifier(value, field=field)
        except ValueError as error:
            parser.error(str(error))
    return args

def main() -> int:
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()

        task_type = ""
        if args.file_suffix=="":
            task_type = "simple_tasks"
        elif "vague" in args.file_suffix:
            task_type = "vague_tasks"
        else:
            task_type = "complex_tasks"
        
        # Initialize task manager
        task_manager = TaskManager(
            base_path=str(REPOSITORY_ROOT),
            gpt_version=args.gpt_version,
            api_key_file=args.openai_api_key_file,
            prompt_decompse_set=args.prompt_decompse_set,
            prompt_allocation_set=args.prompt_allocation_set,
            verbose=args.verbose,
            task_type=task_type,
            method=args.method
        )     
        # Original workflow
        test_file = REPOSITORY_ROOT / "data" / args.test_set / f"FloorPlan{args.floor_plan}{args.file_suffix}.json"
        test_tasks, available_robots, gt_test_tasks = \
            task_manager.load_dataset(str(test_file))
        
        if args.verbose:
            print(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")
        
        # Get AI2thor objects 
        objects_ai = f"\n\nobjects = {PDDLUtils.get_ai2_thor_objects(args.floor_plan)}"
        
        # Process tasks with objects_ai
        task_manager.process_lammap_tasks(test_tasks, available_robots, objects_ai)
        # Log results if enabled
        if args.log_results:
            for idx, task in enumerate(test_tasks):
                task_manager.log_results(
                    task=task,
                    idx=idx,
                    available_robots=available_robots,
                    gt_test_tasks=gt_test_tasks,
                    floor_plan=args.floor_plan,
                    objects_ai=objects_ai
                )
        
        return 0
    except Exception as error:
        print(f"Error in main execution: {error}", file=sys.stderr)
        print(f"Full error: {type(error).__name__}: {error}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())





























#####



    
