"""Scale Plan DAG construction and experiment execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import openai
from openai import APIConnectionError, APIError, OpenAI, OpenAIError, RateLimitError

from AI2ThorSim import get_ai2_thor_objects_randomized
from lamma_p import TaskManager, parse_arguments
from pddl_parser import Action, parse_domain


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


client = None


def get_openai_client() -> OpenAI:
    """Create the SDK client only after the configured API key is loaded."""
    global client
    if client is None:
        client = OpenAI(api_key=openai.api_key)
    return client

# ----------------- Domain-specific configuration -----------------
# Predicates to exclude from *matching* (logical-state) — geometric/sentinels:
LSL_BAN = {"inaction", "at",}

# State predicates used for logical-state matching.
LSL_WHITELIST = {
    "at-location", "holding", "switch-on", "switch-off",
    "object-open", "object-close", "break",
    "sliced", "cleaned", "fridge-open",
    "is-fridge",   # IMPORTANT: keep for matching (dependencies)
}

# Static/class predicates that must NEVER be padded into effects
STATIC_CLASS_PREDS = {"is-fridge"}

# module-level caches
_LSL_MATCH_PREDS: Set[str] = set()  # for matching (SR/RR)
_LSL_PAD_PREDS:   Set[str] = set()  # subset used for LP only

def _is_tuple_lit(lit) -> bool:
    return isinstance(lit, tuple) and len(lit) >= 1 and isinstance(lit[0], str)

def _pred(lit: Tuple) -> str:
    return lit[0] if _is_tuple_lit(lit) else None

def learn_dynamic_predicates_from_effects(actions: Dict[str, "Action"]) -> Set[str]:
    dyn: Set[str] = set()
    for a in actions.values():
        for lit in getattr(a, "effects_pos", []) + getattr(a, "effects_neg", []):
            if _is_tuple_lit(lit):
                dyn.add(_pred(lit))
    return dyn

def build_lsl_sets(actions: Dict[str, "Action"]) -> Tuple[Set[str], Set[str]]:
    """
    Returns (LSL_MATCH, LSL_PAD).
    LSL_MATCH = (dynamic ∪ whitelist) \ BAN
    LSL_PAD   = LSL_MATCH \ STATIC_CLASS_PREDS
    """
    dynamic = learn_dynamic_predicates_from_effects(actions)
    lsl_match = (dynamic | LSL_WHITELIST) - LSL_BAN
    lsl_pad   = lsl_match - STATIC_CLASS_PREDS
    return lsl_match, lsl_pad

def init_lsl(actions: Dict[str, "Action"]) -> None:
    """Call once after loading the domain."""
    global _LSL_MATCH_PREDS, _LSL_PAD_PREDS
    _LSL_MATCH_PREDS, _LSL_PAD_PREDS = build_lsl_sets(actions)

def is_logical_state_literal(lit) -> bool:
    """ONE-ARG checker used by matching code (SR/RR)."""
    if not _is_tuple_lit(lit):
        return False
    return _pred(lit) in _LSL_MATCH_PREDS

def is_paddable_lsl(lit) -> bool:
    """Used ONLY by LP to decide what can be added to effects."""
    if not _is_tuple_lit(lit):
        return False
    return _pred(lit) in _LSL_PAD_PREDS


def get_logical_state_literals(actions: Dict[str, "Action"]) -> Set[Tuple[str, ...]]:
    """Extract all logical state literals (pos & neg) from all actions."""
    LSL: Set[Tuple[str, ...]] = set()
    for act in actions.values():
        for l in getattr(act, "preconds_pos", []) + getattr(act, "preconds_neg", []) + \
                 getattr(act, "effects_pos", [])   + getattr(act, "effects_neg", []):
            if is_logical_state_literal(l):
                LSL.add(l)
    return LSL

def _lit_key(lit: Tuple[str, ...]) -> Tuple[str, int]:
    """('pred','?x','?y') -> ('pred', 2) for schema-level matching."""
    return (lit[0], len(lit) - 1)

def _filter_lsl(lits: List[Tuple[str, ...]]) -> Set[Tuple[str, int]]:
    """Keep only logical-state literals; return schema-level keys."""
    return {_lit_key(l) for l in lits if is_logical_state_literal(l)}

def _canon_sets(actions: Dict[str, "Action"]):
    """
    Canonical LSL-only sets per action: pre+, pre-, eff+, eff- as {(pred,arity)}.
    """
    pre_pos, pre_neg, eff_pos, eff_neg = {}, {}, {}, {}
    for a in actions.values():
        n = a.name
        pre_pos[n] = _filter_lsl(getattr(a, "preconds_pos", []))
        pre_neg[n] = _filter_lsl(getattr(a, "preconds_neg", []))
        eff_pos[n] = _filter_lsl(getattr(a, "effects_pos", []))
        eff_neg[n] = _filter_lsl(getattr(a, "effects_neg", []))
    return pre_pos, pre_neg, eff_pos, eff_neg

def _literal_padding(actions: Dict[str, "Action"]) -> Dict[str, "Action"]:
    """
    LP (paper): for each action a and each **paddable** LSL l in pre(a),
    if l ∉ eff(a) and (¬l) ∉ eff(a), add l to eff(a) with same polarity.
    NOTE: uses is_paddable_lsl (excludes STATIC_CLASS_PREDS like 'is-fridge').
    """
    padded = deepcopy(actions)
    for a in padded.values():
        pre_pos_lsl = [l for l in getattr(a, "preconds_pos", []) if is_paddable_lsl(l)]
        pre_neg_lsl = [l for l in getattr(a, "preconds_neg", []) if is_paddable_lsl(l)]

        eff_pos_set = set(getattr(a, "effects_pos", []))
        eff_neg_set = set(getattr(a, "effects_neg", []))

        # add positive literals that are missing from both eff+ and eff-
        for l in pre_pos_lsl:
            if l not in eff_pos_set and l not in eff_neg_set:
                a.effects_pos.append(l)
                eff_pos_set.add(l)

        # add negative literals that are missing from both eff+ and eff-
        for l in pre_neg_lsl:
            if l not in eff_pos_set and l not in eff_neg_set:
                a.effects_neg.append(l)
                eff_neg_set.add(l)
    return padded

def _exclude_free_actions(pre_pos, pre_neg, eff_pos, eff_neg) -> Set[str]:
    """Actions with no LSL in pre and eff should be excluded (free actions)."""
    free = set()
    for n in pre_pos.keys():
        if not pre_pos[n] and not pre_neg[n] and not eff_pos[n] and not eff_neg[n]:
            free.add(n)
    return free


def _strict_edges(actions: Dict[str, "Action"]) -> Set[Tuple[str, str]]:
    """
    SR: a2 -> a1 iff
      (i)  all LSL in pre(a1) ⊆ eff_padded(a2)
      (ii) ∃ l ∈ eff(a2) ∩ pre_before_padding(a1)   (non-vacuous overlap)
    """
    pre_pos_raw, pre_neg_raw, _, _ = _canon_sets(actions)
    actions_padded = _literal_padding(actions)
    pre_pos, pre_neg, eff_pos, eff_neg = _canon_sets(actions_padded)

    names = [a.name for a in actions.values()]
    free = _exclude_free_actions(pre_pos, pre_neg, eff_pos, eff_neg)
    names = [n for n in names if n not in free]

    edges_sr: Set[Tuple[str, str]] = set()
    for a1 in names:
        if not pre_pos[a1] and not pre_neg[a1]:
            continue
        for a2 in names:
            if a1 == a2:
                continue
            if pre_pos[a1].issubset(eff_pos[a2]) and pre_neg[a1].issubset(eff_neg[a2]):
                if (eff_pos[a2] & pre_pos_raw[a1]) or (eff_neg[a2] & pre_neg_raw[a1]):
                    edges_sr.add((a2, a1))
    return edges_sr

def _relaxed_edges_pure(actions: Dict[str, "Action"]) -> Set[Tuple[str, str]]:
    """
    RR-only (pure): a2->a1 iff pre(a1) ∩ eff(a2) ≠ ∅ (polarity-preserving), using padded eff.
    """
    pre_pos, pre_neg, eff_pos, eff_neg = _canon_sets(_literal_padding(actions))
    names = [a.name for a in actions.values()]
    free = _exclude_free_actions(pre_pos, pre_neg, eff_pos, eff_neg)
    names = [n for n in names if n not in free]

    edges: Set[Tuple[str, str]] = set()
    for a2 in names:
        for a1 in names:
            if a1 == a2:
                continue
            if (pre_pos[a1] & eff_pos[a2]) or (pre_neg[a1] & eff_neg[a2]):
                edges.add((a2, a1))
    return edges

def _relaxed_backfill_edges(actions: Dict[str, "Action"],
                            edges_sr: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """
    RR as SUPPLEMENT:
      - If a1 has NO SR predecessors: add a2->a1 if pre(a1) ∩ eff(a2) ≠ ∅
      - If a1 has NO SR successors:  add a1->a2 if pre(a2) ∩ eff(a1) ≠ ∅
    """
    pre_pos, pre_neg, eff_pos, eff_neg = _canon_sets(_literal_padding(actions))
    names = [a.name for a in actions.values()]
    free = _exclude_free_actions(pre_pos, pre_neg, eff_pos, eff_neg)
    names = [n for n in names if n not in free]

    sr_preds = {t for (_, t) in edges_sr}
    sr_succs = {s for (s, _) in edges_sr}

    no_sr_pred = [n for n in names if n not in sr_preds]
    no_sr_succ = [n for n in names if n not in sr_succs]

    edges_rr: Set[Tuple[str, str]] = set()

    # backfill predecessors (a2 -> a1)
    for a1 in no_sr_pred:
        for a2 in names:
            if a2 == a1:
                continue
            if (pre_pos[a1] & eff_pos[a2]) or (pre_neg[a1] & eff_neg[a2]):
                edges_rr.add((a2, a1))

    # backfill successors (a1 -> a2)
    for a1 in no_sr_succ:
        for a2 in names:
            if a1 == a2:
                continue
            if (pre_pos[a2] & eff_pos[a1]) or (pre_neg[a2] & eff_neg[a1]):
                edges_rr.add((a1, a2))

    return edges_rr - edges_sr

def build_dag_edges(
    actions: Dict[str, "Action"],
    mode: Literal["strict", "relaxed", "both"] = "both",
) -> List[Tuple[str, str]]:
    if mode == "strict":
        return sorted(_strict_edges(actions))
    if mode == "relaxed":
        return sorted(_relaxed_edges_pure(actions))
    sr = _strict_edges(actions)
    rr_backfill = _relaxed_backfill_edges(actions, sr)
    return sorted(sr | rr_backfill)


def build_stage1_prompt(task_desc: str, domain_contents: str, actions: Dict[str, Action], objects_ai_names: Dict[str, str]) -> str:
    prompt = "You are an expert in symbolic task planning for robots.\n"
    prompt += "Here are the contents of the PDDL domain file:\n"
    prompt += f"{domain_contents}\n"
    prompt += f"The task of the robots is to {task_desc}\n"
    prompt += f"You have access to the following objects in the AI2Thor environment: {', '.join(objects_ai_names.values())}.\n"
    prompt += "Your job is to identify all the actions and objects which may be needed to perform the task. \n"


    '''
    prompt += f"Given the task: \"{task_desc}\", identify which actions may be needed to perform the task.\n\n"
    prompt += "Here are the available actions in the domain:\n"

    for a in actions.values():
        # Filter out geometric preconds/effects for brevity
        pre = [str(p) for p in a.preconds_pos if is_logical_state_literal(p)]
        eff = [str(e) for e in a.effects_pos if is_logical_state_literal(e)]
        prompt += f"\nAction: {a.name}\n"
        prompt += f"  Parameters: {a.parameters}\n"
        prompt += f"  Preconditions: {pre}\n"
        prompt += f"  Effects: {eff}\n"
    '''

    prompt += (
        "\nPlease output only a list of action names and clearly identify its parameters from the objects except robot (e.g., `actions: [{action:\"turnon_vacuum\", parameters: {'robot':'robot', 'object':'vaccum|+00.12|+00.92|-00.06'}}, {action:\"clean_area\", parameters: {'robot':'robot', 'object':'vaccum|+00.12|+00.92|-00.06', 'location':'vaccum|+00.12|+00.92|-00.06'}}]`) "
        "that may be required to perform the task. MAKE SURE TO OUTPUT ONLY THE RAW JSON LIST, NO EXTRA TEXT.\n"
    )

    return prompt


def identify_leaf_actions(task_desc: str, domain_content: str, actions: Dict[str, Action], model: str, objects_ai_names: Dict[str, str]) -> List[str]:
    prompt = build_stage1_prompt(task_desc, domain_content, actions, objects_ai_names)

    response = get_openai_client().chat.completions.create(
    model=model,
    response_format={"type": "json_object"},  # or "json_schema" on newer SDKs
    messages=[
        {"role": "system", "content": "You are a symbolic planning assistant. Return only valid JSON."},
        {"role": "user", "content": prompt}
    ],)

    reply = response.choices[0].message.content.strip()

    try:
        leaf_actions = json.loads(reply)
        return leaf_actions
    except json.JSONDecodeError:
        print("[Error] Could not parse LLM response:", reply)
        return []


def build_adjacency(edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Return a map: child -> list of parents (reverse traversal)."""
    adj = defaultdict(list)
    for parent, child in edges:
        adj[child].append(parent)
    return adj


def dfs_action_chain(leaf_actions: List[str], edges: List[Tuple[str, str]]) -> List[str]:
    """
    Perform DFS from leaf actions, walking *incoming* edges (to ancestors),
    and return actions in executable order (predecessors before dependents).
    """
    # edges are (pred, succ). Build reverse adjacency: child -> [parents]
    parents: Dict[str, List[str]] = {}
    for u, v in edges:
        parents.setdefault(v, []).append(u)

    visited: Set[str] = set()
    ordered: List[str] = []

    def dfs(node: str):
        if node in visited:
            return
        visited.add(node)
        for p in parents.get(node, []):
            dfs(p)
        ordered.append(node)  # parents were appended first → executable order

    for a in leaf_actions:
        dfs(a)

    # No reverse here: 'ordered' is already topological (parents → child)
    return ordered


'''
def dfs_action_chain(leaf_actions: List[str], edges: List[Tuple[str, str]]) -> List[str]:
    """
    Perform DFS from leaf actions to collect all reachable ancestor actions.
    Returns actions in reverse topological order.
    """
    adj = build_adjacency(edges)
    visited = set()
    ordered = []

    def dfs(node: str):
        if node in visited:
            return
        visited.add(node)
        for parent in adj.get(node, []):
            dfs(parent)
        ordered.append(node)

    for a in leaf_actions:
        dfs(a)

    return list(reversed(ordered))  # topological-ish order
'''
    
def dfs_action_chain_modified(leaf_actions: List[str], edges: List[Tuple[str, str]]) -> List[List[str]]:
    """
    DFS from each leaf to root, collecting all ancestor chains.
    If all parents of a node are already on the current path (cycle), record the path.
    Filters out any chain that is a strict subset of another.
    """
    adj = build_adjacency(edges)
    all_chains: List[List[str]] = []

    def dfs(node: str, path: List[str], visited: Set[str]):
        if node in visited:
            return  # avoid immediate cycle on this path
        visited.add(node)
        path.append(node)

        parents = adj.get(node, [])

        # Only follow parents that are not yet on the current path
        next_parents = [p for p in parents if p not in visited]

        # Terminal if: (a) no parents, or (b) all parents would cause a cycle
        if not parents or not next_parents:
            # Record root->...->leaf order
            all_chains.append(list(reversed(path)))
        else:
            for parent in next_parents:
                dfs(parent, path, visited)

        path.pop()
        visited.remove(node)

    for leaf in leaf_actions:
        dfs(leaf, [], set())
    return remove_subset_chains(all_chains)



def remove_subset_chains(chains: List[List[str]]) -> List[List[str]]:
    """
    Removes chains that are strict subsets of other chains and drops exact duplicates.
    Keeps a single representative of identical chains (exact same sequence).
    """
    # Sort longest first; tie-break by lexicographic order for determinism
    chains_sorted = sorted(chains, key=lambda x: (-len(x), x))
    unique_chains: List[List[str]] = []
    unique_sets = []           # cache sets for fast subset checks
    seen_exact = set()         # exact sequence dedupe

    for chain in chains_sorted:
        t = tuple(chain)
        if t in seen_exact:
            continue  # exact duplicate

        s = set(chain)
        # strict subset of any already-kept chain?
        if any(s < us for us in unique_sets):
            continue

        unique_chains.append(chain)
        unique_sets.append(s)
        seen_exact.add(t)

    return unique_chains


def extract_required_parameters(action_names: List[str], actions: Dict[str, Action]) -> Dict[str, set]:
    """
    Extract param types from action definitions that use (var, type) tuples.
    """
    type_params = {}
    for name in action_names:
        action = actions[name]
        for var, typ in action.parameters:
            if typ not in type_params:
                type_params[typ.lower()] = set()
            type_params[typ].add(var)
    return type_params

def filter_environment(env: dict, required_types: Dict[str, set]) -> dict:
    """
    Filter environment to only include objects matching required types.
    """
    return {typ: env.get(typ, []) for typ in required_types.keys()}

def filter_environment_ai2thor(objects_ai, param_types, leaf_parameters):
    leaf_values_set = set()
    for values in leaf_parameters.values():
        for v in values:
            leaf_values_set.add(v.lower())
    
    filtered_objects_ai=[]
    for obj in objects_ai:
        if obj['objectId'].lower() in param_types.keys() or obj['objectId'].lower() in leaf_values_set:
            filtered_objects_ai.append(obj)
    return filtered_objects_ai



def save_problem_file(pddl_text: str, filepath: str):
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w") as f:
        f.write(pddl_text)
    
    print(f"[✔] Saved problem file to: {filepath}")



class PlanningError(RuntimeError):
    """Raised when the planner fails or returns no plan."""


class PDDLPlanner:
    """
    Thin wrapper around Fast-Downward.
    Example
    -------
    planner = PDDLPlanner("/opt/downward")          # folder that has fast-downward.py
    plan = planner.run_plan("domain.pddl", "problem.pddl")
    """

    def __init__(self, downward_root: str):
        self.downward_root = Path(downward_root).expanduser().resolve()

        self.fd_script = self.downward_root / "fast-downward.py"
        if not self.fd_script.exists():
            raise FileNotFoundError(f"fast-downward.py not found in {self.fd_script}")

    # --------------------------------------------------------------------- #
    def run_plan(
        self,
        domain_file: str,
        problem_file: str,
        *,
        alias: str = "seq-opt-lmcut",
        search: Optional[str] = None,
        plan_out_dir: str = "plans",
        timeout: Optional[int] = None,
    ) -> List[str]:
        """
        Execute Fast-Downward and return the plan as a list of action strings.
        """
        domain_file = str(Path(domain_file).resolve())
        problem_file = str(Path(problem_file).resolve())
        problem_stem = Path(problem_file).stem

        cmd: List[str] = [sys.executable, str(self.fd_script)]
        if search is None:
            cmd += ["--alias", alias]
        else:
            cmd += ["--search", search]
        cmd += [domain_file, problem_file]

        plan_directory = Path(plan_out_dir).resolve()
        plan_directory.mkdir(parents=True, exist_ok=True)

        # Fast Downward always writes ``sas_plan`` in its working directory.
        # Isolate each invocation so parallel experiments cannot overwrite it.
        with tempfile.TemporaryDirectory(prefix="fast-downward-") as work_directory:
            result = subprocess.run(
                cmd,
                cwd=work_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
            fd_plan = Path(work_directory) / "sas_plan"

            if result.returncode == 0 and fd_plan.is_file():
                final_plan_path = plan_directory / f"{problem_stem}.plan"
                fd_plan.replace(final_plan_path)
            else:
                final_plan_path = None

        out_path = plan_directory / f"{problem_stem}_planner.log"
        err_path = plan_directory / f"{problem_stem}_planner.err"
        out_path.write_text(result.stdout, encoding="utf-8")
        err_path.write_text(result.stderr, encoding="utf-8")

        if result.returncode != 0:
            raise PlanningError(
                f"Fast-Downward exited with code {result.returncode}.\nSee {err_path} for details."
            )

        if final_plan_path is None:
            raise PlanningError("Planner finished but no sas_plan file was produced.")

        # Read actions from plan
        with final_plan_path.open() as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        plan = [ln for ln in lines if not ln.startswith(";") and not ln.startswith("cost")]

        return plan

def get_task_name(domain_path: Path) -> str:
    """Extract problem name from the domain path."""
    if "task1/" in domain_path:
        return "clean floor"
    elif "task4/" in domain_path:
        return "serve water"
    elif "task6/" in domain_path:
        return "set table"
    elif "task9/" in domain_path:
        return "wash plate"
    elif "task10/" in domain_path:
        return "serve coke"
    elif "task11/" in domain_path:
        return "prepare food"
    else:
        raise ValueError(f"Unknown task in domain path: {domain_path}")


class ScalePlanTaskManager(TaskManager):
    """Manages task processing for SPS (Single-Path Synthesis) tasks."""

    def __init__(self, base_path: str, gpt_version: str, api_key_file: str, prompt_decompse_set: str = "pddl_train_task_decomposesep", prompt_allocation_set: str = "pddl_train_task_allocationsep", verbose: bool = True, task_type: str = "simple_tasks", method: str = "Scale_Plan"):
        # Initialize the SPS task manager.
        super().__init__(base_path, gpt_version, api_key_file, prompt_decompse_set, prompt_allocation_set, verbose, task_type, method)
        # Step 1: Parse and preprocess domain
        self.get_domain_and_content()
    
    
    def get_domain_and_content(self) -> Tuple[Dict[str, Action], str]:
        """Parse and return the domain and its content.
        """
        allaction_domain_path = os.path.join(self.resources_path, "allactionrobot.pddl")                    
        self.domain = parse_domain(allaction_domain_path)
        if self.verbose:
            print("Domain:", self.domain)
        self.domain_content = self.file_processor.read_file(allaction_domain_path)
        # 1) configure LSL once
        init_lsl(self.domain.actions)
        return
    
    def create_DAG_and_filter_environment(self, task: str, objects_ai: List[Dict], objects_ai_names: Dict, plot_DAG: bool = False) -> None:
        """Create DAG for the task and filter environment objects.
        """
        edges_strict = build_dag_edges(self.domain.actions, mode="strict")
        edges_relaxed = build_dag_edges(self.domain.actions, mode="relaxed")
        edges_both = build_dag_edges(self.domain.actions, mode="both")
        edges = edges_both  # for later use
        
        all_nodes = list(self.domain.actions.keys())  # ensure this exists

        def make_union_graph(all_nodes, *edge_sets):
            G = nx.DiGraph()
            G.add_nodes_from(all_nodes)
            for es in edge_sets:
                G.add_edges_from(es)
            return G
        # One stable position for all subplots (union graph)
        G_union = make_union_graph(all_nodes, edges_strict, edges_relaxed, edges_both)
        # Try Graphviz 'dot' if available; else spring
        try:
            from networkx.drawing.nx_agraph import graphviz_layout
            pos = graphviz_layout(G_union, prog="dot")
        except Exception:
            pos = nx.spring_layout(G_union, seed=42, k=0.5)

        def plot_graph(ax, edges, title, all_nodes=None, show_all_nodes=False, pos=None):
            G = nx.DiGraph()
            if show_all_nodes and all_nodes is not None:
                G.add_nodes_from(all_nodes)
            G.add_edges_from(edges)
            nx.draw(
                G, pos, ax=ax,
                with_labels=True, node_size=2500, node_color='lightblue',
                font_size=9, font_weight='bold', arrowsize=18
            )
            ax.set_title(title)

        if plot_DAG:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f"DAGs for Task: '{task}'", fontsize=16)
            plot_graph(axes[0], edges_strict,  "Strict Edges Only",        all_nodes, True, pos)
            plot_graph(axes[1], edges_relaxed, "Relaxed Edges Only",       all_nodes, True, pos)
            plot_graph(axes[2], edges_both,    "Strict + Relaxed Edges",   all_nodes, True, pos)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Step 2: Task → leaf actions (via LLM)
        leaf_actions = identify_leaf_actions(task, self.domain_content, self.domain.actions, model=self.gpt_version, objects_ai_names=objects_ai_names)
        # if self.verbose:
        leaf_actions_list=[]
        leaf_parameters = {}
        actions_data = leaf_actions.get("actions", leaf_actions)
        for a in actions_data:
            leaf_actions_list.append(a['action'])
            params = a.get("parameters", {})
            for k, v in params.items():
                if k not in leaf_parameters:
                    leaf_parameters[k] = set()
                leaf_parameters[k].add(v)


        # Step 3: DAG traversal → all required actions
        action_chain = dfs_action_chain(leaf_actions_list, edges)
        param_types = extract_required_parameters(action_chain, self.domain.actions)
        #if self.verbose:
        #    for t, vars in param_types.items():

        # Step 4: Load and filter environment
        self.filtered_env = filter_environment_ai2thor(objects_ai, param_types, leaf_parameters)
        filtered_env_names = {obj['objectType'].lower(): obj['objectId'] for obj in self.filtered_env}
        self.filtered_objects_category_str = f"objects = {list(filtered_env_names.keys())}"
        
        if self.verbose:
            print("Filtered environment for this task:", filtered_env_names)
            print("\n")

        self.objects_ai_category_str = f"objects = {list(objects_ai_names.keys())}"
        
        self.filtered_env_names_str = f"objects = {filtered_env_names}"                    
        
        self.objects_ai= self.filtered_objects_category_str   #Store the filtered objects here
        return

    def generate_LLM_PDDL(self, task_desc: str, filtered_env: dict, domain_content: str, robots: List[dict] = None) -> str:
        """Build a LLM prompt to generate a valid PDDL problem file with domain file injected."""
        
        prompt = f"""You are a symbolic planner working with PDDL.

        Your job is to generate a valid PDDL problem files for the task: {task_desc}.

        The available robots are: {robots}

        Below is the domain file being used. You must use ONLY the predicates, types, and object structures defined in this domain:

        --- START OF DOMAIN FILE ---
        {domain_content}
        --- END OF DOMAIN FILE ---

        The relevant objects currently available in the environment are: {filtered_env}

        The tasks may require either one or multiple problem files based on the required robots and their skills assigned to subtasks.

        Your output must be a complete PDDL problem file like this:

        (define (problem <problem-name 1>)
        (:domain <domain-name>)
        (:objects
            ...  ; object declarations grouped by type
        )
        (:init
            ...  ; initial state facts using valid predicates
        )
        (:goal
            ...  ; valid goal conditions using defined predicates
        )
        )

        .
        .
        .

        (define (problem <problem-name N>)
        (:domain <domain-name>)
        (:objects
            ...  ; object declarations grouped by type
        )
        (:init
            ...  ; initial state facts using valid predicates
        )
        (:goal
            ...  ; valid goal conditions using defined predicates
        )
        )
        Instructions:
        - Use only the predicates/types from the domain file above and available robots
        - Do not invent any predicates or types
        - Be consistent with arity and ordering
        - Output ONLY the PDDL code (no commentary)
        """
        
        if "gpt" not in self.gpt_version:
            _, text = self.llm.query_model(prompt, self.gpt_version, max_completion_tokens=1000, stop=["def"])
        else:
            messages = [{"role": "system", "content": "You are a Robot PDDL problem Expert"},
                    {"role": "user", "content": prompt}]
            _, text = self.llm.query_model(messages, self.gpt_version, max_completion_tokens=1400)
        return text

    def process_Scale_Plan_tasks(self, test_tasks: List[str], available_robots: List[dict], objects_ai: List[Dict], objects_ai_names: Dict, plot_DAG: bool = False, use_DAG: bool = True) -> None:
        """Process a list of tasks.
        
        Args:
            test_tasks (List[str]): List of tasks to process
            available_robots (List[dict]): List of available robots for tasks
            objects_ai (List[Dict]): List of available objects information
        """
        try:
            self.decomposed_plan = []
            self.allocated_plan = []
            #self.code_plan = []
            self.combined_plan = []
            self.code_planpddl = []
            self.create_log_folders(test_tasks)

            # Process each task
            for task_idx, (task, robots) in enumerate(zip(test_tasks, available_robots)):
                print(f"\n{'='*50}")
                print(f"Processing Task: {task}: {task_idx + 1}/{len(test_tasks)}")
                print(f"{'='*50}")
                if self.verbose:    
                    print(f"Using robots: {[r['name'] for r in robots]}")
                try:
                    if not use_DAG:
                        self.filtered_objects_category_str = f"objects = {list(objects_ai_names.keys())}"
                        filtered_env_names = {obj['objectType'].lower(): obj['objectId'] for obj in objects_ai}
                        self.filtered_env_names_str = f"objects = {filtered_env_names}"
                
                    start_time = time.time()
                    # Create DAG and filter environment
                    if use_DAG:
                        self.create_DAG_and_filter_environment(task, objects_ai, objects_ai_names, plot_DAG)

                    # Step 5: Build prompt for LLM to generate problem.pddl
                    decomposed_plan = self._generate_decomposed_plan(task, self.domain_content, robots, self.filtered_objects_category_str)  #Changed from self.objects_ai_category_str to self.filtered_objects_category_str
                    self.decomposed_plan.append(decomposed_plan)
                    if self.verbose:
                        if decomposed_plan:
                            print("✓ Decomposed plan generated")
                        else:
                            print("✗ Failed to generate decomposed plan")
                    allocated_plan = self._generate_allocation_plan(decomposed_plan, robots, self.filtered_objects_category_str) #Changed from self.objects_ai_category_str to self.filtered_objects_category_str
                    self.allocated_plan.append(allocated_plan)
                    if self.verbose:
                        if allocated_plan:
                            print("✓ Allocation plan generated")
                        else:
                            print("✗ Failed to generate allocation plan")
                    problem_summary = self._generate_problem_summary(decomposed_plan, allocated_plan, robots)
                    if self.verbose:
                        if problem_summary:
                            print("✓ Problem summary generated")
                        else:
                            print("✗ Failed to generate problem summary")

                    code_plan = self._generate_problem_files(robots, problem_summary, self.filtered_objects_category_str)
                    self.code_plan.append(code_plan) 


                    combined_plan = self._combine_all_plans(task_idx, allocated_plan, "")  #This combines and corrects all plans
                    self.combined_plan.append(combined_plan)
                    if self.verbose:
                        print("✓ Plans combined")

                    # Match references and store final PDDL plan
                    matched_plan = self._match_references_for_plan(combined_plan, self.filtered_env_names_str)
                    self.code_planpddl.append(matched_plan)
                    if self.verbose:
                        print("✓ References matched")
                    end_time = time.time()
                    self.planning_times.append(end_time - start_time)

                    # Calculate completion rate
                    #tc, total = self.planner.calculate_completion_rate(self.log_folders[task_idx], task_idx, "validated_subtask")
                    tc, total =0, 0
                    self.tc.append(tc)
                    self.total_subtasks.append(total)
                    #if self.verbose:
                                    

                except (OpenAIError, APIError, RateLimitError, APIConnectionError) as e:
                    print(f"OpenAI-related error while processing task '{task}': {e}")
                    continue  # Skip this task but continue with others
                except Exception as task_error:
                    print(f"Unexpected error while processing task '{task}': {task_error}")
                    continue
            
            print(f"\n{'='*50}")
            print(f"All {len(test_tasks)} tasks processed")
            print(f"{'='*50}")

        except Exception as e:
            print(f"Error processing tasks: {str(e)}")
            raise


    def process_LLM_as_Planner_tasks(self, test_tasks: List[str], available_robots: List[dict], objects_ai: List[Dict], objects_ai_names: Dict, plot_DAG: bool = False, use_DAG: bool = True) -> None:
        """Process a list of tasks.
        
        Args:
            test_tasks (List[str]): List of tasks to process
            available_robots (List[dict]): List of available robots for tasks
            objects_ai (List[Dict]): List of available objects information
        """
        try:
            self.combined_plan = []
            self.code_planpddl = []
            self.create_log_folders(test_tasks)

            # Process each task
            for task_idx, (task, robots) in enumerate(zip(test_tasks, available_robots)):
                print(f"\n{'='*50}")
                print(f"Processing Task: {task}: {task_idx + 1}/{len(test_tasks)}")
                print(f"{'='*50}")
                if self.verbose:    
                    print(f"Using robots: {[r['name'] for r in robots]}")
                try:
                    if not use_DAG:
                        self.filtered_objects_category_str = f"objects = {list(objects_ai_names.keys())}"
                        filtered_env_names = {obj['objectType'].lower(): obj['objectId'] for obj in objects_ai}
                        self.filtered_env_names_str = f"objects = {filtered_env_names}"
                
                    start_time = time.time()
                    # Create DAG and filter environment
                    if use_DAG:
                        self.create_DAG_and_filter_environment(task, objects_ai, objects_ai_names, plot_DAG)

                    combined_plan = self._generate_llm_as_planner_plan(task, self.domain_content, robots, self.filtered_objects_category_str)
                    self.combined_plan.append(combined_plan)
                    if self.verbose:
                        print("✓ Plans combined")

                    # Match references and store final PDDL plan
                    matched_plan = self._match_references_for_plan(combined_plan, self.filtered_env_names_str)
                    self.code_planpddl.append(matched_plan)
                    if self.verbose:
                        print("✓ References matched")
                    end_time = time.time()
                    self.planning_times.append(end_time - start_time)

                    # Calculate completion rate
                    #tc, total = self.planner.calculate_completion_rate(self.log_folders[task_idx], task_idx, "validated_subtask")
                    tc, total =0, 0
                    self.tc.append(tc)
                    self.total_subtasks.append(total)
                    #if self.verbose:
                                    

                except (OpenAIError, APIError, RateLimitError, APIConnectionError) as e:
                    print(f"OpenAI-related error while processing task '{task}': {e}")
                    continue  # Skip this task but continue with others
                except Exception as task_error:
                    print(f"Unexpected error while processing task '{task}': {task_error}")
                    continue
            
            print(f"\n{'='*50}")
            print(f"All {len(test_tasks)} tasks processed")
            print(f"{'='*50}")

        except Exception as e:
            print(f"Error processing tasks: {str(e)}")
            raise
    
    def process_LLM_PDDL_tasks(self, test_tasks: List[str], available_robots: List[dict], objects_ai: List[Dict], objects_ai_names: Dict, plot_DAG: bool = False, use_DAG: bool = True) -> None:
        """Process a list of tasks.
        
        Args:
            test_tasks (List[str]): List of tasks to process
            available_robots (List[dict]): List of available robots for tasks
            objects_ai (List[Dict]): List of available objects information
        """
        try:
            self.create_log_folders(test_tasks)

            # Process each task
            for task_idx, (task, robots) in enumerate(zip(test_tasks, available_robots)):
                print(f"\n{'='*50}")
                print(f"Processing Task: {task}: {task_idx + 1}/{len(test_tasks)}")
                print(f"{'='*50}")
                if self.verbose:    
                    print(f"Using robots: {[r['name'] for r in robots]}")
                try:
                    
                    if not use_DAG:
                        self.filtered_objects_category_str = f"objects = {list(objects_ai_names.keys())}"
                        filtered_env_names = {obj['objectType'].lower(): obj['objectId'] for obj in objects_ai}
                        self.filtered_env_names_str = f"objects = {filtered_env_names}"
                
                    start_time = time.time()
                    # Create DAG and filter environment
                    if use_DAG:
                        self.create_DAG_and_filter_environment(task, objects_ai, objects_ai_names, plot_DAG)


                    code_plan = self.generate_LLM_PDDL(
                        task_desc=task,
                        filtered_env=self.filtered_objects_category_str,
                        domain_content=self.domain_content,
                        robots=robots
                    )
                    self.code_plan.append(code_plan) 

                    if self.verbose:
                        if code_plan:
                            print("✓ Problem files generated")
                        else:
                            print("✗ Failed to generate problem files")
                    
                    self.file_processor.split_pddl_tasks(task_idx, self.log_folders[task_idx], code_plan, False, None)
                    self._validate_and_plan(task_idx)

                    combined_plan = self._combine_all_plans(task_idx, "", "validated_subtask")  #This combines and corrects all plans
                    self.combined_plan.append(combined_plan)
                    if self.verbose:
                        print("Combined Plans:\n", combined_plan, "\n")
                        print("✓ Plans combined")

                    # Match references and store final PDDL plan
                    matched_plan = self._match_references_for_plan(combined_plan, self.filtered_env_names_str)
                    self.code_planpddl.append(matched_plan)
                    if self.verbose:
                        print("✓ References matched")
                    end_time = time.time()
                    self.planning_times.append(end_time - start_time)

                    # Calculate completion rate
                    tc, total = self.planner.calculate_completion_rate(self.log_folders[task_idx], task_idx, "validated_subtask")
                    self.tc.append(tc)
                    self.total_subtasks.append(total)
                    #if self.verbose:
                                    

                except (OpenAIError, APIError, RateLimitError, APIConnectionError) as e:
                    print(f"OpenAI-related error while processing task '{task}': {e}")
                    continue  # Skip this task but continue with others
                except Exception as task_error:
                    print(f"Unexpected error while processing task '{task}': {task_error}")
                    continue
            
            print(f"\n{'='*50}")
            print(f"All {len(test_tasks)} tasks processed")
            print(f"{'='*50}")

        except Exception as e:
            print(f"Error processing tasks: {str(e)}")
            raise  

def _task_type_from_suffix(file_suffix: str) -> str:
    if not file_suffix:
        return "simple_tasks"
    return "vague_tasks" if "vague" in file_suffix else "complex_tasks"


def main() -> int:
    args = parse_arguments()
    task_manager = ScalePlanTaskManager(
        base_path=str(REPOSITORY_ROOT),
        gpt_version=args.gpt_version,
        api_key_file=args.openai_api_key_file,
        prompt_decompse_set=args.prompt_decompse_set,
        prompt_allocation_set=args.prompt_allocation_set,
        verbose=args.verbose,
        task_type=_task_type_from_suffix(args.file_suffix),
        method=args.method,
    )

    test_file = (
        REPOSITORY_ROOT
        / "data"
        / args.test_set
        / f"FloorPlan{args.floor_plan}{args.file_suffix}.json"
    )
    test_tasks, available_robots, ground_truth = task_manager.load_dataset(
        str(test_file)
    )

    if args.verbose:
        print(f"Test tasks ({len(test_tasks)}): {test_tasks}")
        print(
            "Available robots:",
            [[robot["name"] for robot in task_robots] for task_robots in available_robots],
        )

    objects_ai = get_ai2_thor_objects_randomized(args.floor_plan)
    objects_ai_names = {obj["objectType"]: obj["objectId"] for obj in objects_ai}

    processors = {
        "Scale_Plan": (task_manager.process_Scale_Plan_tasks, True),
        "LLM_as_Planner": (task_manager.process_LLM_as_Planner_tasks, False),
        "LLM_PDDL": (task_manager.process_LLM_PDDL_tasks, False),
    }
    if args.method not in processors:
        print(f"Unsupported method for Scale_Plan_DAG.py: {args.method}", file=sys.stderr)
        return 2

    processor, use_dag = processors[args.method]
    processor(
        test_tasks,
        available_robots,
        objects_ai,
        objects_ai_names,
        plot_DAG=False,
        use_DAG=use_dag,
    )

    if args.log_results:
        for index, task in enumerate(test_tasks):
            task_manager.log_results(
                task=task,
                idx=index,
                available_robots=available_robots,
                gt_test_tasks=ground_truth,
                floor_plan=args.floor_plan,
                objects_ai=objects_ai,
            )

    print("All tasks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    
