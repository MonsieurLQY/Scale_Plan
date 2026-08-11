"""Minimal PDDL domain parser used by Scale Plan's DAG construction."""

from __future__ import annotations

import re
from pathlib import Path


__all__ = ["Action", "Domain", "parse_domain"]

SExpression = str | list["SExpression"]
Literal = tuple[str, ...]

_TOKEN_PATTERN = re.compile(r"[()]|[^()\s]+")
_CONTROL_EXPRESSIONS = {"and", "not", "or", "when", "forall", "exists", "imply"}


class Action:
    """PDDL action data required by the dependency-graph builder."""

    __slots__ = (
        "name",
        "parameters",
        "preconds_pos",
        "preconds_neg",
        "effects_pos",
        "effects_neg",
        "raw_pre_tree",
        "raw_eff_tree",
    )

    def __init__(self, name: str):
        self.name = name
        self.parameters: list[tuple[str, str]] = []
        self.preconds_pos: list[Literal] = []
        self.preconds_neg: list[Literal] = []
        self.effects_pos: list[Literal] = []
        self.effects_neg: list[Literal] = []
        self.raw_pre_tree: SExpression | None = None
        self.raw_eff_tree: SExpression | None = None

    def __repr__(self) -> str:
        return (
            f"Action({self.name}, params={self.parameters}, "
            f"+pre={self.preconds_pos}, -pre={self.preconds_neg}, "
            f"+eff={self.effects_pos}, -eff={self.effects_neg})"
        )


class Domain:
    """Parsed domain metadata and actions."""

    def __init__(self, name: str):
        self.name = name
        self.types: list[str] = []
        self.predicates: list[tuple[str, list[str]]] = []
        self.actions: dict[str, Action] = {}

    def __repr__(self) -> str:
        return (
            f"Domain({self.name}, {len(self.types)} types, "
            f"{len(self.predicates)} predicates, {len(self.actions)} actions)"
        )


def _strip_comments(text: str) -> str:
    return re.sub(r";[^\n]*", "", text)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


def _read_expression(tokens: list[str], index: int = 0) -> tuple[SExpression, int]:
    if index >= len(tokens):
        raise ValueError("Unexpected end of PDDL input")

    token = tokens[index]
    if token == ")":
        raise ValueError("Unexpected closing parenthesis")
    if token != "(":
        return token, index + 1

    expression: list[SExpression] = []
    index += 1
    while index < len(tokens) and tokens[index] != ")":
        child, index = _read_expression(tokens, index)
        expression.append(child)
    if index >= len(tokens):
        raise ValueError("Unbalanced parentheses in PDDL input")
    return expression, index + 1


def _parse_typed_parameters(tokens: list[str]) -> list[tuple[str, str]]:
    """Parse PDDL's grouped typed-list syntax, defaulting to ``object``."""
    parameters = []
    pending_variables = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if token.startswith("?"):
            pending_variables.append(token)
            index += 1
            continue
        if token == "-" and pending_variables:
            if index + 1 >= len(tokens):
                raise ValueError("Missing parameter type after '-'")
            parameter_type = tokens[index + 1]
            parameters.extend((variable, parameter_type) for variable in pending_variables)
            pending_variables.clear()
            index += 2
            continue
        index += 1

    parameters.extend((variable, "object") for variable in pending_variables)
    return parameters


def _split_literals(expression: SExpression) -> tuple[list[Literal], list[Literal]]:
    """Extract unconditional positive and negative literals."""
    positive: list[Literal] = []
    negative: list[Literal] = []

    def visit(node: SExpression, polarity: bool = True) -> None:
        if not isinstance(node, list) or not node:
            return

        head = node[0]
        normalized_head = head.lower() if isinstance(head, str) else ""
        arguments = node[1:]

        if normalized_head == "and":
            for argument in arguments:
                visit(argument, polarity)
        elif normalized_head == "not":
            if arguments:
                visit(arguments[0], not polarity)
        elif normalized_head in {"or", "when", "imply", "forall", "exists"}:
            # These remain available in the raw expression tree.
            return
        elif isinstance(head, str) and normalized_head not in _CONTROL_EXPRESSIONS:
            literal = tuple(str(part) for part in node)
            (positive if polarity else negative).append(literal)

    visit(expression)
    return positive, negative


def _section_value(section: list[SExpression], index: int, key: str) -> SExpression:
    if index + 1 >= len(section):
        raise ValueError(f"Missing value for {key}")
    return section[index + 1]


def _parse_action(section: list[SExpression], domain: Domain) -> None:
    if len(section) < 2 or not isinstance(section[1], str):
        raise ValueError("Malformed :action section")

    action = Action(section[1])
    index = 2
    while index < len(section):
        key = str(section[index]).lower()
        if key == ":parameters":
            raw_parameters = _section_value(section, index, key)
            if not isinstance(raw_parameters, list):
                raise ValueError(f"Parameters for action '{action.name}' must be a list")
            action.parameters = _parse_typed_parameters(
                [str(token) for token in raw_parameters]
            )
            index += 2
        elif key == ":precondition":
            action.raw_pre_tree = _section_value(section, index, key)
            action.preconds_pos, action.preconds_neg = _split_literals(
                action.raw_pre_tree
            )
            index += 2
        elif key == ":effect":
            action.raw_eff_tree = _section_value(section, index, key)
            action.effects_pos, action.effects_neg = _split_literals(action.raw_eff_tree)
            index += 2
        else:
            index += 1

    domain.actions[action.name] = action


def parse_domain(path: str | Path) -> Domain:
    """Parse the domain fields needed by Scale Plan."""
    domain_path = Path(path)
    tokens = _tokenize(_strip_comments(domain_path.read_text(encoding="utf-8")))
    if not tokens:
        raise ValueError(f"PDDL domain is empty: {domain_path}")

    expression, next_index = _read_expression(tokens)
    if next_index != len(tokens):
        raise ValueError("Unexpected content after the top-level PDDL expression")
    if not isinstance(expression, list) or not expression:
        raise ValueError("PDDL domain must contain a top-level list")
    if str(expression[0]).lower() != "define":
        raise ValueError("PDDL domain is missing the top-level 'define'")
    if len(expression) < 2:
        raise ValueError("PDDL domain is missing its domain declaration")

    declaration = expression[1]
    if (
        not isinstance(declaration, list)
        or len(declaration) < 2
        or str(declaration[0]).lower() != "domain"
    ):
        raise ValueError("PDDL domain is missing '(domain <name>)'")

    domain = Domain(str(declaration[1]))
    for section in expression[2:]:
        if not isinstance(section, list) or not section:
            continue
        head = str(section[0]).lower()
        if head == ":types":
            domain.types = [str(token) for token in section[1:]]
        elif head == ":predicates":
            for predicate in section[1:]:
                if isinstance(predicate, list) and predicate:
                    domain.predicates.append(
                        (str(predicate[0]), [str(token) for token in predicate[1:]])
                    )
        elif head == ":action":
            _parse_action(section, domain)

    return domain
