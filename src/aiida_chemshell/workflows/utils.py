"""Shared utility helpers for aiida-chemshell WorkChains."""

from collections.abc import Mapping

from aiida.orm import Node


def apply_default_input_node_tags(inputs: Mapping, default_tags: dict) -> None:
    """
    Apply default labels and descriptions to WorkChain input nodes.

    Iterates over the provided input nodes and, for any node named in
    ``default_tags`` that does not already have a label or description set,
    assigns the corresponding default value. Existing labels and descriptions
    are left untouched. Namespaced inputs (e.g. an input namespace containing
    multiple nodes) are handled by tagging every node they contain.

    Parameters
    ----------
    inputs : Mapping
        The WorkChain input namespace (i.e. ``self.inputs``).
    default_tags : dict[str, tuple[str, str]]
        A mapping of input port name to a ``(label, description)`` tuple.
    """
    for port_name, (label, description) in default_tags.items():
        if port_name not in inputs:
            continue
        entry = inputs[port_name]
        nodes = [entry] if isinstance(entry, Node) else list(entry.values())
        for node in nodes:
            if not node.label:
                node.label = label
            if not node.description:
                node.description = description
