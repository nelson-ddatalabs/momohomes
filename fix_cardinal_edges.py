#!/usr/bin/env python3
"""
Fix Cardinal Edges
==================
Force edges to be perfectly cardinal by snapping to nearest axis.
"""

def snap_edges_to_cardinal(edges):
    """
    Snap edges to perfect cardinal directions.

    This ensures edges are perfectly horizontal or vertical.
    """
    for edge in edges:
        dx = edge.end[0] - edge.start[0]
        dy = edge.end[1] - edge.start[1]

        # Determine primary direction
        if abs(dx) > abs(dy):
            # Primarily horizontal - make it perfectly horizontal
            edge.end = (edge.end[0], edge.start[1])  # Keep Y constant
        else:
            # Primarily vertical - make it perfectly vertical
            edge.end = (edge.start[0], edge.end[1])  # Keep X constant

        # Recalculate cardinal direction
        edge.cardinal_direction = edge._determine_cardinal_direction()

    return edges


def ensure_edge_continuity_after_snap(edges):
    """
    After snapping to cardinal, ensure edges still connect.

    Adjust endpoints to maintain continuity.
    """
    for i in range(len(edges)):
        current = edges[i]
        next_edge = edges[(i + 1) % len(edges)]

        # Force next edge to start where current ends
        next_edge.start = current.end

    return edges