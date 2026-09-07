"""One package per research direction.

A direction is a coherent line of experiments with its own question, inputs,
decision rules, outputs, and limitations; a version is one immutable step
inside a direction. Each direction package exposes a narrow run interface and
keeps its versions private behind it. Directions never import each other.

See `fineweb_polygons.registry` for the single declaration of which directions
and versions exist.
"""
