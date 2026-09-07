"""Numbered post-processing stages of Direction 1 (V7-V10).

Each stage reads the previous version's published rows and writes a new
immutable artifact plus a manifest. Stages never change document selection
made by an earlier version.
"""
