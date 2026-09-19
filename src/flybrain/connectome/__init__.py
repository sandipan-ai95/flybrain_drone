"""Connectome data ingestion and graph construction.

Milestone 1 provides only a *synthetic* connectome for closed-loop bring-up.
Real Drosophila datasets (FlyWire, hemibrain) are wired in from Milestone 2
via :mod:`flybrain.connectome.loader`.
"""
from .graph import Connectome, SynapticEdges
from .annotations import NeuronAnnotations, Population

__all__ = ["Connectome", "SynapticEdges", "NeuronAnnotations", "Population"]
