"""Hybrid KG + RAG Pipeline Modules"""

__all__ = ['KGRetriever', 'TextRetriever', 'HybridPipeline']


def __getattr__(name):
	"""Lazily import submodules so importing the package stays lightweight."""
	if name == 'KGRetriever':
		from .kg_retriever import KGRetriever
		return KGRetriever
	if name == 'TextRetriever':
		from .text_retriever import TextRetriever
		return TextRetriever
	if name == 'HybridPipeline':
		from .pipeline import HybridPipeline
		return HybridPipeline
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
