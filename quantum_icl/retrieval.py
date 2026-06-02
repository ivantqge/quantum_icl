"""Retrieval strategies for selecting in-context examples.

Each retriever selects up to k verified library entries to show for a query
task. Retrieval is restricted to the query's own tier so the demonstrated gate
set and target format always match.
"""

import math
import random
import re
from collections import Counter

# Target-computable scalar features used by default StructuralRetrieval.
_STRUCT_KEYS = [
    "num_qubits", "num_edges", "num_components",
    "state_sparsity", "state_nonzero", "state_top_amp", "state_entropy",
    "unitary_sparsity", "unitary_diag_mass", "unitary_frob_to_identity",
]

# Hidden-generator scalar features; used ONLY by OracleRetrieval (diagnostics).
_ORACLE_KEYS = [
    "gen_gate_count", "gen_two_qubit_gate_count", "gen_depth", "gen_t_count",
]


def _same_tier(query_task, library):
    return [e for e in library.all() if e["tier"] == query_task.tier]


class Retriever:
    name = "base"

    def select(self, query_task, library, k):
        raise NotImplementedError


class NoRetrieval(Retriever):
    name = "none"

    def select(self, query_task, library, k):
        return []


class RandomRetrieval(Retriever):
    name = "random"

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def select(self, query_task, library, k):
        pool = _same_tier(query_task, library)
        if len(pool) <= k:
            return list(pool)
        return self.rng.sample(pool, k)


def _tokenize(text):
    return [t for t in re.split(r"\W+", text.lower()) if t]


class TextRetrieval(Retriever):
    """TF-IDF cosine similarity over task descriptions (lightweight baseline)."""

    name = "text"

    def select(self, query_task, library, k):
        pool = _same_tier(query_task, library)
        if len(pool) <= k:
            return list(pool)

        docs = [_tokenize(e["description"]) for e in pool]
        df = Counter()
        for toks in docs:
            df.update(set(toks))
        n_docs = len(docs)
        idf = {t: math.log((1 + n_docs) / (1 + df[t])) + 1.0 for t in df}

        def vec(toks):
            tf = Counter(toks)
            return {t: tf[t] * idf.get(t, 0.0) for t in tf}

        def cosine(a, b):
            if not a or not b:
                return 0.0
            dot = sum(a[t] * b.get(t, 0.0) for t in a)
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            return dot / (na * nb) if na and nb else 0.0

        qv = vec(_tokenize(query_task.description))
        scored = sorted(
            zip(pool, (cosine(qv, vec(d)) for d in docs)),
            key=lambda x: x[1], reverse=True,
        )
        return [e for e, _ in scored[:k]]


class StructuralRetrieval(Retriever):
    """Similarity by quantum/task structural features (smaller distance = closer)."""

    name = "structural"

    @staticmethod
    def _distance(qf, cf, keys=_STRUCT_KEYS):
        dist = 0.0
        # Strong weight on matching qubit count.
        if qf.get("num_qubits") != cf.get("num_qubits"):
            dist += 5.0
        for key in keys:
            if key in qf and key in cf:
                dist += abs(qf[key] - cf[key])
        # List-valued features (L1 on sorted sequence).
        for key in ("degree_sequence", "unitary_phases"):
            dq, dc = qf.get(key), cf.get(key)
            if dq is not None and dc is not None:
                m = max(len(dq), len(dc))
                dq = list(dq) + [0] * (m - len(dq))
                dc = list(dc) + [0] * (m - len(dc))
                dist += sum(abs(a - b) for a, b in zip(dq, dc))
        return dist

    def select(self, query_task, library, k):
        pool = _same_tier(query_task, library)
        if len(pool) <= k:
            return list(pool)
        scored = sorted(
            ((e, self._distance(query_task.features, e["features"])) for e in pool),
            key=lambda x: x[1],
        )
        return [e for e, _ in scored[:k]]


class OracleRetrieval(Retriever):
    """DIAGNOSTIC ONLY: retrieve by hidden-generator features.

    Cheats by comparing generator gate count / T-count etc. -- features the
    LLM cannot derive from the target. Use solely as an upper bound on what
    structural retrieval could achieve given perfect target understanding.
    """

    name = "oracle"

    def select(self, query_task, library, k):
        pool = _same_tier(query_task, library)
        if len(pool) <= k:
            return list(pool)
        qf = getattr(query_task, "oracle_features", {}) or {}
        scored = sorted(
            ((e, StructuralRetrieval._distance(
                qf, e.get("oracle_features", {}), keys=_ORACLE_KEYS))
             for e in pool),
            key=lambda x: x[1],
        )
        return [e for e, _ in scored[:k]]


RETRIEVERS = {
    "none": lambda rng=None: NoRetrieval(),
    "random": lambda rng=None: RandomRetrieval(rng),
    "text": lambda rng=None: TextRetrieval(),
    "structural": lambda rng=None: StructuralRetrieval(),
    "oracle": lambda rng=None: OracleRetrieval(),
}


def make_retriever(name, rng=None) -> Retriever:
    return RETRIEVERS[name](rng)
