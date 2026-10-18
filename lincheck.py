"""Linearizability checker for key/value histories (Wing & Gong search with
memoisation, in the style of Lowe's P-compositionality optimisation).

A history is a list of Op records.  Each operation has an invocation time
``call`` and a response time ``ret``.  An operation whose outcome is unknown
(the client timed out) has ``ret = None`` and ``pending = True``: it may have
taken effect at any point after its call, or never.

Keys are independent registers, so the history is split per key and each key
is checked separately.  That is sound and makes the search exponentially
cheaper.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

INF = float("inf")


@dataclass
class Op:
    process: int
    kind: str            # 'put' | 'get' | 'cas'
    key: str
    arg: Any             # put: value; get: None; cas: (old, new)
    call: float
    ret: Optional[float]  # None if pending
    result: Any = None   # put: 'ok'; get: value or None; cas: bool

    @property
    def pending(self) -> bool:
        return self.ret is None


def step(state, op: Op) -> Tuple[bool, Any]:
    """Sequential register specification.  Returns (legal, new_state)."""
    if op.kind == "put":
        return True, op.arg
    if op.kind == "get":
        return (op.result == state), state
    if op.kind == "cas":
        old, new = op.arg
        if state == old:
            return (op.result is True), new
        return (op.result is False), state
    raise ValueError("unknown op kind %r" % op.kind)


def step_pending(state, op: Op):
    """A pending op may have happened (effect applied) or not.  Reads of a
    pending op are dropped before search; writes/cas apply their effect."""
    if op.kind == "put":
        return op.arg
    if op.kind == "cas":
        old, new = op.arg
        return new if state == old else state
    return state


def check_register(ops: List[Op], initial=None, max_states: int = 2_000_000):
    """Check one key's history.  Returns (ok, witness_order or None)."""
    ops = [o for o in ops if not (o.pending and o.kind == "get")]
    n = len(ops)
    if n == 0:
        return True, []
    rets = [INF if o.pending else o.ret for o in ops]
    order = sorted(range(n), key=lambda i: ops[i].call)
    completed_mask = 0
    for i, o in enumerate(ops):
        if not o.pending:
            completed_mask |= 1 << i
    seen = set()
    path: List[int] = []
    budget = [max_states]

    def dfs(done: int, state) -> bool:
        if done & completed_mask == completed_mask:
            return True
        key = (done, state if isinstance(state, (int, str, type(None), tuple)) else repr(state))
        if key in seen:
            return False
        seen.add(key)
        budget[0] -= 1
        if budget[0] < 0:
            raise RuntimeError("linearizability search exceeded state budget")
        # earliest response among not-yet-linearised ops bounds candidates
        min_ret = INF
        for i in range(n):
            if not (done >> i) & 1 and rets[i] < min_ret:
                min_ret = rets[i]
        for i in order:
            if (done >> i) & 1:
                continue
            o = ops[i]
            if o.call > min_ret:
                break  # sorted by call: no later op can be minimal either
            if o.pending:
                path.append(i)
                if dfs(done | (1 << i), step_pending(state, o)):
                    return True
                path.pop()
                continue
            ok, ns = step(state, o)
            if ok:
                path.append(i)
                if dfs(done | (1 << i), ns):
                    return True
                path.pop()
        return False

    ok = dfs(0, initial)
    return ok, ([ops[i] for i in path] if ok else None)


def check_history(history: List[Op], initial=None):
    """Check a multi-key history.  Returns (ok, bad_key or None)."""
    by_key = {}
    for o in history:
        by_key.setdefault(o.key, []).append(o)
    for k in sorted(by_key):
        ok, _ = check_register(by_key[k], initial)
        if not ok:
            return False, k
    return True, None
