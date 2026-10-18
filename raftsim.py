"""raftsim: a deterministic discrete-event simulator for the Raft consensus
algorithm, with fault injection, safety invariants and history recording for
linearizability checking.  Pure standard library.

Everything runs on one virtual clock driven by a seeded RNG, so a failing
seed reproduces exactly.
"""
import heapq
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lincheck import Op, check_history

FOLLOWER, CANDIDATE, LEADER = "follower", "candidate", "leader"


class InvariantViolation(AssertionError):
    pass


@dataclass
class Config:
    election_min: int = 150
    election_max: int = 300
    heartbeat: int = 50
    latency_min: int = 5
    latency_max: int = 25
    drop_prob: float = 0.0
    max_batch: int = 16


class Node:
    def __init__(self, nid: int, n: int, cluster: "Cluster"):
        self.id, self.n, self.cl = nid, n, cluster
        self.peers = [i for i in range(n) if i != nid]
        # persistent state (survives crashes)
        self.term = 0
        self.voted_for: Optional[int] = None
        self.log: List[tuple] = [(0, None)]  # index 0 is a sentinel
        self.alive = True
        self._reset_volatile()

    # ---- volatile state ----------------------------------------------
    def _reset_volatile(self):
        self.role = FOLLOWER
        self.leader_id: Optional[int] = None
        self.commit = 0
        self.applied = 0
        self.kv: Dict[str, Any] = {}
        self.results: Dict[int, tuple] = {}  # index -> (term, result)
        self.votes = set()
        self.next_index: Dict[int, int] = {}
        self.match_index: Dict[int, int] = {}
        self.epoch = 0  # invalidates stale timers

    @property
    def last_index(self):
        return len(self.log) - 1

    def last_term(self):
        return self.log[-1][0]

    def majority(self):
        return self.n // 2 + 1

    # ---- timers ------------------------------------------------------
    def reset_election_timer(self):
        self.epoch += 1
        c = self.cl.cfg
        d = self.cl.rng.randint(c.election_min, c.election_max)
        self.cl.schedule(d, ("timer", self.id, "election", self.epoch))

    def start_heartbeat_timer(self):
        self.cl.schedule(self.cl.cfg.heartbeat, ("timer", self.id, "heartbeat", self.epoch))

    def on_timer(self, kind, epoch):
        if not self.alive or epoch != self.epoch:
            return
        if kind == "election" and self.role != LEADER:
            self.start_election()
        elif kind == "heartbeat" and self.role == LEADER:
            self.broadcast_append()
            self.start_heartbeat_timer()

    # ---- role transitions --------------------------------------------
    def step_down(self, term):
        if term > self.term:
            self.term, self.voted_for = term, None
        if self.role != FOLLOWER:
            self.role = FOLLOWER
        self.reset_election_timer()

    def start_election(self):
        self.role = CANDIDATE
        self.term += 1
        self.voted_for = self.id
        self.votes = {self.id}
        self.leader_id = None
        self.reset_election_timer()
        if len(self.votes) >= self.majority():
            return self.become_leader()
        for p in self.peers:
            self.cl.send(self.id, p, ("RV", self.term, self.id, self.last_index, self.last_term()))

    def become_leader(self):
        self.cl.record_leader(self.term, self.id)
        self.role = LEADER
        self.leader_id = self.id
        self.epoch += 1
        self.next_index = {p: self.last_index + 1 for p in self.peers}
        self.match_index = {p: 0 for p in self.peers}
        self.log.append((self.term, None))  # no-op commits prior-term entries
        self.advance_commit()
        self.broadcast_append()
        self.start_heartbeat_timer()

    # ---- RPC dispatch --------------------------------------------------
    def on_message(self, src, msg):
        if not self.alive:
            return
        kind = msg[0]
        getattr(self, "on_" + kind)(src, *msg[1:])

    def on_RV(self, src, term, cand, last_index, last_term):
        if term > self.term:
            self.step_down(term)
        grant = False
        if term == self.term and self.voted_for in (None, cand):
            up_to_date = (last_term, last_index) >= (self.last_term(), self.last_index)
            if up_to_date:
                grant = True
                self.voted_for = cand
                self.reset_election_timer()
        self.cl.send(self.id, src, ("RVR", self.term, grant))

    def on_RVR(self, src, term, granted):
        if term > self.term:
            return self.step_down(term)
        if self.role != CANDIDATE or term != self.term or not granted:
            return
        self.votes.add(src)
        if len(self.votes) >= self.majority():
            self.become_leader()

    def send_append(self, p):
        ni = self.next_index[p]
        prev = ni - 1
        entries = self.log[ni:ni + self.cl.cfg.max_batch]
        self.cl.send(self.id, p, ("AE", self.term, self.id, prev, self.log[prev][0], entries, self.commit))

    def broadcast_append(self):
        for p in self.peers:
            self.send_append(p)

    def on_AE(self, src, term, leader, prev, prev_term, entries, leader_commit):
        if term < self.term:
            return self.cl.send(self.id, src, ("AER", self.term, False, 0, None, None))
        if term > self.term or self.role != FOLLOWER:
            self.step_down(term)
        else:
            self.reset_election_timer()
        self.leader_id = leader
        if prev > self.last_index:
            return self.cl.send(self.id, src, ("AER", self.term, False, 0, None, self.last_index + 1))
        if self.log[prev][0] != prev_term:
            ct = self.log[prev][0]
            ci = prev
            while ci > 1 and self.log[ci - 1][0] == ct:
                ci -= 1
            return self.cl.send(self.id, src, ("AER", self.term, False, 0, ct, ci))
        for off, e in enumerate(entries):
            idx = prev + 1 + off
            if idx <= self.last_index:
                if self.log[idx][0] != e[0]:
                    if idx <= self.commit:
                        raise InvariantViolation("node %d would truncate committed index %d" % (self.id, idx))
                    del self.log[idx:]
                    self.log.append(e)
            else:
                self.log.append(e)
        match = prev + len(entries)
        if leader_commit > self.commit:
            self.commit = max(self.commit, min(leader_commit, match))
            self.apply()
        self.cl.send(self.id, src, ("AER", self.term, True, match, None, None))

    def on_AER(self, src, term, ok, match, conflict_term, conflict_index):
        if term > self.term:
            return self.step_down(term)
        if self.role != LEADER or term != self.term:
            return
        if ok:
            if match > self.match_index[src]:
                self.match_index[src] = match
                self.advance_commit()
            self.next_index[src] = max(self.next_index[src], match + 1)
            if self.next_index[src] <= self.last_index:
                self.send_append(src)
        else:
            if conflict_term is not None:
                hint = None
                for i in range(self.last_index, 0, -1):
                    if self.log[i][0] == conflict_term:
                        hint = i + 1
                        break
                    if self.log[i][0] < conflict_term:
                        break
                ni = hint if hint is not None else conflict_index
            else:
                ni = conflict_index if conflict_index is not None else self.next_index[src] - 1
            self.next_index[src] = max(1, min(ni, self.next_index[src] - 1))
            self.send_append(src)

    # ---- commit / apply ------------------------------------------------
    def advance_commit(self):
        for n in range(self.last_index, self.commit, -1):
            if self.log[n][0] != self.term:
                break  # only count replicas for current-term entries (Raft 5.4.2)
            reps = 1 + sum(1 for p in self.peers if self.match_index[p] >= n)
            if reps >= self.majority():
                self.commit = n
                self.apply()
                break

    def apply(self):
        while self.applied < self.commit:
            self.applied += 1
            term, cmd = self.log[self.applied]
            self.cl.record_commit(self.applied, (term, cmd))
            self.results[self.applied] = (term, self.exec(cmd))

    def exec(self, cmd):
        if cmd is None:
            return None
        op = cmd[0]
        if op == "put":
            self.kv[cmd[1]] = cmd[2]
            return "ok"
        if op == "get":
            return self.kv.get(cmd[1])
        if op == "cas":
            if self.kv.get(cmd[1]) == cmd[2]:
                self.kv[cmd[1]] = cmd[3]
                return True
            return False
        raise ValueError(cmd)

    # ---- client interface ----------------------------------------------
    def propose(self, cmd):
        if not self.alive or self.role != LEADER:
            return None
        self.log.append((self.term, cmd))
        if self.n == 1:
            self.advance_commit()
        else:
            self.broadcast_append()
        return self.last_index, self.term

    # ---- crash / restart -----------------------------------------------
    def crash(self):
        self.alive = False

    def restart(self):
        self.alive = True
        self._reset_volatile()  # term, vote and log persist
        self.reset_election_timer()


class Cluster:
    def __init__(self, n=5, seed=0, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.rng = random.Random(seed)
        self.now = 0
        self.q: list = []
        self.seq = 0
        self.nodes = [Node(i, n, self) for i in range(n)]
        self.blocked = set()  # directed (src, dst) pairs that drop messages
        self.leaders_by_term: Dict[int, int] = {}
        self.committed: Dict[int, tuple] = {}
        self.history: List[Op] = []
        self.msgs_sent = 0
        self.msgs_dropped = 0
        for nd in self.nodes:
            nd.reset_election_timer()

    # ---- event loop ----------------------------------------------------
    def schedule(self, delay, ev):
        self.seq += 1
        heapq.heappush(self.q, (self.now + delay, self.seq, ev))

    def send(self, src, dst, msg):
        self.msgs_sent += 1
        if (src, dst) in self.blocked or self.rng.random() < self.cfg.drop_prob:
            self.msgs_dropped += 1
            return
        c = self.cfg
        self.schedule(self.rng.randint(c.latency_min, c.latency_max), ("msg", src, dst, msg))

    def run_until(self, t):
        while self.q and self.q[0][0] <= t:
            when, _, ev = heapq.heappop(self.q)
            self.now = when
            kind = ev[0]
            if kind == "msg":
                _, src, dst, msg = ev
                if (src, dst) not in self.blocked:
                    self.nodes[dst].on_message(src, msg)
            elif kind == "timer":
                _, nid, tk, epoch = ev
                self.nodes[nid].on_timer(tk, epoch)
            elif kind == "call":
                ev[1]()
        self.now = max(self.now, t)

    def run_for(self, d):
        self.run_until(self.now + d)

    # ---- invariants ----------------------------------------------------
    def record_leader(self, term, nid):
        prev = self.leaders_by_term.setdefault(term, nid)
        if prev != nid:
            raise InvariantViolation("two leaders in term %d: %d and %d" % (term, prev, nid))

    def record_commit(self, index, entry):
        prev = self.committed.setdefault(index, entry)
        if prev != entry:
            raise InvariantViolation("state machine safety: index %d committed as %r and %r" % (index, prev, entry))

    def check_log_matching(self):
        for a in self.nodes:
            for b in self.nodes:
                if a.id >= b.id:
                    continue
                for i in range(min(a.last_index, b.last_index), 0, -1):
                    if a.log[i][0] == b.log[i][0]:
                        if a.log[:i + 1] != b.log[:i + 1]:
                            raise InvariantViolation("log matching violated between %d and %d at %d" % (a.id, b.id, i))
                        break

    def check_leader_completeness(self):
        for idx, entry in self.committed.items():
            for nd in self.nodes:
                if nd.role == LEADER and nd.alive and nd.term >= entry[0] and nd.last_index >= 1:
                    if idx > nd.last_index or nd.log[idx] != entry:
                        raise InvariantViolation("leader %d (term %d) lacks committed index %d" % (nd.id, nd.term, idx))

    def check_all(self):
        self.check_log_matching()
        self.check_leader_completeness()

    # ---- fault injection -----------------------------------------------
    def leader(self) -> Optional[Node]:
        ls = [n for n in self.nodes if n.alive and n.role == LEADER]
        return max(ls, key=lambda n: n.term) if ls else None

    def partition(self, *groups):
        self.heal()
        gid = {}
        for gi, g in enumerate(groups):
            for x in g:
                gid[x] = gi
        for a in range(len(self.nodes)):
            for b in range(len(self.nodes)):
                if a != b and gid.get(a, -1) != gid.get(b, -1):
                    self.blocked.add((a, b))

    def isolate(self, nid):
        self.partition([nid], [i for i in range(len(self.nodes)) if i != nid])

    def heal(self):
        self.blocked.clear()

    def crash(self, nid):
        self.nodes[nid].crash()

    def restart(self, nid):
        self.nodes[nid].restart()

    def wait_for_leader(self, limit=5000):
        end = self.now + limit
        while self.now < end:
            self.run_for(10)
            l = self.leader()
            if l and sum(1 for n in self.nodes if n.alive and n.term == l.term) >= l.majority():
                return l
        return None

    # ---- clients -------------------------------------------------------
    def start_clients(self, nclients=4, keys=("a", "b", "c"), think=(5, 40),
                      timeout=600, until=10_000):
        self._next_proc = nclients
        for c in range(nclients):
            self._client_next(c, keys, think, timeout, until)

    def _client_next(self, proc, keys, think, timeout, until):
        if self.now >= until:
            return
        d = self.rng.randint(*think)
        self.schedule(d, ("call", lambda: self._client_invoke(proc, keys, think, timeout, until)))

    def _client_invoke(self, proc, keys, think, timeout, until):
        if self.now >= until:
            return
        r = self.rng.random()
        key = self.rng.choice(keys)
        if r < 0.4:
            kind, arg, cmd = "put", self.rng.randint(0, 99), None
            cmd = ("put", key, arg)
        elif r < 0.8:
            kind, arg, cmd = "get", None, ("get", key)
        else:
            old, new = self.rng.choice([None, *range(0, 100)][:4]), self.rng.randint(0, 99)
            kind, arg, cmd = "cas", (old, new), ("cas", key, old, new)
        op = Op(proc, kind, key, arg, call=self.now, ret=None)
        self.history.append(op)
        deadline = self.now + timeout
        self._try_submit(op, cmd, deadline, keys, think, timeout, until, proc)

    def _try_submit(self, op, cmd, deadline, keys, think, timeout, until, proc):
        if self.now >= deadline:
            return self._client_timeout(op, keys, think, timeout, until, proc)
        target = self.leader() or self.rng.choice(self.nodes)
        got = target.propose(cmd)
        if got is None:
            return self.schedule(15, ("call", lambda: self._try_submit(op, cmd, deadline, keys, think, timeout, until, proc)))
        self._poll(op, target, got, deadline, keys, think, timeout, until, proc)

    def _poll(self, op, node, got, deadline, keys, think, timeout, until, proc):
        idx, term = got
        res = node.results.get(idx)
        if node.alive and res is not None and res[0] == term:
            op.ret, op.result = self.now, res[1]
            return self._client_next(proc, keys, think, timeout, until)
        if self.now >= deadline:
            return self._client_timeout(op, keys, think, timeout, until, proc)
        self.schedule(10, ("call", lambda: self._poll(op, node, got, deadline, keys, think, timeout, until, proc)))

    def _client_timeout(self, op, keys, think, timeout, until, proc):
        # outcome unknown: leave op pending and continue as a fresh process
        self._next_proc += 1
        self._client_next(self._next_proc, keys, think, timeout, until)

    def chaos(self, until, period=(300, 900)):
        """Random nemesis: partitions, leader isolation, crashes, healing."""
        def act():
            if self.now >= until:
                self.heal()
                for nd in self.nodes:
                    if not nd.alive:
                        nd.restart()
                return
            n = len(self.nodes)
            choice = self.rng.choice(["heal", "split", "isolate_leader", "crash", "restart", "flaky"])
            if choice == "heal":
                self.heal()
                self.cfg.drop_prob = 0.0
            elif choice == "split":
                ids = list(range(n)); self.rng.shuffle(ids)
                k = self.rng.randint(1, n // 2)
                self.partition(ids[:k], ids[k:])
            elif choice == "isolate_leader":
                l = self.leader()
                if l:
                    self.isolate(l.id)
            elif choice == "crash":
                alive = [x for x in self.nodes if x.alive]
                if len(alive) > self.nodes[0].majority():
                    self.rng.choice(alive).crash()
            elif choice == "restart":
                dead = [x for x in self.nodes if not x.alive]
                if dead:
                    self.rng.choice(dead).restart()
            elif choice == "flaky":
                self.cfg.drop_prob = 0.2
            self.schedule(self.rng.randint(*period), ("call", act))
        self.schedule(self.rng.randint(*period), ("call", act))

    def run_chaos_test(self, duration=10_000, nclients=4, check_every=250):
        self.start_clients(nclients=nclients, until=duration)
        self.chaos(until=duration)
        t = 0
        while t < duration:
            t += check_every
            self.run_until(t)
            self.check_all()
        self.heal(); self.cfg.drop_prob = 0.0
        for nd in self.nodes:
            if not nd.alive:
                nd.restart()
        self.run_until(duration + 3000)
        self.check_all()
        return check_history(self.history)

    def stats(self):
        done = sum(1 for o in self.history if not o.pending)
        return {
            "virtual_ms": self.now,
            "terms": max(self.leaders_by_term) if self.leaders_by_term else 0,
            "committed_entries": len(self.committed),
            "ops_completed": done,
            "ops_indeterminate": len(self.history) - done,
            "messages": self.msgs_sent,
            "dropped": self.msgs_dropped,
        }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Raft simulator with chaos + linearizability checking")
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5, help="run seeds 0..N-1")
    ap.add_argument("--seed", type=int, default=None, help="run a single seed")
    ap.add_argument("--duration", type=int, default=10_000, help="virtual ms of chaos")
    ap.add_argument("--clients", type=int, default=4)
    a = ap.parse_args(argv)
    seeds = [a.seed] if a.seed is not None else range(a.seeds)
    bad = 0
    for s in seeds:
        c = Cluster(a.nodes, seed=s)
        try:
            ok, key = c.run_chaos_test(a.duration, a.clients)
        except InvariantViolation as e:
            print("seed %d: INVARIANT VIOLATION: %s" % (s, e)); bad += 1; continue
        st = c.stats()
        print("seed %d: %s | %s" % (s, "linearizable" if ok else "NOT LINEARIZABLE (key %r)" % key,
                                    ", ".join("%s=%s" % kv for kv in st.items())))
        bad += 0 if ok else 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
