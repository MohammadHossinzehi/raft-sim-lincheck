# raft-sim-lincheck

A deterministic, single-threaded simulator for the **Raft consensus algorithm**, plus a **Jepsen style linearizability checker**. It runs a replicated key/value store on a virtual clock, throws partitions, crashes and message loss at it, and then proves (or refutes) that what clients observed was linearizable. Pure Python standard library, no dependencies.

## Why

Raft is easy to read and notoriously easy to get subtly wrong. The only way to gain confidence is to attack it with faults and check invariants. Real clusters make that slow and unrepeatable. Because every message delay, timeout and fault here is drawn from one seeded RNG on a virtual clock, a bad run is reproduced exactly by its seed, and thousands of virtual seconds of chaos run in milliseconds.

## What is implemented

`raftsim.py`
- Leader election with randomised timeouts, the up to date log vote restriction, and step down on higher terms.
- Log replication with batching and the fast conflict backtracking optimisation (conflict term and index hints).
- The commit rule from section 5.4.2: only entries of the current term are committed by counting replicas, and a new leader appends a no-op so older entries commit with it.
- Persistent versus volatile state. A crashed node keeps term, vote and log, and rebuilds its key/value state by replay after restart.
- A replicated KV state machine with `put`, `get` and `cas`. Reads go through the log, so they are linearizable by construction (that is what gets verified).
- Nemesis: random partitions, leader isolation, crash and restart, flaky links.
- Runtime safety invariants: election safety (one leader per term), state machine safety (a committed index never changes), log matching, leader completeness, and a guard against truncating committed entries.
- Simulated clients that record a concurrent history. Requests that time out are recorded as **indeterminate** (they may or may not have happened), exactly as in Jepsen.

`lincheck.py`
- Wing and Gong style search over linearization orders with memoisation on (linearised set, register state), split per key (registers compose, so this is sound and exponentially cheaper).
- Correct handling of pending operations: a timed out write may have taken effect at any point after its call, or never.

## Run it

```
python raftsim.py --seeds 10 --duration 10000
python raftsim.py --seed 3 --nodes 3 --clients 6
python -m unittest -v
```

Sample output:

```
seed 0: linearizable | virtual_ms=13000, terms=14, committed_entries=525, ops_completed=512, ops_indeterminate=20, messages=28024, dropped=552
```

Using it as a library:

```python
from raftsim import Cluster
c = Cluster(5, seed=42)
leader = c.wait_for_leader()
leader.propose(("put", "x", 1))
c.isolate(leader.id)          # partition the leader away
c.run_for(2000)               # a new leader is elected on the majority side
c.heal(); c.run_for(1000)
c.check_all()                 # raises InvariantViolation on any safety bug
```

## Design decisions

- **One event heap, one RNG.** Messages, timers and client actions are all events ordered by (time, sequence number). Nothing depends on wall time or thread scheduling, so runs are bit for bit reproducible; a test asserts that the same seed gives the same history and different seeds differ.
- **Timer epochs.** Each node bumps an epoch whenever its timers are reset, so stale timer events are ignored instead of being removed from the heap.
- **Honest clients.** A client that times out cannot know whether its command committed. The history keeps it pending and the process is retired, which is what makes the checker's pending operation semantics matter.
- **Invariants checked while running**, not only at the end, so a violation is caught near the step that caused it.

## Testing

19 tests cover the checker (stale reads, real time order, pending writes visible or invisible but never un-happening, cas), and the protocol (election, replication, failover, a minority leader's uncommitted entry being overwritten after a heal, crash recovery of persistent state, a single node cluster, determinism, and chaos runs over many seeds with 3 and 5 nodes and message loss). One test corrupts a read in a real history and confirms the checker rejects it, so a green result means something.

## Limitations

No snapshots or log compaction, no membership changes, no pre-vote or leader leases. Reads are logged rather than served through ReadIndex. Those are natural next steps.
