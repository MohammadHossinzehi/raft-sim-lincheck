import unittest

from lincheck import Op, check_history, check_register
from raftsim import Cluster, Config, InvariantViolation, LEADER


def put(p, k, v, c, r):
    return Op(p, "put", k, v, c, r, "ok")


def get(p, k, v, c, r):
    return Op(p, "get", k, None, c, r, v)


class LinCheckerTests(unittest.TestCase):
    def test_sequential_ok(self):
        h = [put(1, "x", 1, 0, 1), get(1, "x", 1, 2, 3)]
        self.assertTrue(check_history(h)[0])

    def test_stale_read_rejected(self):
        h = [put(1, "x", 1, 0, 1), put(1, "x", 2, 2, 3), get(2, "x", 1, 4, 5)]
        self.assertFalse(check_history(h)[0])

    def test_concurrent_either_order(self):
        h = [put(1, "x", 1, 0, 10), get(2, "x", None, 1, 2), get(3, "x", 1, 3, 9)]
        self.assertTrue(check_history(h)[0])

    def test_pending_write_may_be_visible(self):
        h = [Op(1, "put", "x", 5, 0, None, None), get(2, "x", 5, 10, 11)]
        self.assertTrue(check_history(h)[0])

    def test_pending_write_may_be_invisible(self):
        h = [Op(1, "put", "x", 5, 0, None, None), get(2, "x", None, 10, 11)]
        self.assertTrue(check_history(h)[0])

    def test_pending_write_cannot_unhappen(self):
        h = [Op(1, "put", "x", 5, 0, None, None), get(2, "x", 5, 10, 11), get(2, "x", None, 12, 13)]
        self.assertFalse(check_history(h)[0])

    def test_cas(self):
        ok = [put(1, "x", 1, 0, 1), Op(1, "cas", "x", (1, 2), 2, 3, True), get(1, "x", 2, 4, 5)]
        self.assertTrue(check_history(ok)[0])
        bad = [put(1, "x", 1, 0, 1), Op(1, "cas", "x", (9, 2), 2, 3, True)]
        self.assertFalse(check_history(bad)[0])

    def test_real_time_order_respected(self):
        h = [get(1, "x", 1, 0, 1), put(2, "x", 1, 5, 6)]  # read finished before write began
        self.assertFalse(check_history(h)[0])


class RaftTests(unittest.TestCase):
    def test_elects_single_leader(self):
        c = Cluster(5, seed=1)
        l = c.wait_for_leader()
        self.assertIsNotNone(l)
        self.assertEqual(sum(1 for n in c.nodes if n.role == LEADER), 1)

    def test_replication_and_commit(self):
        c = Cluster(5, seed=2)
        l = c.wait_for_leader()
        l.propose(("put", "k", 7))
        c.run_for(500)
        for n in c.nodes:
            self.assertEqual(n.kv.get("k"), 7)

    def test_leader_failover(self):
        c = Cluster(5, seed=3)
        l = c.wait_for_leader()
        l.propose(("put", "k", 1)); c.run_for(300)
        c.crash(l.id)
        l2 = c.wait_for_leader()
        self.assertNotEqual(l2.id, l.id)
        self.assertGreater(l2.term, l.term)
        l2.propose(("put", "k", 2)); c.run_for(300)
        self.assertEqual(l2.kv["k"], 2)

    def test_minority_cannot_commit_and_is_overwritten(self):
        c = Cluster(5, seed=4)
        old = c.wait_for_leader()
        c.run_for(300)
        c.isolate(old.id)
        stale = old.propose(("put", "x", "lost"))
        c.run_for(1500)
        self.assertLess(old.commit, stale[0])  # never committed
        others = [n for n in c.nodes if n.id != old.id]
        new = max((n for n in others if n.role == LEADER), key=lambda n: n.term)
        new.propose(("put", "x", "won")); c.run_for(400)
        c.heal(); c.run_for(1500)
        for n in c.nodes:
            self.assertEqual(n.kv.get("x"), "won")
        c.check_all()

    def test_crash_recovery_keeps_persistent_state(self):
        c = Cluster(3, seed=5)
        l = c.wait_for_leader()
        l.propose(("put", "k", 9)); c.run_for(300)
        f = next(n for n in c.nodes if n.id != l.id)
        term, log = f.term, list(f.log)
        c.crash(f.id); c.run_for(200); c.restart(f.id)
        self.assertEqual((f.term, f.log), (term, log))
        self.assertEqual(f.kv, {})  # volatile state is rebuilt by replay
        c.run_for(800)
        self.assertEqual(f.kv.get("k"), 9)

    def test_single_node_cluster(self):
        c = Cluster(1, seed=6)
        l = c.wait_for_leader()
        l.propose(("put", "a", 1)); c.run_for(50)
        self.assertEqual(l.kv["a"], 1)

    def test_deterministic_by_seed(self):
        def run(seed):
            c = Cluster(5, seed=seed)
            c.run_chaos_test(3000)
            return [(o.process, o.kind, o.key, o.arg, o.call, o.ret, o.result) for o in c.history]
        self.assertEqual(run(11), run(11))
        self.assertNotEqual(run(11), run(12))

    def test_chaos_many_seeds_are_linearizable(self):
        for seed in range(12):
            c = Cluster(5, seed=seed)
            ok, key = c.run_chaos_test(6000)
            self.assertTrue(ok, "seed %d not linearizable on key %r" % (seed, key))
            self.assertGreater(len(c.committed), 5)

    def test_chaos_with_message_loss_three_nodes(self):
        for seed in range(6):
            c = Cluster(3, seed=100 + seed, cfg=Config(drop_prob=0.1))
            ok, _ = c.run_chaos_test(5000)
            self.assertTrue(ok)

    def test_invariant_detects_double_leader(self):
        c = Cluster(3, seed=0)
        c.record_leader(5, 0)
        with self.assertRaises(InvariantViolation):
            c.record_leader(5, 1)

    def test_corrupted_history_is_flagged(self):
        """Sanity check that the checker has teeth: corrupt one read in a
        genuinely linearizable chaos history and it must be rejected."""
        c = Cluster(3, seed=7)
        c.run_chaos_test(3000)
        h = c.history
        victim = next(o for o in h if o.kind == "get" and not o.pending)
        victim.result = "garbage-value"
        self.assertFalse(check_history(h)[0])


if __name__ == "__main__":
    unittest.main()
