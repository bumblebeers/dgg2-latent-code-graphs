import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cg"))
import numpy as np
from cg_ops import *

def chain():
    # 0 -> 1 -> 2 -> 3 ; 4 -> 1 ; 5 (test) -> 0 ; selected = 1
    adj = np.zeros((16, 16), bool)
    for a, b in [(0, 1), (1, 2), (2, 3), (4, 1), (5, 0)]:
        adj[a, b] = True
    kind = np.full(16, FUNC); kind[5] = TEST; kind[3] = CLASS
    return GraphState.new(adj, kind, 1)

def test_basics():
    s = chain(); g = s.to_grid()
    assert g[0, 1] == EDGE and g[1, 0] == BG and g[1, 1] == SEL and g[5, 5] == TEST and g[3, 3] == CLASS
    assert reverse_edges(s).to_grid()[1, 0] == EDGE and reverse_edges(s).to_grid()[0, 1] == BG
    assert (reverse_edges(reverse_edges(s)).to_grid() == g).all()
    assert (reverse_order(reverse_order(s)).to_grid() == g).all()
    assert (reverse_order(s).to_grid() == g[::-1, ::-1]).all()            # = rot180 of the image
    assert (reverse_edges(s).to_grid() == g.T).all()                       # = transpose of the image

def test_selected_ops():
    s = chain()
    d = delete_selected(s).to_grid()
    assert d[1, 1] == BG and (d[1, :] == BG).all() and (d[:, 1] == BG).all() and d[2, 3] == EDGE
    i = isolate_selected(s).to_grid()
    assert i[1, 1] == SEL and i[0, 1] == BG and i[1, 2] == BG and i[2, 3] == EDGE
    kc = keep_callees(s)
    assert set(np.flatnonzero(kc.present)) == {1, 2} and kc.to_grid()[1, 2] == EDGE and kc.to_grid()[2, 3] == BG
    kr = keep_callers(s)
    assert set(np.flatnonzero(kr.present)) == {0, 1, 4} and kr.to_grid()[0, 1] == EDGE
    assert set(np.flatnonzero(mark_reachable(s).marked)) == {2, 3}
    assert set(np.flatnonzero(mark_ancestors(s).marked)) == {0, 4, 5}
    f = selected_first(s)
    assert f.selected == 0 and f.to_grid()[0, 0] == SEL and f.to_grid()[1, 0] == EDGE  # old 0 -> new 1, calls new 0
    t = delete_tests(s)
    assert not t.present[5] and t.to_grid()[5, 0] == BG
    c = delete_classes(s)
    assert not c.present[3] and c.present[5] and c.to_grid()[2, 3] == BG
    s3 = GraphState.new(s.adj, np.where(np.arange(16) == 1, CLASS, s.kind), 1)     # the selected symbol is a class
    assert delete_classes(s3).present[1] and not delete_classes(s3).present[3]     # ...and is never deleted by kind

def test_compositions_and_mirror_identities():
    s = chain()
    # the held-out ops are edge-reversal conjugates of trained ones
    assert (apply(s, "keep_callers").to_grid() == reverse_edges(keep_callees(reverse_edges(s))).to_grid()).all()
    assert (apply(s, "mark_ancestors").to_grid() == reverse_edges(mark_reachable(reverse_edges(s))).to_grid()).all()
    assert (apply(s, "reverse_edges+reverse_order").to_grid() == s.to_grid().T[::-1, ::-1]).all()
    # order matters for delete_tests+mark_reachable when a test lies on a path
    adj = np.zeros((16, 16), bool); adj[0, 1] = adj[1, 2] = True
    kind = np.full(16, FUNC); kind[1] = TEST
    s2 = GraphState.new(adj, kind, 0)
    assert mark_reachable(s2).marked[2] and not apply(s2, "delete_tests+mark_reachable").marked[2]
    assert len(ALL_TARGETS) == 22 and ALL_TARGETS[0] == "identity" and len(set(ALL_TARGETS)) == 22

def test_render_roundtrip():
    g = chain().to_grid(); img = render(g)
    assert img.shape == (768, 768, 3)
    coarse = img.reshape(16, 48, 16, 48, 3).mean((1, 3))
    assert (grid_from_rgb16(coarse) == g).all()
    assert (grid_from_rgb16(np.moveaxis(coarse, -1, 0) / 255.0) == g).all()
    noisy = coarse + np.random.default_rng(0).normal(0, 12, coarse.shape)
    assert (grid_from_rgb16(noisy) == g).mean() > 0.99

if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
