import os, sys, tempfile, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cg"))
from cg_extract import extract

FILES = {
    "pkg/__init__.py": "from .core import helper\n",
    "pkg/core.py": """
        import os
        from . import util
        from .util import fmt as _fmt

        def helper(x):
            return _fmt(x) + util.fmt(x)

        async def fetch(u):
            def inner(v):
                return helper(v)
            return inner(u)

        class Base:
            def run(self):
                return self.step()
            def step(self):
                return helper(1)

        class Child(Base):
            def step(self):
                return super().step() + Base().run()
    """,
    "pkg/util.py": "def fmt(x):\n    return str(x)\n",
    "tests/test_core.py": "from pkg import helper\nfrom pkg.core import Child\n\ndef test_it():\n    assert helper(1) and Child().run()\n",
}

def test_extract():
    with tempfile.TemporaryDirectory() as d:
        for rel, src in FILES.items():
            p = os.path.join(d, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(textwrap.dedent(src))
        g = extract(d)
    ids = {n["id"]: n for n in g["nodes"]}
    E = {(e["source"], e["target"], e["type"]) for e in g["edges"]}
    assert ids["pkg.core.fetch"]["is_async"] and ids["pkg.core.fetch.inner"]["kind"] == "function"
    assert ids["pkg.core.Base.run"]["kind"] == "method" and ids["tests.test_core.test_it"]["is_test"]
    for a, b in [("pkg.core.helper", "pkg.util.fmt"), ("pkg.core.fetch", "pkg.core.fetch.inner"),
                 ("pkg.core.fetch.inner", "pkg.core.helper"), ("pkg.core.Base.run", "pkg.core.Base.step"),
                 ("pkg.core.Child.step", "pkg.core.Base.step"), ("pkg.core.Child.step", "pkg.core.Base"),
                 ("tests.test_core.test_it", "pkg.core.helper"),      # through the package re-export
                 ("tests.test_core.test_it", "pkg.core.Child")]:
        assert (a, b, "calls") in E, (a, b)
    assert ("pkg.core.Child", "pkg.core.Base", "inherits") in E
    assert ("pkg.util.fmt", "pkg.core.helper", "calls") not in E      # directed, not symmetric
    assert g["graph"]["n_asymmetric_call_edges"] == g["graph"]["n_call_edges"]

if __name__ == "__main__":
    test_extract(); print("ok test_extract")
