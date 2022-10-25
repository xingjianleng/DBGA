import nox


_py_versions = range(8, 11)

@nox.session(python=[f"3.{v}" for v in _py_versions])
def test(session):
    session.install(".")
    session.chdir("tests")
    session.run("pytest")
