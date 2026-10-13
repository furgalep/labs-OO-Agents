"""Edit results: real diff member; agent sees message with absolute path and line counts."""

from __future__ import annotations

import pytest

from nooa.tools.shell_tools import ShellTools


@pytest.mark.asyncio
async def test_match_replace_returns_real_diff_with_absolute_path(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("one\ntwo\nthree\nfour\n")
    shell = ShellTools(cwd=tmp_path)
    m = await shell.read("a.py", lines=(2, 3))
    r = await shell.replace(m, "TWO\n")
    assert f.read_text() == "one\nTWO\nfour\n"
    assert r.message == f"Edited {f.resolve()} (replaced lines 2-3 (2 lines) with 1 line)"
    assert "@@ -2,2 +2 @@" in r.diff
    assert "-two\n" in r.diff and "-three\n" in r.diff and "+TWO\n" in r.diff
    assert str(r) == r.message


@pytest.mark.asyncio
async def test_path_replace_returns_real_diff_with_line_numbers(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("a\nb\nc\n")
    shell = ShellTools(cwd=tmp_path)
    r = await shell.replace(str(f), "c\n", "c1\nc2\n")
    assert r.message == f"Edited {f.resolve()} (replaced line 3 (1 line) with 2 lines)"
    assert "@@ -3 +3,2 @@" in r.diff
    assert "-c\n" in r.diff and "+c1\n" in r.diff and "+c2\n" in r.diff


@pytest.mark.asyncio
async def test_deletion_is_visible_in_diff(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("keep\ndrop1\ndrop2\nkeep2\n")
    shell = ShellTools(cwd=tmp_path)
    m = await shell.read("c.txt", lines=(2, 3))
    r = await shell.replace(m, "")
    assert "with 0 lines" in r.message
    assert "-drop1\n" in r.diff and "-drop2\n" in r.diff


@pytest.mark.asyncio
async def test_write_file_reports_absolute_path_and_overwrite(tmp_path):
    shell = ShellTools(cwd=tmp_path)
    target = (tmp_path / "sub" / "new.txt").resolve()
    r1 = await shell.write_file(str(target), "x\ny\n")
    assert r1.message == f"Created {target} (2 lines)"
    r2 = await shell.write_file(str(target), "z\n")
    assert r2.message == f"Overwrote {target} (1 lines)"


@pytest.mark.asyncio
async def test_relative_paths_are_still_accepted_and_reported_absolute(tmp_path):
    shell = ShellTools(cwd=tmp_path)
    w = await shell.write_file("sub/new.txt", "a\nb\n")
    target = (tmp_path / "sub" / "new.txt").resolve()
    assert w.message == f"Created {target} (2 lines)"
    r = await shell.replace("sub/new.txt", "a\n", "z\n")
    assert r.message.startswith(f"Edited {target} ")
    assert target.read_text() == "z\nb\n"


@pytest.mark.asyncio
async def test_match_replace_still_works_after_relative_read(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("a\nb\n")
    shell = ShellTools(cwd=tmp_path)
    m = await shell.read("c.txt", lines=(1, 1))
    await shell.replace(m, "z\n")
    assert f.read_text() == "z\nb\n"
