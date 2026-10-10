"""Confirm the frozen UI matches the reviewed source, including nested code."""
import argparse
import hashlib
import json
import types
from pathlib import Path
from PyInstaller.archive.readers import CArchiveReader

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('vstudio_ui.motion', 'vstudio_ui.acoustic', 'vstudio_ui.shell',
           'vstudio_ui.widgets.nav_rail', 'vstudio_ui.pages.pipeline',
           'vstudio_ui.pages.task_page', 'vstudio_ui.pages.overview')


def shape(code):
    if not isinstance(code, types.CodeType):
        return repr(code)
    return (code.co_code.hex(), code.co_argcount, code.co_kwonlyargcount,
            code.co_flags, code.co_names, code.co_varnames, code.co_freevars,
            code.co_cellvars, code.co_firstlineno, code.co_linetable.hex(),
            tuple(shape(value) for value in code.co_consts))


def verify(exe):
    archive = CArchiveReader(str(exe))
    pyz_name = next(name for name in archive.toc if name.endswith('.pyz'))
    pyz = archive.open_embedded_archive(pyz_name)
    checks = []
    for module in MODULES:
        path = ROOT / (module.replace('.', '/') + '.py')
        content = path.read_bytes()
        expected = compile(content, str(path), 'exec', optimize=0)
        actual = pyz.extract(module)
        if shape(expected) != shape(actual):
            raise ValueError('Frozen module differs from current source: ' + module)
        checks.append({'module':module, 'source_sha256':hashlib.sha256(content).hexdigest()})
    return {'status':'PASSED', 'exe':str(exe), 'modules':checks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = verify(Path(args.exe).resolve())
    Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('FROZEN_UI_SOURCES_MATCH', len(result['modules']))
