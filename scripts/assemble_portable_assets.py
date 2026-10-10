"""Copy every configured model and runtime resource into an onedir bundle.
ModelPaths/defaults.json is the sole model inventory; nested paths are deduplicated.
"""
import argparse, hashlib, json, shutil, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def model_roots(root=ROOT):
    config = json.loads((root/'settings'/'defaults.json').read_text(encoding='utf-8'))
    paths = {}
    for name, value in config['models'].items():
        relative = Path(value)
        source = (root/relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(root.resolve()):
            raise ValueError(f'Model {name} must have an application-relative path: {value}')
        if not source.exists(): raise FileNotFoundError(f'Model {name}: {source}')
        if source.is_dir() and not any(p.suffix.lower() in ('.bin', '.pth', '.pt', '.onnx', '.safetensors') for p in files_under(source)):
            raise FileNotFoundError(f'Model {name} has no weight payload: {source}')
        paths[name] = relative
    roots = []
    for path in sorted(set(paths.values()), key=lambda p: (len(p.parts), str(p))):
        if not any(path.is_relative_to(parent) for parent in roots): roots.append(path)
    return paths, roots


def files_under(path):
    if path.is_file(): yield path; return
    for item in path.rglob('*'):
        if item.is_file() and not any(p in ('__pycache__', '.git', 'logs', '.cache') for p in item.parts) and item.suffix != '.pyc':
            yield item


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def assemble(destination, root=ROOT):
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    models, roots = model_roots(root)
    for required in ('tools/ffmpeg/bin', 'nltk_data'):
        if not (root/required).is_dir(): raise FileNotFoundError(required)
        roots.append(Path(required))
    roots.append(Path('Vocal2Midi/inference'))
    roots.append(Path('vstudio_path_compat.py'))
    entries = []
    for relative in roots:
        print('RESOURCE', relative, flush=True)
        for source in files_under(root/relative):
            path = source.relative_to(root); target = destination/path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != source.stat().st_size or target.stat().st_mtime_ns != source.stat().st_mtime_ns:
                shutil.copy2(source, target)
            entries.append({'path':path.as_posix(), 'bytes':target.stat().st_size, 'sha256':digest(target)})
    # FunASR's registration and TorchScript inspect source at import time.
    # Keep these files physical even if modules are also present in the PYZ.
    for package in ('funasr', 'modelscope'):
        source_root = Path(importlib.util.find_spec(package).origin).parent
        for source in source_root.rglob('*.py'):
            path = Path(package)/source.relative_to(source_root); target = destination/path
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
            entries.append({'path':path.as_posix(), 'bytes':target.stat().st_size, 'sha256':digest(target)})
    data = {'schema':1, 'release_status':'UNVERIFIED', 'target':{'os':'Windows', 'gpu':'RX 7600M XT', 'arch':'gfx1102', 'rocm':'10.0.0', 'torch':'2.13.0+rocm10.0.0', 'onnx':'DirectML GPU (unsupported operators: CPU)'}, 'models':{k:v.as_posix() for k,v in models.items()}, 'files':entries}
    (destination.parent/'build-manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ASSETS_OK', len(models), 'models;', len(entries), 'files;', sum(f['bytes'] for f in entries), 'bytes', flush=True)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--dest', default='dist/VocalStudioBackend/_internal'); parser.add_argument('--inventory', action='store_true'); args = parser.parse_args()
    if args.inventory:
        models, roots = model_roots()
        print(json.dumps({'models':{k:str(v) for k,v in models.items()}, 'roots':[str(p) for p in roots]}, indent=2))
    else: assemble(args.dest)
