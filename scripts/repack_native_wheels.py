"""Repack existing CPython 3.13 native builds; retain hashes and local provenance.
Used when this Windows host lacks a C/C++ compiler. Not an upstream wheel.
"""
import argparse, csv, hashlib, base64, io, json, zipfile
from pathlib import Path
from importlib import metadata


def pack(name, out):
    dist = metadata.distribution(name)
    site = Path(dist.locate_file('')).resolve()
    package = name.replace('-', '_')
    info = Path(dist._path)
    wheel = f'{package}-{dist.version}-cp313-cp313-win_amd64.whl'
    records = []
    candidates = {path for directory in (site/package, info) for path in directory.rglob('*') if path.is_file()}
    # Some wheels also install compiled modules at site-packages top level.
    for entry in dist.files or ():
        path = Path(dist.locate_file(entry)).resolve()
        if path.is_relative_to(site) and path.is_file(): candidates.add(path)
    with zipfile.ZipFile(out / wheel, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(candidates):
                if '__pycache__' in path.parts or path.name in ('RECORD', 'INSTALLER', 'REQUESTED', 'direct_url.json'): continue
                relative = path.relative_to(site).as_posix()
                data = path.read_bytes(); archive.writestr(relative, data)
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                records.append([relative, 'sha256='+digest, str(len(data))])
        relative = info.name+'/RECORD'
        records.append([relative, '', ''])
        stream = io.StringIO(); csv.writer(stream, lineterminator='\n').writerows(records)
        archive.writestr(relative, stream.getvalue())
    return {'package':name, 'version':dist.version, 'wheel':wheel, 'sha256':hashlib.sha256((out/wheel).read_bytes()).hexdigest(), 'provenance':'Repacked installed native build from '+str(site/package)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', default='temp/rocm10-wheels')
    parser.add_argument('--packages', nargs='+', default=['pyopenjtalk', 'madmom', 'jieba_fast']); args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((out/'provenance.json').read_text()) if (out/'provenance.json').exists() else []
    result = [item for item in previous if item['package'] not in args.packages]
    result += [pack(name, out) for name in args.packages]
    (out/'provenance.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
