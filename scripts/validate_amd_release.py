"""Real AMD acceptance. Requires an assembled exe and supplied local test clips.
This script never marks the bundle usable unless every required case passes.
"""
import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path


def finalize_bundle(bundle, report_path):
    """Mark only the exact binary/assets that passed all required target checks."""
    bundle = Path(bundle); report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('environment_policy') != 'isolated_no_developer_gpu_mask_v1':
        raise ValueError('Clean-environment AMD acceptance is required')
    required = {'ui_smoke','info','separation','midi','asr_speaker','tts','singing_pipeline','speech_pipeline'}
    cases = {item['name']:item for item in report['cases']}
    if report['status'] != 'PASSED' or not required.issubset(cases) or not all(cases[name]['ok'] for name in required):
        raise ValueError('Required AMD acceptance has not passed')
    if cases['midi'].get('directml_gpu_node_events',0) <= 0: raise ValueError('No actual DirectML GPU events')
    tested = Path(report['exe'])
    def sha(path):
        with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
    binary = bundle/tested.name
    if sha(binary) != sha(tested): raise ValueError('Binary differs from the tested release')
    manifest = bundle/'build-manifest.json'
    data = json.loads(manifest.read_text(encoding='utf-8'))
    tested_data = json.loads((tested.parent/'build-manifest.json').read_text(encoding='utf-8'))
    if data['files'] != tested_data['files']: raise ValueError('Asset inventory differs from the tested release')
    data.update(release_status='VERIFIED_RX7600M_XT', executable_sha256=sha(binary), acceptance='amd-acceptance.json', pending_ui_checks=['Cross-monitor DPI; only one monitor connected', 'Manual drag/snap interaction'])
    manifest.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(report_path,bundle/'amd-acceptance.json')
    return data


def run_release(executable, singing, speech, reference, out):
    executable = executable.resolve(); out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'VOCALSTUDIO_SMOKE', 'VOCALSTUDIO_UI_SMOKE', 'VOCALSTUDIO_MIOPEN'):
        env.pop(key, None)
    for key in list(env):
        if key.startswith(('CUDA_', 'ROCM_', 'ROCR_', 'HIP_', 'HSA_')) or key in ('AMD_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL', 'FFMPEG_DIR'):
            env.pop(key, None)
    env['PATH'] = os.pathsep.join([str(executable.parent/'_internal'/'tools'/'ffmpeg'/'bin'), os.path.join(env.get('SystemRoot', 'C:\\Windows'), 'System32'), env.get('SystemRoot', 'C:\\Windows')])
    env.update(PYTHONUTF8='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    report = {'status':'UNVERIFIED', 'exe':str(executable), 'environment_policy':'isolated_no_developer_gpu_mask_v1', 'cases':[]}
    cases = [
        ('info', ['info'], (), ''),
        ('separation', ['separate', '--audio', str(singing), '--model', 'Karaoke'], ('.wav',), ''),
        ('midi', ['transcribe', '--audio', str(singing), '--lang', 'ja', '--formats', 'mid,txt'], ('.mid', '.txt'), 'GPU=AMD Radeon RX 7600M XT'),
        ('asr_speaker', ['asr', '--audio', str(speech), '--speakers', '1', '--lang', 'en', '--srt', '--fmt', 'txt,srt,json'], ('.txt', '.srt', '.json', '.wav'), 'diarization done'),
        ('tts', ['tts', '--ref-audio', str(reference), '--lang', 'en', '--text', 'Hello from Vocal Studio. Every voice has more possibilities.'], ('.wav',), ''),
        ('singing_pipeline', ['pipeline', '--audio', str(singing), '--steps', 'separation,content_detection,vocal2midi,speech_asr', '--model', 'Karaoke', '--lang', 'ja'], ('.wav', '.mid', '.txt'), 'vocal2midi done'),
        ('speech_pipeline', ['pipeline', '--audio', str(speech), '--steps', 'diarization,content_detection,speech_asr', '--speakers', '1', '--lang', 'en', '--srt'], ('.txt', '.srt'), 'speech_asr done'),
    ]
    def record(name, args, extensions, marker='', smoke=False):
        destination = out/name
        if destination.exists(): raise FileExistsError(f'Use a fresh acceptance output directory: {destination}')
        destination.mkdir()
        cmd = [str(executable), *args]
        if name not in ('info','ui_smoke'): cmd += ['--device','rocm','--output-dir',str(destination)]
        settings = env.copy()
        if smoke: settings.update(VOCALSTUDIO_SMOKE='1', QT_QPA_PLATFORM='offscreen')
        if name == 'midi': settings['VOCALSTUDIO_ORT_PROFILE_DIR'] = str(destination/'onnx-profile')
        start = time.monotonic()
        try:
            process = subprocess.run(cmd, cwd=out, env=settings, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace', timeout=1200)
            text = process.stdout; code = process.returncode
        except subprocess.TimeoutExpired as exc:
            text = str(exc); code = -1
        (out/(name+'.log')).write_text(text, encoding='utf-8')
        files = [p for p in destination.rglob('*') if p.is_file()]
        missing = [ext for ext in extensions if not any(p.suffix == ext and p.stat().st_size > 0 for p in files)]
        ok = code == 0 and not missing and (not marker or marker in text)
        if name == 'info': ok = ok and '2.13.0+rocm10.0.0' in text and '7600M XT' in text and '22 ready, 0 missing' in text
        if smoke: ok = ok and 'SMOKE_FROZEN_PASS' in text
        gpu_nodes = 0
        if name == 'midi':
            for profile in (destination/'onnx-profile').glob('*.json'):
                events = json.loads(profile.read_text(encoding='utf-8'))
                gpu_nodes += sum(event.get('args',{}).get('provider') == 'DmlExecutionProvider' for event in events)
            ok = ok and gpu_nodes > 0
        item = dict(name=name, ok=ok, exit_code=code, seconds=round(time.monotonic()-start,2), missing=missing, files=[str(p.relative_to(out)) for p in files])
        if name == 'midi': item['directml_gpu_node_events'] = gpu_nodes
        report['cases'].append(item)
        (out/'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(item), flush=True)
    record('ui_smoke', [], (), smoke=True)
    for case in cases: record(*case)
    report['status'] = 'PASSED' if all(case['ok'] for case in report['cases']) else 'FAILED'
    (out/'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if report['status'] == 'PASSED': finalize_bundle(executable.parent, out/'acceptance.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--exe', required=True)
    parser.add_argument('--singing', required=True); parser.add_argument('--speech', required=True); parser.add_argument('--reference', required=True)
    parser.add_argument('--out', required=True); parser.add_argument('--relocate-to')
    args = parser.parse_args(); exe = Path(args.exe).resolve()
    if args.relocate_to:
        relocated = Path(args.relocate_to).resolve()
        if relocated.exists(): raise FileExistsError('Relocation destination must be new')
        shutil.copytree(exe.parent, relocated)
        exe = relocated/exe.name
    report = run_release(exe, Path(args.singing).resolve(), Path(args.speech).resolve(), Path(args.reference).resolve(), Path(args.out).resolve())
    raise SystemExit(0 if report['status'] == 'PASSED' else 1)
