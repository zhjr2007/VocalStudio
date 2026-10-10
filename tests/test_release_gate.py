import json
from pathlib import Path
import pytest
from scripts.validate_amd_release import finalize_bundle


def candidate(tmp_path):
    tested=tmp_path/'tested'; bundle=tmp_path/'bundle'; tested.mkdir(); bundle.mkdir()
    for root in (tested,bundle):
        (root/'VocalStudioBackend.exe').write_bytes(b'tested binary')
        (root/'build-manifest.json').write_text(json.dumps(dict(release_status='UNVERIFIED',files=[])))
    names=('ui_smoke','info','separation','midi','asr_speaker','tts','singing_pipeline','speech_pipeline')
    report=dict(status='PASSED',environment_policy='isolated_no_developer_gpu_mask_v1',exe=str(tested/'VocalStudioBackend.exe'),cases=[dict(name=n,ok=True,directml_gpu_node_events=1 if n=='midi' else 0) for n in names])
    path=tmp_path/'acceptance.json'
    path.write_text(json.dumps(report))
    return bundle,path,report


def test_failed_acceptance_cannot_mark_bundle_usable(tmp_path):
    bundle,path,report=candidate(tmp_path); report['cases'][-1]['ok']=False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='has not passed'): finalize_bundle(bundle,path)
    assert json.loads((bundle/'build-manifest.json').read_text())['release_status']=='UNVERIFIED'


def test_changed_binary_cannot_reuse_another_packages_acceptance(tmp_path):
    bundle,path,_=candidate(tmp_path); (bundle/'VocalStudioBackend.exe').write_bytes(b'new binary')
    with pytest.raises(ValueError,match='Binary differs'): finalize_bundle(bundle,path)


def test_matching_verified_binary_records_target_and_pending_checks(tmp_path):
    bundle,path,_=candidate(tmp_path)
    manifest=finalize_bundle(bundle,path)
    assert manifest['release_status']=='VERIFIED_RX7600M_XT'
    assert manifest['pending_ui_checks'] and (bundle/'amd-acceptance.json').is_file()


def test_acceptance_with_inherited_gpu_masks_cannot_mark_bundle_usable(tmp_path):
    bundle,path,report=candidate(tmp_path); report.pop('environment_policy')
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Clean-environment'): finalize_bundle(bundle,path)
