import json

import pytest

from tools.anima1024_reload import consume_reload_request


def prepare_request(root):
    run = root / 'run'
    run.mkdir()
    (run / 'save_quit').touch()
    stop = root / 'stop_campaign'
    stop.write_text('authorized step-limit update')
    (root / 'reload_request.json').write_text(json.dumps(dict(
        max_steps=33000, stop_mtime_ns=stop.stat().st_mtime_ns,
        stop_contents=stop.read_text())))
    (root / 'campaign_state.json').write_text(json.dumps(dict(
        status='stopped', target_step=500, run_dir=str(run))))


def test_reload_preserves_run_and_clears_only_handover_signals(tmp_path):
    prepare_request(tmp_path)
    consume_reload_request(tmp_path)
    state = json.loads((tmp_path / 'campaign_state.json').read_text())
    assert state['total_steps'] == 33000 and state['status'] == 'ready'
    assert state['run_dir'] == str(tmp_path / 'run')
    assert not (tmp_path / 'run/save_quit').exists()
    assert not (tmp_path / 'stop_campaign').exists()
    assert (tmp_path / 'reload_applied.json').exists()


def test_newer_stop_request_cancels_automatic_resume(tmp_path):
    prepare_request(tmp_path)
    (tmp_path / 'stop_campaign').write_text('user explicitly stopped training')
    with pytest.raises(RuntimeError, match='newer stop request'):
        consume_reload_request(tmp_path)
    assert (tmp_path / 'stop_campaign').exists()
    assert json.loads((tmp_path / 'campaign_state.json').read_text())['status'] == 'stopped'
