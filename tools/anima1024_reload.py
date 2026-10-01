#!/usr/bin/env python3
"""Apply an authorized campaign config change after a graceful saved handover.

The GPU cache keeps running. The old controller saves its first next step and
exits; the isolated queue then starts this helper to resume under the new limit.
A newer stop request cancels this automatic handover.
"""
import argparse
import json
import os
from pathlib import Path
import sys

try:
    from .anima1024_campaign import atomic_json
except ImportError:
    from anima1024_campaign import atomic_json


def consume_reload_request(artifacts):
    request_file = artifacts / 'reload_request.json'
    request = json.loads(request_file.read_text())
    stop_file = artifacts / 'stop_campaign'
    if (not stop_file.is_file() or stop_file.stat().st_mtime_ns != request['stop_mtime_ns']
            or stop_file.read_text() != request['stop_contents']):
        raise RuntimeError('A newer stop request superseded the authorized reload; keeping training stopped')
    state_path = artifacts / 'campaign_state.json'
    state = json.loads(state_path.read_text())
    if state['status'] != 'stopped':
        raise RuntimeError('Old controller has not completed the saved handover')
    state.update(status='ready', target_step=500, total_steps=request['max_steps'])
    atomic_json(state_path, state)
    # The launcher can remain alive briefly after the trainer consumes the
    # handover signal; remove any signal rewritten during that exit window.
    if state.get('run_dir'):
        (Path(state['run_dir']) / 'save_quit').unlink(missing_ok=True)
    stop_file.unlink()
    request_file.rename(artifacts / 'reload_applied.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--repo', required=True)
    args = parser.parse_args()
    consume_reload_request(args.artifacts)
    campaign = Path(__file__).with_name('anima1024_campaign.py')
    os.execv(sys.executable, [sys.executable, '-u', str(campaign), '--config', str(args.config),
                            '--artifacts', str(args.artifacts), '--repo', args.repo])


if __name__ == '__main__':
    main()
