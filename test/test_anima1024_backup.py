"""A corrupt remote trainer state must never authorize local pruning."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.anima1024_campaign import file_hash, stage_is_saved, verified_upload


class FakeHub:
    def __init__(self, folder, corrupt=False):
        self.folder, self.corrupt = folder, corrupt

    def upload_folder(self, **kwargs):
        pass

    def get_paths_info(self, repo, paths):
        result = []
        for name in paths:
            p = self.folder / Path(name).name
            lfs = SimpleNamespace(sha256='invalid' if self.corrupt else file_hash(p)) if p.suffix == '.pt' else None
            result.append(SimpleNamespace(path=name, size=p.stat().st_size,
                                          lfs=lfs, blob_id=file_hash(p, 'sha1')))
        return result


def test_verifies_lfs_states_and_git_metadata(tmp_path):
    (tmp_path / 'state.pt').write_bytes(b'optimizer state')
    (tmp_path / 'latest').write_text('global_step10')
    result = verified_upload(FakeHub(tmp_path), 'public/repo', tmp_path, 'states/step10')
    assert len(result) == 2
    assert all(len(item['sha256']) == 64 for item in result)


def test_corrupt_remote_state_rejects_backup(tmp_path):
    (tmp_path / 'state.pt').write_bytes(b'irreplaceable optimizer state')
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        verified_upload(FakeHub(tmp_path, corrupt=True), 'public/repo', tmp_path, 'states/step10')
    assert (tmp_path / 'state.pt').exists()


def test_old_epoch_does_not_finish_later_stage(tmp_path):
    (tmp_path / 'latest').write_text('global_step6636')
    (tmp_path / 'epoch1').mkdir()
    (tmp_path / 'epoch1/adapter_model.safetensors').write_bytes(b'epoch one')
    assert not stage_is_saved(tmp_path, 7000)
    (tmp_path / 'latest').write_text('global_step7000')
    (tmp_path / 'step7000').mkdir()
    (tmp_path / 'step7000/adapter_model.safetensors').write_bytes(b'next stage')
    assert stage_is_saved(tmp_path, 7000)
