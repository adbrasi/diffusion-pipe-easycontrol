import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import krea2_fullbudget_campaign as campaign
from tools import krea2_upload_stage as upload


class CampaignSafetyTests(unittest.TestCase):
    def test_public_backups_require_explicit_mode_and_exclude_source_manifests(self):
        upload.check_visibility(private=False, public=True)
        upload.check_visibility(private=True, public=False)
        for private, public in [(True, True), (False, False)]:
            with self.assertRaisesRegex(RuntimeError, 'visibility'):
                upload.check_visibility(private, public)
        options = upload.artifact_upload_options(True)
        self.assertNotIn('data_plan_final.json', options['allow_patterns'])
        self.assertNotIn('hf_storage_audit.json', options['allow_patterns'])
        with patch.object(campaign, 'BACKUP_REPO', 'owner/public'), patch.object(campaign, 'BACKUP_PUBLIC', True):
            args = campaign.backup_args(Path('/new/run'), 500)
            self.assertIn('--public', args)
            self.assertEqual(args[args.index('--repo') + 1], 'owner/public')

    def test_training_losses_ignore_module_dump_and_preserve_nonfinite(self):
        logs = ('  loss: loss_fn\n'
                'steps: 249 loss: 0.0868 iter time (s): 2.784\n'
                'steps: 250 loss: 0.1612 iter time (s): 2.782\n')
        self.assertEqual(campaign.training_losses(logs), [0.0868, 0.1612])
        self.assertTrue(campaign.math.isnan(campaign.training_losses(
            'steps: 251 loss: nan iter time (s): 2.8\n')[0]))

    def recipe(self):
        return dict(output_dir=str(campaign.OUTPUT),
                    model=dict(type='krea2_native', base_quant='fp8_scaled'),
                    adapter=dict(rank=64), optimizer=dict(lr=.0004),
                    micro_batch_size_per_gpu=2, gradient_accumulation_steps=1)

    def test_campaign_lr_rejects_previous_lr_and_namespaces_jobs(self):
        recipe = self.recipe()
        with patch.object(campaign, 'LR', .0001), patch.object(campaign, 'JOB_PREFIX', 'fullbudget_lr1e4'):
            with self.assertRaisesRegex(RuntimeError, 'LR differs'):
                campaign.validate_recipe(recipe)
            recipe['optimizer']['lr'] = .0001
            campaign.validate_recipe(recipe)
            self.assertEqual(campaign.job_name('fullbudget_train000250'),
                             'fullbudget_lr1e4_train000250')

    def test_scratch_recipe_rejects_old_run_and_adapter_initialization(self):
        recipe = self.recipe()
        campaign.validate_recipe(recipe)
        bad = copy.deepcopy(recipe)
        bad['adapter']['init_from_existing'] = '/old/A1000'
        with self.assertRaisesRegex(RuntimeError, 'Fresh training'):
            campaign.validate_recipe(bad)
        bad = copy.deepcopy(recipe)
        bad['output_dir'] = '/old/A1000'
        with self.assertRaisesRegex(RuntimeError, 'Wrong training output'):
            campaign.validate_recipe(bad)

    def test_pruning_requires_backup_and_never_touches_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'fresh'
            run = output / 'run'
            art = root / 'artifacts'
            art.mkdir()
            for n in (250, 500, 750):
                (run / f'global_step{n}').mkdir(parents=True)
            (run / 'latest').write_text('global_step750')
            with patch.object(campaign, 'OUTPUT', output), patch.object(campaign, 'ART', art), patch.object(campaign, 'ROOT', root):
                campaign.prune(run)
                self.assertTrue((run / 'global_step250').is_dir())
                (art / 'backup_step250.json').write_text(json.dumps({'verified': True}))
                campaign.prune(run)
                self.assertFalse((run / 'global_step250').exists())
                self.assertTrue((run / 'global_step500').is_dir())
                self.assertTrue((run / 'global_step750').is_dir())
                baseline = root / 'A1000'
                baseline.mkdir()
                with self.assertRaisesRegex(RuntimeError, 'outside the new training run'):
                    campaign.prune(baseline)


if __name__ == '__main__':
    unittest.main()
