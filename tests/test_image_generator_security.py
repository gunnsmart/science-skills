"""Offline adversarial/error-path tests; no untrusted code or model weights execute."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import zipfile

import test_image_generator_notebook as notebook
import test_image_generator_ui as ui_tests


class ValidationSecurityTests(notebook.NotebookCase):
    def test_encoded_hf_filename_must_not_escape_repository(self):
        for filename in ('%2Ftmp/escape.safetensors', '..%2Fescape.safetensors',
                         'nested/%2E%2E/escape.safetensors', '%5Ctmp%5Cescape.safetensors',
                         'nested/%00escape.safetensors'):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.settings(MODEL='Custom', SOURCE='Single safetensors file',
                              PIPELINE_CLASS='StableDiffusionPipeline',
                              CUSTOM_MODEL='https://huggingface.co/org/repo/resolve/main/' + filename)

    def test_optional_model_sources_reject_credential_urls(self):
        for field in ('LORA_SOURCE', 'SINGLE_FILE_CONFIG'):
            for url in ('https://user:secret@host/model', 'https://host/model?token=secret'):
                with self.subTest(field=field, url=url), self.assertRaises(ValueError):
                    self.settings(**{field: url})

    def test_lora_filename_validation_happens_before_weight_loading(self):
        for name in ('../../outside.safetensors', '/tmp/outside.safetensors', 'unsafe.bin'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.settings(LORA_SOURCE='org/lora', LORA_WEIGHT_NAME=name)
        self.assertFalse(notebook.Pipeline.loads)

    def test_large_or_overdeep_extra_json_fails_cleanly(self):
        for raw in ('{"data":"' + 'a' * 70000 + '"}', '{"data":' + '[' * 100 + '0' + ']' * 100 + '}'):
            with self.assertRaises(ValueError):
                self.settings(EXTRA_KWARGS_JSON=raw)

    def test_valid_nested_checkpoint_and_encoded_branch_still_work(self):
        source = 'https://huggingface.co/org/repo/resolve/feature%2Fv2/nested/model.safetensors'
        self.assertEqual(self.ns['single_file_location'](source), dict(
            repo_id='org/repo', revision='feature/v2', filename='nested/model.safetensors'))
        self.settings(LORA_SOURCE='org/lora', LORA_WEIGHT_NAME='nested/adapter.safetensors')

    def test_redaction_handles_nested_fields_and_does_not_mutate_input(self):
        value = {'prompt': 'private-test-token', 'nested': [{'api_key': 'another-secret'}],
                 'url': 'https://user:pass@host/path?token=some-secret', 'count': 2}
        clean = self.ns['redact_metadata'](value)
        self.assertEqual(clean['nested'][0]['api_key'], '[REDACTED]')
        self.assertEqual(clean['count'], 2)
        self.assertNotIn('private-test-token', json.dumps(clean))
        self.assertNotIn('some-secret', clean['url'])
        self.assertNotIn('user:pass', clean['url'])
        self.assertEqual(value['prompt'], 'private-test-token')

    def test_interrupted_loader_clears_allocated_pipeline(self):
        class InterruptedPipeline(notebook.Pipeline):
            def enable_vae_tiling(self):
                raise KeyboardInterrupt()
        self.ns['CUSTOM_LOADERS']['interrupted'] = lambda settings, dtype: InterruptedPipeline()
        with self.assertRaises(KeyboardInterrupt):
            self.ns['get_pipeline'](self.settings(BACKEND='interrupted'))
        self.assertIsNone(self.ns['PIPE'])
        self.assertIsNone(self.ns['PIPE_KEY'])


@unittest.skipIf(ui_tests.ipywidgets is None, 'ipywidgets needed')
class KernelOutputTests(notebook.NotebookCase):
    def setUp(self):
        super().setUp()
        self.ns.update(Path=Path, display=lambda *args: None)
        notebook.execute('ui-helpers', self.ns)
        notebook.execute('ui-dashboard', self.ns)
        self.ui = self.ns['GENERATOR_UI']
        self.addCleanup(self.ui.close)

    def test_ipython_output_suppression_cannot_turn_failure_into_success(self):
        # Output.__exit__ returns True in a real IPython kernel. Headless tests
        # previously exercised only the non-suppressing branch.
        self.ns['generate_images'] = Mock(side_effect=RuntimeError('weights unavailable'))
        shell = SimpleNamespace(showtraceback=Mock())
        # Exercise the real Output.__exit__ with an IPython-like shell, rather
        # than relying only on a mocked context manager's return value.
        with patch('ipywidgets.widgets.widget_output.get_ipython', return_value=shell), redirect_stdout(io.StringIO()):
            self.ui._generate()
        shell.showtraceback.assert_not_called()  # Our handler caught the error first.
        self.assertEqual(self.ui.progress.bar_style, 'danger')
        self.assertIn('ไม่สำเร็จ', self.ui.status.value)
        self.assertIsNone(self.ui.last_archive)
        self.assertTrue(self.ui.download_button.disabled)
        self.assertFalse(self.ui.busy)

    def test_ui_exception_output_does_not_echo_auth_secret(self):
        self.ns['generate_images'] = Mock(side_effect=RuntimeError('Authorization: private-test-token'))
        text = io.StringIO()
        with redirect_stdout(text):
            self.ui._generate()
        self.assertNotIn('private-test-token', text.getvalue())
        self.assertIn('[REDACTED]', text.getvalue())

    def test_kernel_output_suppression_preserves_interrupt_and_validation_state(self):
        for mode in ('interrupt', 'invalid-prompt'):
            self.ui.controls['PROMPT'].value = 'valid' if mode == 'interrupt' else ''
            self.ns['generate_images'] = Mock(side_effect=KeyboardInterrupt())
            with self.subTest(mode=mode), patch.object(type(self.ui.output), '__exit__', return_value=True), redirect_stdout(io.StringIO()):
                self.ui._generate()
            self.assertEqual(self.ui.progress.bar_style, 'warning' if mode == 'interrupt' else 'danger')
            self.assertTrue(self.ui.download_button.disabled)
            self.assertFalse(self.ui.busy)

    def test_engine_returning_missing_archive_is_not_success(self):
        self.ns['generate_images'] = Mock(return_value='/nonexistent/missing-output.zip')
        with redirect_stdout(io.StringIO()):
            self.ui._generate()
        self.assertEqual(self.ui.progress.bar_style, 'danger')
        self.assertIsNone(self.ui.last_archive)


@unittest.skipIf(notebook.Image is None, 'Pillow needed')
class ExportSecurityTests(notebook.NotebookCase):
    setup_export = notebook.ExportTests.setup_export

    def test_zip_contains_only_generated_manifest_not_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.setup_export(tmp)
            secret = Path(tmp) / 'unrelated-private.txt'
            secret.write_text('private data')
            def callback(stage, *_):
                if stage == 'saving':
                    run = next(p for p in Path(tmp).iterdir() if p.is_dir())
                    (run / 'unrelated.txt').write_text('not a generated image')
                    (run / 'secret-link.txt').symlink_to(secret)
            with redirect_stdout(io.StringIO()):
                archive = self.ns['generate_images'](self.settings(WIDTH=128, HEIGHT=128), progress_callback=callback)
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(set(bundle.namelist()), {'metadata.json', 'README.txt',
                    'image_001_seed_42.png', 'image_001_seed_42.jpg'})

    def test_symlink_replacing_generated_file_is_not_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.setup_export(tmp)
            secret = Path(tmp) / 'private.txt'
            secret.write_text('private data')
            def callback(stage, *_):
                if stage == 'saving':
                    image = next(Path(tmp).glob('*/image_*.png'))
                    image.unlink()
                    image.symlink_to(secret)
            with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                self.ns['generate_images'](self.settings(WIDTH=128, HEIGHT=128), progress_callback=callback)
            self.assertIsNone(self.ns['LAST_ZIP'])
            self.assertFalse(list(Path(tmp).glob('*.zip')))

    def test_export_redacts_known_token_in_user_supplied_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.setup_export(tmp)
            settings = self.settings(WIDTH=128, HEIGHT=128, PROMPT='accidentally pasted private-test-token')
            with redirect_stdout(io.StringIO()):
                archive = self.ns['generate_images'](settings)
            with zipfile.ZipFile(archive) as bundle:
                raw = bundle.read('metadata.json').decode()
                self.assertNotIn('private-test-token', raw)
                self.assertIn('[REDACTED]', raw)
            self.assertEqual(settings['prompt'], 'accidentally pasted private-test-token')

    def test_metadata_write_failure_does_not_mask_original_disk_error(self):
        original = Path.write_text
        def fail_metadata(path, data, *args, **kwargs):
            if 'metadata.json' in path.name:
                original(path, '{"truncated":', *args, **kwargs)
                raise OSError('original disk full')
            return original(path, data, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            self.setup_export(tmp)
            with patch.object(Path, 'write_text', new=fail_metadata):
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(OSError, 'original disk full'):
                    self.ns['generate_images'](self.settings(WIDTH=128, HEIGHT=128))
            self.assertIsNone(self.ns['LAST_ZIP'])
            self.assertIsNone(self.ns['PIPE'])
            self.assertFalse(list(Path(tmp).glob('*.zip')))

    def test_failed_metadata_replace_keeps_previous_complete_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.setup_export(tmp)
            run = Path(tmp) / 'run'
            run.mkdir()
            self.ns['write_metadata'](run, {'status': 'running', 'images': [1]})
            with patch.object(Path, 'replace', side_effect=OSError('replace failed')):
                with self.assertRaisesRegex(OSError, 'replace failed'):
                    self.ns['write_metadata'](run, {'status': 'complete', 'images': [1, 2]})
            self.assertEqual(json.loads((run / 'metadata.json').read_text())['images'], [1])
            self.assertFalse((run / 'metadata.json.part').exists())
