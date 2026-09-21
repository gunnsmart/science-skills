"""Second-pass regressions for upstream API differences and error cleanup."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_image_generator_notebook as notebook


class LoaderFollowupTests(notebook.NotebookCase):
    def test_uppercase_checkpoint_suffix_is_rejected_before_loader(self):
        for source in ('/content/model.SAFETENSORS', 'https://huggingface.co/org/repo/resolve/main/model.SAFETENSORS'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.settings(MODEL='Custom', SOURCE='Single safetensors file',
                              CUSTOM_MODEL=source, PIPELINE_CLASS='StableDiffusionPipeline')
        with self.assertRaises(ValueError):
            self.settings(LORA_SOURCE='org/lora', LORA_WEIGHT_NAME='adapter.SAFETENSORS')
        self.assertFalse(notebook.Pipeline.loads)

    def test_local_checkpoint_suffix_cannot_hide_query_fragment_or_newline(self):
        sources = ['/content/model.safetensors' + suffix for suffix in ('?download=1', '#note', '?', '#')]
        sources += ['/content/model.safe\ntensors', '/content/model.safe\rtensors']
        for source in sources:
            with self.subTest(source=repr(source)), self.assertRaises(ValueError):
                self.settings(MODEL='Custom', SOURCE='Single safetensors file',
                              CUSTOM_MODEL=source,
                              PIPELINE_CLASS='StableDiffusionPipeline')
        self.assertFalse(notebook.Pipeline.loads)

    def test_modern_tiling_preferred_and_actual_status_recorded(self):
        modern, legacy = Mock(), Mock()
        pipe = SimpleNamespace(vae=SimpleNamespace(enable_tiling=modern),
                               enable_vae_tiling=legacy, to=Mock())
        self.ns['CUSTOM_LOADERS']['modern'] = lambda settings, dtype: pipe
        self.ns['get_pipeline'](self.settings(BACKEND='modern'))
        modern.assert_called_once_with()
        legacy.assert_not_called()
        self.assertEqual(self.ns['PIPE_LOAD_INFO']['vae_tiling'], 'enabled')
        self.ns['unload_model']()
        self.assertEqual(self.ns['PIPE_LOAD_INFO'], {})

    def test_legacy_tiling_still_supported_and_disabled_tiling_not_called(self):
        pipe, _, _ = self.ns['get_pipeline'](self.settings())
        self.assertIn('tiling', pipe.events)
        pipe, _, _ = self.ns['get_pipeline'](self.settings(VAE_TILING=False))
        self.assertNotIn('tiling', pipe.events)
        self.assertEqual(self.ns['PIPE_LOAD_INFO']['vae_tiling'], 'off')

    def test_local_lora_with_unsafe_suffix_is_rejected_before_base_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'adapter.SAFETENSORS'
            file.touch()
            with self.assertRaises(ValueError):
                self.ns['get_pipeline'](self.settings(LORA_SOURCE=str(file)))
            self.assertFalse(notebook.Pipeline.loads)

    def test_tiling_uses_vae_api_when_pipeline_wrapper_is_absent(self):
        # PixArtSigmaPipeline / ZImagePipeline in Diffusers 0.36 expose a VAE
        # without the old pipeline-level enable_vae_tiling wrapper.
        tiling = Mock()
        pipe = SimpleNamespace(vae=SimpleNamespace(enable_tiling=tiling), to=Mock())
        self.ns['CUSTOM_LOADERS']['modern'] = lambda settings, dtype: pipe
        with redirect_stdout(io.StringIO()):
            self.ns['get_pipeline'](self.settings(BACKEND='modern'))
        tiling.assert_called_once_with()

    def test_private_lora_autodiscovery_uses_explicit_token(self):
        class StrictPipeline(notebook.Pipeline):
            def load_lora_weights(self, source, **kwargs):
                if not kwargs.get('weight_name'):
                    raise AssertionError('would invoke upstream unauthenticated filename guessing')
                super().load_lora_weights(source, **kwargs)
        self.diffusers.StableDiffusionPipeline = StrictPipeline
        listing = Mock(return_value=['README.md', 'nested/adapter.safetensors'])
        with patch.dict('sys.modules', {'huggingface_hub':SimpleNamespace(list_repo_files=listing)}):
            pipe, _, _ = self.ns['get_pipeline'](self.settings(LORA_SOURCE='org/private-lora'))
        listing.assert_called_once_with(repo_id='org/private-lora', token='private-test-token')
        event = next(e for e in pipe.events if isinstance(e, tuple) and e[0] == 'lora')
        self.assertEqual(event[2]['weight_name'], 'nested/adapter.safetensors')

    def test_multiple_or_missing_lora_files_fail_before_base_model_load(self):
        for filenames in (['a.safetensors', 'b.safetensors'], ['README.md', 'old.bin']):
            listing = Mock(return_value=filenames)
            with self.subTest(filenames=filenames), patch.dict('sys.modules', {'huggingface_hub':SimpleNamespace(list_repo_files=listing)}):
                with self.assertRaises(ValueError):
                    self.ns['get_pipeline'](self.settings(LORA_SOURCE='org/lora'))
            self.assertFalse(notebook.Pipeline.loads)

    def test_local_single_lora_file_bypasses_upstream_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'adapter.safetensors'
            file.touch()
            pipe, _, _ = self.ns['get_pipeline'](self.settings(LORA_SOURCE=str(file)))
            event = next(e for e in pipe.events if isinstance(e, tuple) and e[0] == 'lora')
            self.assertEqual(event[1], str(file.parent))
            self.assertEqual(event[2]['weight_name'], file.name)

    def test_local_directory_with_one_lora_needs_no_hub_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'adapter.safetensors').touch()
            (Path(tmp) / 'README.md').touch()
            pipe, _, _ = self.ns['get_pipeline'](self.settings(LORA_SOURCE=tmp))
            event = next(e for e in pipe.events if isinstance(e, tuple) and e[0] == 'lora')
            self.assertEqual(event[2]['weight_name'], 'adapter.safetensors')

    def test_cached_lora_does_not_repeat_repo_listing(self):
        listing = Mock(return_value=['adapter.safetensors'])
        settings = self.settings(LORA_SOURCE='org/lora')
        with patch.dict('sys.modules', {'huggingface_hub':SimpleNamespace(list_repo_files=listing)}):
            first = self.ns['get_pipeline'](settings)[0]
            second = self.ns['get_pipeline']({**settings, 'prompt':'new prompt'})[0]
        self.assertIs(first, second)
        self.assertEqual(listing.call_count, 1)

    def test_loader_cleanup_error_does_not_mask_original_failure(self):
        self.cuda.is_available = lambda: True
        self.cuda.empty_cache = Mock(side_effect=[None, RuntimeError('cleanup error')])
        def fail(settings, dtype):
            raise ValueError('original loader failure')
        self.ns['CUSTOM_LOADERS']['fail'] = fail
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'original loader failure'):
            self.ns['get_pipeline'](self.settings(BACKEND='fail'))
        self.assertIsNone(self.ns['PIPE'])
        self.assertIsNone(self.ns['PIPE_KEY'])


@unittest.skipIf(notebook.Image is None, 'Pillow needed')
class ArchiveFollowupTests(notebook.NotebookCase):
    setup_export = notebook.ExportTests.setup_export

    @unittest.skipUnless(os.name == 'posix', 'POSIX file permissions')
    def test_zip_is_private_even_with_permissive_umask(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.setup_export(tmp)
            previous = os.umask(0o022)
            try:
                with redirect_stdout(io.StringIO()):
                    archive = self.ns['generate_images'](self.settings(WIDTH=128, HEIGHT=128))
            finally:
                os.umask(previous)
            self.assertEqual(stat.S_IMODE(archive.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(self.ns['LAST_RUN_DIR'].stat().st_mode), 0o700)
