"""UI regression tests with real ipywidgets, fake inference and optional real image export.

python -m pip install pillow ipywidgets nbformat
python -m unittest discover -s tests -p 'test_image_generator*.py' -v
No model downloads, GPU, Colab account or credentials needed.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import Mock

import test_image_generator_notebook as notebook

try:
    import ipywidgets
except ImportError:
    ipywidgets = None


class DimensionsTests(notebook.NotebookCase):
    def setUp(self):
        super().setUp()
        notebook.execute("ui-helpers", self.ns)

    def test_every_preset_ratio_resolution_is_exact_and_aligned(self):
        for model, preset in self.ns["MODEL_REGISTRY"].items():
            for ratio, (a, b) in self.ns["ASPECT_RATIOS"].items():
                for _, resolution in self.ns["RESOLUTION_OPTIONS"]:
                    with self.subTest(model=model, ratio=ratio, resolution=resolution):
                        w, h = self.ns["ui_dimensions"](model, ratio, resolution)
                        self.assertEqual(w * b, h * a)
                        self.assertEqual(w % preset["multiple"], 0)
                        self.assertEqual(h % preset["multiple"], 0)
                        self.assertTrue(128 <= w <= 4096 and 128 <= h <= 4096)
                        self.settings(MODEL=model, WIDTH=w, HEIGHT=h)

    def test_square_auto_uses_model_native_size(self):
        for model, preset in self.ns["MODEL_REGISTRY"].items():
            self.assertEqual(self.ns["ui_dimensions"](model, "1:1"), (preset["size"], preset["size"]))

    def test_unknown_or_invalid_custom_dimensions_fail(self):
        for args in (("Missing", "1:1"), ("SDXL", "17:2"), ("SDXL", "1:1", "Huge"),
                     ("SDXL", "Custom", "Auto", 129, 512), ("Custom", "Custom", "Auto", 520, 512)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.ns["ui_dimensions"](*args)
        self.assertEqual(self.ns["ui_dimensions"]("SDXL", "Custom", "Auto", 768, 1024), (768, 1024))

    def test_ui_snapshot_does_not_modify_defaults(self):
        values = dict(self.ns["DEFAULT_SETTINGS"])
        before = dict(values)
        settings = self.ns["ui_settings"](values, "9:16", "Auto")
        self.assertEqual(values, before)
        self.assertEqual(settings["width"] * 16, settings["height"] * 9)
        self.assertEqual(settings["ui"], dict(aspect_ratio="9:16", resolution="Auto"))


@unittest.skipIf(ipywidgets is None, "Install ipywidgets for actual widget controller tests")
class DashboardTests(notebook.NotebookCase):
    def setUp(self):
        super().setUp()
        self.ns.update(Path=Path, display=lambda *args: None)
        notebook.execute("ui-helpers", self.ns)
        notebook.execute("ui-dashboard", self.ns)
        self.ui = self.ns["GENERATOR_UI"]
        self.c = self.ui.controls
        self.addCleanup(self.ui.close)

    def test_controls_match_registry_and_opening_never_loads_model(self):
        self.assertEqual(set(self.c["MODEL"].options), {*self.ns["MODEL_REGISTRY"], "Custom"})
        self.assertFalse(notebook.Pipeline.loads)
        self.assertTrue(self.ui.download_button.disabled)
        self.assertIsNone(self.ui.advanced.selected_index)
        self.assertEqual(self.ui.custom_box.layout.display, "none")
        self.assertNotIn("HF_TOKEN", self.ui.defaults)
        self.assertNotIn("PIPE", self.ui.defaults)

    def test_each_model_restores_recommended_steps_guidance(self):
        self.c["PROMPT"].value = "My prompt"
        self.c["RATIO"].value = "16:9"
        for model, preset in self.ns["MODEL_REGISTRY"].items():
            with self.subTest(model=model):
                self.c["MODEL"].value = "Custom"
                self.c["LORA_SOURCE"].value = "org/old-lora"
                self.c["EXTRA_KWARGS_JSON"].value = '{"max_sequence_length":128}'
                self.c["STEPS"].value = 99
                self.c["GUIDANCE"].value = 22
                self.c["MODEL"].value = model
                self.assertEqual(self.c["STEPS"].value, preset["steps"])
                self.assertEqual(self.c["GUIDANCE"].value, preset["guidance"])
                self.assertEqual(self.c["LORA_SOURCE"].value, "")
                self.assertEqual(self.c["EXTRA_KWARGS_JSON"].value, "{}")
                self.assertEqual(self.c["PROMPT"].value, "My prompt")
                self.assertEqual(self.c["RATIO"].value, "16:9")

    def test_ratio_resolution_and_custom_pixel_controls(self):
        self.c["MODEL"].value = "SDXL"
        self.c["RATIO"].value = "9:16"
        self.c["RESOLUTION"].value = "768"
        settings = self.ui.snapshot()
        self.assertEqual(settings["width"] * 16, settings["height"] * 9)
        self.assertTrue(self.c["WIDTH"].disabled)
        self.assertIn(f'{settings["width"]} × {settings["height"]}', self.ui.dimension_preview.value)
        self.c["RATIO"].value = "Custom"
        self.assertFalse(self.c["WIDTH"].disabled)
        self.assertTrue(self.c["RESOLUTION"].disabled)
        self.c["WIDTH"].value, self.c["HEIGHT"].value = 640, 896
        self.assertEqual((self.ui.snapshot()["width"], self.ui.snapshot()["height"]), (640, 896))

    def test_custom_source_does_not_leak_into_preset_selection(self):
        self.c["MODEL"].value = "Custom"
        self.assertEqual(self.ui.custom_box.layout.display, "")
        self.c["CUSTOM_MODEL"].value = "/content/a.safetensors"
        self.c["SOURCE"].value = "Single safetensors file"
        self.c["PIPELINE_CLASS"].value = "StableDiffusionXLPipeline"
        self.assertEqual(self.ui.snapshot()["source"], "/content/a.safetensors")
        self.c["MODEL"].value = "SD 1.5"
        self.assertEqual(self.ui.snapshot()["source"], self.ns["MODEL_REGISTRY"]["SD 1.5"]["repo"])
        self.assertEqual(self.ui.snapshot()["source_type"], "Repository / directory")
        self.assertEqual(self.ui.snapshot()["pipeline"], "StableDiffusionPipeline")

    def test_random_and_fixed_seed_controls(self):
        self.assertTrue(self.c["SEED"].disabled)
        settings = self.ui.snapshot()
        self.assertTrue(0 <= settings["seed"] < 2**32)
        self.c["RANDOM_SEED"].value = False
        self.assertFalse(self.c["SEED"].disabled)
        self.c["SEED"].value = 123
        self.assertEqual(self.ui.snapshot()["seed"], 123)

    def test_button_reads_current_widgets_locks_and_downloads_own_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "result.zip"
            archive.touch()
            calls = []
            def generate(settings, progress_callback):
                calls.append(settings)
                self.assertTrue(self.ui.generate_button.disabled)
                self.assertTrue(self.c["MODEL"].disabled)
                self.ui._generate()  # A reentrant click must not start another job.
                progress_callback("loading", 0, settings["count"])
                progress_callback("saving", settings["count"], settings["count"])
                return archive
            self.ns["generate_images"] = generate
            self.c["PROMPT"].value = "Fresh widget prompt"
            self.ns["PROMPT"] = "Stale legacy prompt"
            self.c["NUM_IMAGES"].value = 2
            with redirect_stdout(io.StringIO()):
                self.ui.generate_button.click()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["prompt"], "Fresh widget prompt")
            self.assertEqual(calls[0]["count"], 2)
            self.assertEqual(self.ui.progress.value, 2)
            self.assertEqual(self.ui.last_archive, archive)
            self.assertFalse(self.ui.generate_button.disabled)
            self.assertFalse(self.ui.download_button.disabled)
            self.ns["LAST_ZIP"] = Path("unrelated-old-result.zip")
            download = self.ns["download_archive"] = Mock()
            self.ui.download_button.click()
            download.assert_called_once_with(archive)

    def test_validation_failure_disables_old_download_without_loading(self):
        generate = self.ns["generate_images"] = Mock()
        self.ui.last_archive = Path("old.zip")
        self.c["PROMPT"].value = "  "
        with redirect_stdout(io.StringIO()):
            self.ui._generate()
        generate.assert_not_called()
        self.assertIsNone(self.ui.last_archive)
        self.assertTrue(self.ui.download_button.disabled)
        self.assertFalse(self.ui.busy)
        self.assertFalse(self.ui.generate_button.disabled)
        self.assertEqual(self.ui.progress.bar_style, "danger")

    def test_errors_and_interrupts_restore_controls(self):
        for error, style in ((RuntimeError("model failed"), "danger"), (KeyboardInterrupt(), "warning")):
            self.ns["generate_images"] = Mock(side_effect=error)
            with self.subTest(error=type(error).__name__), redirect_stdout(io.StringIO()):
                self.ui._generate()
            self.assertFalse(self.ui.busy)
            self.assertFalse(self.c["MODEL"].disabled)
            self.assertTrue(self.ui.download_button.disabled)
            self.assertEqual(self.ui.progress.bar_style, style)

    def test_invalid_custom_resolution_is_not_silently_rounded(self):
        self.c["RATIO"].value = "Custom"
        self.c["WIDTH"].value = 129
        with self.assertRaises(ValueError):
            self.ui.snapshot()
        self.assertIn("128–4096", self.ui.summary.value)
        self.assertEqual(self.c["WIDTH"].value, 129)

    def test_status_escapes_markup(self):
        self.ui._set_status("<script>unsafe</script>")
        self.assertNotIn("<script>", self.ui.status.value)
        self.assertIn("&lt;script&gt;", self.ui.status.value)

    def test_close_detaches_callbacks(self):
        self.ui.close()
        self.assertEqual(self.ui.generate_button._click_handlers.callbacks, [])
        self.assertIsNone(self.ui.root.comm)

    @unittest.skipIf(notebook.Image is None, "Install Pillow for end-to-end mocked image export")
    def test_widget_to_engine_to_real_zip_with_metadata(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            self.ns.update(WORK_DIR=Path(tmp), Image=notebook.Image, platform=platform, version=lambda _: "test")
            notebook.execute("generation-engine", self.ns)
            self.c["RATIO"].value = "Custom"
            self.c["WIDTH"].value = self.c["HEIGHT"].value = 128
            self.c["RANDOM_SEED"].value = False
            self.c["SEED"].value = 123
            self.c["NUM_IMAGES"].value = 2
            with redirect_stdout(io.StringIO()):
                self.ui._generate()
            self.assertIsNotNone(self.ui.last_archive)
            with zipfile.ZipFile(self.ui.last_archive) as bundle:
                metadata = json.loads(bundle.read("metadata.json"))
                self.assertEqual(metadata["settings"]["ui"]["aspect_ratio"], "Custom")
                self.assertEqual([image["seed"] for image in metadata["images"]], [123, 124])
            self.assertEqual(self.ui.progress.value, 2)
            self.assertEqual(self.ui.progress.bar_style, "success")
            self.assertIsNone(self.ns["PIPE"])


if __name__ == "__main__":
    unittest.main()
