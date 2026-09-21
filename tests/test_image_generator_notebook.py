"""Offline notebook tests; no model weights, credentials, network or GPU needed.

Run: python -m unittest discover -s tests -p 'test_image_generator*.py' -v
Optional Pillow enables the real PNG/JPEG/ZIP export tests.
"""
import ast
from contextlib import nullcontext, redirect_stdout
from datetime import datetime, timezone
import gc
import io
import json
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    from PIL import Image
except ImportError:
    Image = None

NOTEBOOK = Path(__file__).resolve().parents[1] / "Image_generator.ipynb"
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = {cell["id"]: "".join(cell["source"]) for cell in NB["cells"]}


def execute(cell_id, namespace):
    with redirect_stdout(io.StringIO()):
        exec(compile(CELLS[cell_id], f"{NOTEBOOK.name}:{cell_id}", "exec"), namespace)


class Generator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class Pipeline:
    loads = []

    def __init__(self):
        self.events = []
        self.calls = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.loads.append(("repo", source, kwargs))
        return cls()

    @classmethod
    def from_single_file(cls, source, **kwargs):
        cls.loads.append(("single", source, kwargs))
        return cls()

    def to(self, device):
        self.events.append(("to", device))
        return self

    def enable_vae_tiling(self):
        self.events.append("tiling")

    def enable_model_cpu_offload(self):
        self.events.append("offload")

    def enable_sequential_cpu_offload(self):
        self.events.append("sequential")

    def load_lora_weights(self, source, **kwargs):
        self.events.append(("lora", source, kwargs))

    def set_adapters(self, adapters, adapter_weights):
        self.events.append(("adapters", adapters, adapter_weights))

    def __call__(self, prompt, width, height, num_inference_steps, generator,
                 negative_prompt=None, guidance_scale=1, num_images_per_prompt=1,
                 output_type="pil", return_dict=True):
        self.calls.append(dict(seed=generator.seed, width=width, height=height))
        if Image is None:
            raise RuntimeError("Pillow required for inference mock")
        return SimpleNamespace(images=[Image.new("RGBA", (width, height), (0, 80, 255, 128))])


class NotebookCase(unittest.TestCase):
    def setUp(self):
        Pipeline.loads = []
        self.cuda = SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False,
                                    empty_cache=lambda: None)
        self.torch = SimpleNamespace(cuda=self.cuda, float16="fp16", bfloat16="bf16", float32="fp32",
                                     Generator=Generator, inference_mode=nullcontext)
        self.diffusers = SimpleNamespace(DiffusionPipeline=Pipeline, StableDiffusionPipeline=Pipeline,
                                         StableDiffusionXLPipeline=Pipeline)
        self.ns = dict(torch=self.torch, diffusers=self.diffusers, HF_TOKEN="private-test-token", gc=gc)
        for cell in ("registry", "settings", "validation", "loader"):
            execute(cell, self.ns)

    def settings(self, **overrides):
        return self.ns["resolve_settings"]({**self.ns, **overrides})


class NotebookTest(NotebookCase):
    def test_qwen_guidance_control_routes_to_true_cfg(self):
        class QwenImagePipeline:
            def __call__(self, prompt, width, height, num_inference_steps, generator,
                         negative_prompt=None, guidance_scale=None, true_cfg_scale=4):
                pass
        for values in (dict(MODEL="Qwen Image"), dict(MODEL="Custom", CUSTOM_MODEL="org/qwen")):
            kwargs, _ = self.ns["build_call_kwargs"](QwenImagePipeline(), self.settings(**values, GUIDANCE=7), None)
            self.assertEqual(kwargs["true_cfg_scale"], 7)
            self.assertNotIn("guidance_scale", kwargs)

    def test_qwen_guidance_cannot_be_overridden_by_extra_kwargs(self):
        class QwenImagePipeline:
            def __call__(self, prompt, width, height, num_inference_steps, generator,
                         negative_prompt=None, guidance_scale=None, true_cfg_scale=4):
                pass
        with self.assertRaisesRegex(ValueError, "GUIDANCE"):
            self.ns["build_call_kwargs"](QwenImagePipeline(), self.settings(
                MODEL="Qwen Image", EXTRA_KWARGS_JSON='{"true_cfg_scale":8}'), None)

    def test_single_file_revision_conflict_and_encoded_branch(self):
        values = dict(MODEL="Custom", SOURCE="Single safetensors file",
                      PIPELINE_CLASS="StableDiffusionPipeline",
                      CUSTOM_MODEL="https://hf.co/org/repo/resolve/feature%2Fv2/nested/model.safetensors")
        settings = self.settings(**values)
        self.assertEqual(self.ns["single_file_location"](settings["source"])["revision"], "feature/v2")
        with self.assertRaisesRegex(ValueError, "REVISION"):
            self.settings(**values, REVISION="main")
        self.settings(**values, REVISION="feature/v2")

    def test_custom_backend_cache_accounts_for_loader_settings(self):
        self.ns["CUSTOM_LOADERS"]["custom"] = lambda settings, dtype: Pipeline()
        first = self.ns["get_pipeline"](self.settings(BACKEND="custom"))[0]
        second = self.ns["get_pipeline"](self.settings(BACKEND="custom", EXTRA_KWARGS_JSON='{"max_sequence_length":128}'))[0]
        self.assertIsNot(first, second)

    def test_single_file_url_rejects_unsupported_or_private_url(self):
        for source in ("https://example.com/file.safetensors",
                       "https://secret@huggingface.co/org/repo/resolve/main/file.safetensors",
                       "https://huggingface.co/org/repo/resolve/main/file.safetensors?token=secret"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.settings(MODEL="Custom", SOURCE="Single safetensors file",
                              PIPELINE_CLASS="StableDiffusionPipeline", CUSTOM_MODEL=source)

    def test_single_file_hf_url_downloads_the_correct_revision(self):
        for kind in ("resolve", "blob"):
            settings = self.settings(MODEL="Custom", SOURCE="Single safetensors file",
                PIPELINE_CLASS="StableDiffusionPipeline",
                CUSTOM_MODEL=f"https://huggingface.co/org/repo/{kind}/v2/nested/model.safetensors")
            calls = []
            def download(**kwargs):
                calls.append(kwargs)
                return "/cache/model.safetensors"
            with patch.dict("sys.modules", {"huggingface_hub":SimpleNamespace(hf_hub_download=download)}):
                self.ns["load_diffusers"](settings, "fp32")
            self.assertEqual(calls, [dict(repo_id="org/repo", filename="nested/model.safetensors",
                                          revision="v2", token="private-test-token")])
            self.assertEqual(Pipeline.loads[-1][1], "/cache/model.safetensors")

    def test_all_code_cells_compile_and_are_clean(self):
        self.assertEqual(NB["nbformat"], 4)
        self.assertEqual(len(CELLS), len(NB["cells"]))
        for cell in NB["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_all_presets_have_valid_defaults(self):
        for name, preset in self.ns["MODEL_REGISTRY"].items():
            with self.subTest(model=name):
                settings = self.settings(MODEL=name)
                self.assertEqual(settings["source"], preset["repo"])
                self.assertEqual(settings["width"] % preset["multiple"], 0)
                self.assertEqual(settings["pipeline"], preset["pipeline"])
                self.assertEqual(settings["steps"], preset["steps"])

    def test_default_settings_exclude_runtime_and_secrets(self):
        defaults = self.ns["DEFAULT_SETTINGS"]
        self.assertIn(defaults["MODEL"], self.ns["MODEL_REGISTRY"])
        self.assertEqual(set(defaults), set(self.ns["SETTING_FIELDS"]))
        self.assertNotIn("HF_TOKEN", defaults)
        self.assertNotIn("PIPE", defaults)
        self.assertTrue(all(isinstance(value, (str, int, float, bool)) for value in defaults.values()))

    def test_custom_repo_and_local_directory(self):
        for source in ("owner/new-model", "/content/local-diffusers"):
            settings = self.settings(MODEL="Custom", CUSTOM_MODEL=source)
            self.assertEqual(settings["source"], source)
            self.assertEqual(settings["pipeline"], "Auto")
        with self.assertRaises(ValueError):
            self.settings(MODEL="Custom")
        with self.assertRaises(ValueError):
            self.settings(MODEL="Custom", CUSTOM_MODEL="https://huggingface.co/org/model")

    def test_single_file_validation(self):
        values = dict(MODEL="Custom", CUSTOM_MODEL="/content/model.safetensors", SOURCE="Single safetensors file")
        with self.assertRaisesRegex(ValueError, "PIPELINE_CLASS"):
            self.settings(**values)
        values["PIPELINE_CLASS"] = "StableDiffusionXLPipeline"
        self.assertEqual(self.settings(**values)["source"], values["CUSTOM_MODEL"])
        for filename in ("/content/unsafe.ckpt", "http://example.com/model.safetensors"):
            with self.assertRaises(ValueError):
                self.settings(**{**values, "CUSTOM_MODEL":filename})

    def test_bad_numeric_settings_fail_before_loading(self):
        for update in (dict(WIDTH=-8), dict(WIDTH=513), dict(WIDTH=64), dict(HEIGHT=8192),
                       dict(STEPS=-1), dict(STEPS=201), dict(STEPS=1.5), dict(NUM_IMAGES=0),
                       dict(NUM_IMAGES=9), dict(SEED=-2), dict(SEED=2**32-1, NUM_IMAGES=2),
                       dict(GUIDANCE=float("nan")), dict(GUIDANCE=-2), dict(LORA_SCALE=float("inf")),
                       dict(PROMPT="  "), dict(PRECISION="INT8"), dict(MEMORY_MODE="Magic")):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.settings(**update)
        self.assertFalse(Pipeline.loads)

    def test_seed_and_zero_guidance(self):
        settings = self.settings(SEED=-1, NUM_IMAGES=8, GUIDANCE=0, STEPS=1)
        self.assertTrue(0 <= settings["seed"] <= 2**32-8)
        self.assertEqual(settings["guidance"], 0)
        self.assertEqual(settings["steps"], 1)

    def test_extras_must_be_finite_object_without_reserved_keys(self):
        for extras in ('[]', 'null', '{', '{"generator": 5}', '{"prompt":"override"}',
                       '{"test":NaN}', '{"nested":[1e999]}'):
            with self.subTest(extras=extras), self.assertRaises(ValueError):
                self.settings(EXTRA_KWARGS_JSON=extras)
        self.assertEqual(self.settings(MODEL="Qwen Image")["guidance"], 4)
        self.assertEqual(self.settings(MODEL="Qwen Image")["guidance_parameter"], "true_cfg_scale")
        settings = self.settings(MODEL="FLUX.1 schnell", EXTRA_KWARGS_JSON='{"max_sequence_length": 128}')
        self.assertEqual(settings["extra"], {"max_sequence_length":128})

    def test_routing_supported_args(self):
        settings = self.settings(NEGATIVE_PROMPT="blur")
        generator = object()
        kwargs, warnings = self.ns["build_call_kwargs"](Pipeline(), settings, generator)
        self.assertIs(kwargs["generator"], generator)
        self.assertEqual(kwargs["negative_prompt"], "blur")
        self.assertEqual(kwargs["num_images_per_prompt"], 1)
        self.assertEqual(kwargs["output_type"], "pil")
        self.assertFalse(warnings)

    def test_unsupported_negative_is_explicit_and_not_sent(self):
        class Minimal:
            def __call__(self, prompt, width, height, num_inference_steps, generator):
                pass
        kwargs, warnings = self.ns["build_call_kwargs"](Minimal(), self.settings(NEGATIVE_PROMPT="blur"), None)
        self.assertNotIn("negative_prompt", kwargs)
        self.assertNotIn("guidance_scale", kwargs)
        self.assertEqual(len(warnings), 2)
        with self.assertRaisesRegex(ValueError, "extra kwarg"):
            self.ns["build_call_kwargs"](Minimal(), self.settings(EXTRA_KWARGS_JSON='{"nonsense":1}'), None)

    def test_wrong_pipeline_contract_fails(self):
        class Wrong:
            def __call__(self, image, **kwargs):
                pass
        with self.assertRaisesRegex(ValueError, "contract"):
            self.ns["build_call_kwargs"](Wrong(), self.settings(), None)

    def test_cpu_and_cuda_precision_routing(self):
        self.assertEqual(self.ns["select_runtime"](self.settings()), ("CPU", "fp32"))
        for update in (dict(MEMORY_MODE="GPU"), dict(PRECISION="FP16")):
            with self.assertRaises(ValueError):
                self.ns["select_runtime"](self.settings(**update))
        self.cuda.is_available = lambda: True
        self.assertEqual(self.ns["select_runtime"](self.settings()), ("Model CPU offload", "fp16"))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.ns["select_runtime"](self.settings(MODEL="FLUX.1 schnell")), ("Model CPU offload", "fp32"))
        with self.assertRaisesRegex(ValueError, "BF16"):
            self.ns["select_runtime"](self.settings(PRECISION="BF16"))
        self.cuda.is_bf16_supported = lambda: True
        self.assertEqual(self.ns["select_runtime"](self.settings(PRECISION="BF16"))[1], "bf16")

    def test_repo_load_is_safe_and_revision_pinned(self):
        self.ns["get_pipeline"](self.settings(REVISION="a"*40))
        kind, source, kwargs = Pipeline.loads[-1]
        self.assertEqual(kind, "repo")
        self.assertTrue(kwargs["use_safetensors"])
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertEqual(kwargs["revision"], "a"*40)
        self.assertEqual(kwargs["token"], "private-test-token")

    def test_single_file_config_forwarded(self):
        settings = self.settings(MODEL="Custom", CUSTOM_MODEL="/content/model.safetensors",
            SOURCE="Single safetensors file", PIPELINE_CLASS="StableDiffusionXLPipeline",
            SINGLE_FILE_CONFIG="org/config")
        self.ns["get_pipeline"](settings)
        self.assertEqual(Pipeline.loads[-1][0], "single")
        self.assertEqual(Pipeline.loads[-1][2]["config"], "org/config")

    def test_missing_pipeline_or_single_file_api(self):
        with self.assertRaisesRegex(ValueError, "Diffusers pipeline"):
            self.ns["get_pipeline"](self.settings(PIPELINE_CLASS="MissingPipeline"))
        with self.assertRaises(ValueError):
            self.ns["get_pipeline"](self.settings(BACKEND="not-registered"))
        self.assertIsNone(self.ns["PIPE"])

    def test_model_cache_and_lora_configuration(self):
        settings = self.settings(LORA_SOURCE="org/lora", LORA_WEIGHT_NAME="adapter.safetensors", LORA_SCALE=0.6)
        pipe, _, _ = self.ns["get_pipeline"](settings)
        self.assertIn(("adapters", ["user_lora"], [0.6]), pipe.events)
        self.assertTrue(next(e for e in pipe.events if isinstance(e, tuple) and e[0] == "lora")[2]["use_safetensors"])
        self.assertIs(self.ns["get_pipeline"]({**settings, "prompt":"different"})[0], pipe)
        self.assertEqual(len(Pipeline.loads), 1)
        changed = self.ns["get_pipeline"]({**settings, "lora_scale":0.3})[0]
        self.assertIsNot(changed, pipe)
        self.assertEqual(len(Pipeline.loads), 2)
        self.ns["unload_model"]()
        self.assertIsNone(self.ns["PIPE"])
        self.assertIsNone(self.ns["PIPE_KEY"])

    def test_offload_does_not_move_whole_model_to_gpu(self):
        self.cuda.is_available = lambda: True
        for mode, event in (("Model CPU offload", "offload"), ("Sequential CPU offload", "sequential")):
            pipe, _, _ = self.ns["get_pipeline"](self.settings(MEMORY_MODE=mode))
            self.assertIn(event, pipe.events)
            self.assertNotIn(("to", "cuda"), pipe.events)

    def test_loader_failure_clears_model(self):
        def fail(settings, dtype):
            raise RuntimeError("simulated loader failure")
        self.ns["CUSTOM_LOADERS"]["fail"] = fail
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            self.ns["get_pipeline"](self.settings(BACKEND="fail"))
        self.assertIsNone(self.ns["PIPE"])
        self.assertIsNone(self.ns["PIPE_KEY"])

    def test_default_generate_cell_does_not_download_weights(self):
        execute("generate", self.ns)
        self.assertIsNone(self.ns["LAST_ZIP"])
        self.assertFalse(Pipeline.loads)


@unittest.skipIf(Image is None, "Install Pillow to test actual image export")
class ExportTests(NotebookCase):
    def setup_export(self, tmpdir):
        self.ns.update(WORK_DIR=Path(tmpdir), Image=Image, display=lambda image: None,
                       platform=platform, version=lambda name: "test-version")
        execute("generation-engine", self.ns)

    def test_failed_zip_export_does_not_leave_a_partial_archive(self):
        import zipfile
        for error in (OSError("disk full"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmpdir:
                self.setup_export(tmpdir)
                with patch.object(zipfile.ZipFile, "write", side_effect=error):
                    with redirect_stdout(io.StringIO()), self.assertRaises(type(error)):
                        self.ns["generate_images"](self.settings(WIDTH=128, HEIGHT=128))
                self.assertFalse(list(Path(tmpdir).glob("*.zip")))
                self.assertFalse(list(Path(tmpdir).glob("*.part")))
                self.assertIsNone(self.ns["LAST_ZIP"])
                self.assertIsNone(self.ns["PIPE"])
                metadata = json.loads(next(Path(tmpdir).glob("*/metadata.json")).read_text())
                self.assertEqual(metadata["status"], "failed_partial")

    def test_real_images_archive_metadata_and_unload(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmpdir:
            self.setup_export(tmpdir)
            settings = self.settings(NUM_IMAGES=2, WIDTH=128, HEIGHT=128)
            with redirect_stdout(io.StringIO()):
                archive = self.ns["generate_images"](settings)
            self.assertEqual(archive, self.ns["LAST_ZIP"])
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(len(bundle.namelist()), 6)
                raw = bundle.read("metadata.json").decode()
                self.assertNotIn("private-test-token", raw)
                self.assertNotIn("HF_TOKEN", raw)
                metadata = json.loads(raw)
                self.assertEqual(metadata["status"], "complete")
                self.assertEqual([r["seed"] for r in metadata["images"]], [42, 43])
                for record in metadata["images"]:
                    with Image.open(io.BytesIO(bundle.read(record["file"]))) as image:
                        self.assertEqual(image.size, (128, 128))
                        self.assertEqual(image.mode, "RGBA")
                    with Image.open(io.BytesIO(bundle.read(record["jpeg"]))) as jpeg:
                        self.assertEqual(jpeg.mode, "RGB")
                        # Transparent blue was flattened on white, not black.
                        self.assertGreater(jpeg.getpixel((0, 0))[0], 100)
            self.assertIsNone(self.ns["PIPE"])

    def test_keep_model_and_disable_jpeg(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmpdir:
            self.setup_export(tmpdir)
            with redirect_stdout(io.StringIO()):
                archive = self.ns["generate_images"](self.settings(EXPORT_JPEG=False, KEEP_MODEL_IN_MEMORY=True, WIDTH=128, HEIGHT=128))
            with zipfile.ZipFile(archive) as bundle:
                self.assertFalse(any(name.endswith(".jpg") for name in bundle.namelist()))
            self.assertIsNotNone(self.ns["PIPE"])

    def test_partial_failure_no_stale_zip_and_cleans_up(self):
        class FailingPipeline(Pipeline):
            # Preserve the explicit callable contract for routing.
            def __call__(self, prompt, width, height, num_inference_steps, generator):
                if self.calls:
                    raise RuntimeError("simulated OOM")
                return super().__call__(prompt, width, height, num_inference_steps, generator)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.setup_export(tmpdir)
            self.ns["LAST_ZIP"] = Path("old.zip")
            self.ns["CUSTOM_LOADERS"]["fail-second"] = lambda settings, dtype: FailingPipeline()
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "OOM"):
                self.ns["generate_images"](self.settings(BACKEND="fail-second", NUM_IMAGES=2, WIDTH=128, HEIGHT=128, KEEP_MODEL_IN_MEMORY=True))
            self.assertIsNone(self.ns["LAST_ZIP"])
            self.assertIsNone(self.ns["PIPE"])
            self.assertFalse(list(Path(tmpdir).glob("*.zip")))
            metadata = json.loads(next(Path(tmpdir).glob("*/metadata.json")).read_text())
            self.assertEqual(metadata["status"], "failed_partial")
            self.assertEqual(len(metadata["images"]), 1)

    def test_safety_flagged_image_is_not_exported(self):
        class FlaggedPipeline(Pipeline):
            def __call__(self, prompt, width, height, num_inference_steps, generator):
                result = super().__call__(prompt, width, height, num_inference_steps, generator)
                result.nsfw_content_detected = [True]
                return result
        with tempfile.TemporaryDirectory() as tmpdir:
            self.setup_export(tmpdir)
            self.ns["CUSTOM_LOADERS"]["flagged"] = lambda settings, dtype: FlaggedPipeline()
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "safety"):
                self.ns["generate_images"](self.settings(BACKEND="flagged", WIDTH=128, HEIGHT=128))
            self.assertFalse(list(Path(tmpdir).rglob("*.png")))
            self.assertIsNone(self.ns["LAST_ZIP"])


if __name__ == "__main__":
    unittest.main()
