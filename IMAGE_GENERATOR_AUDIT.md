# Image Generator — Security & Bug Review

วันที่ตรวจ: 2026-09-21

## ขอบเขต

ตรวจ `Image_generator.ipynb` และชุดทดสอบ โดยมองเป็น **Notebook ส่วนตัวบน Colab/Jupyter** ไม่ใช่บริการรับ input จากผู้ใช้หลายคนหรือ sandbox สำหรับโมเดล/โค้ดที่ไม่เชื่อถือ

- ตรวจ source และ error paths ของ validation, loader, generation/export และ widget callbacks
- ตรวจ implementation ของ `ipywidgets.Output.__exit__` ใน ipywidgets 8.1.9 และส่วน single-file loader ของ Diffusers 0.36.0
- ทดสอบด้วย widgets จริง, mocked inference และ PNG/JPEG/ZIP จริงจาก Pillow
- ไม่ดาวน์โหลด weights, ไม่ทดสอบ CUDA inference, ไม่ทดสอบ browser ของ Colab จริง และไม่ได้ตรวจ CVE ของ dependency ทั้งชุด

ระดับความรุนแรงด้านล่างแยก **บั๊กการทำงาน**, **ความเสี่ยงข้อมูลส่วนตัว** และ **การเพิ่มการตรวจสอบ input** ไม่ใช่คะแนน CVSS หรือข้อยืนยันว่ามี remote exploit

## สิ่งที่พบและแก้แล้ว

### IG-01 — แผงอาจรายงานสำเร็จหลัง inference ล้มเหลว
**ผลกระทบสูงต่อความถูกต้องของ UI**

`Output.__exit__` คืน `True` เมื่อมี IPython shell/kernel จึงสามารถกลืน exception ใน `with self.output:` ได้ โค้ดเดิมดัก error นอก context แล้วไปตั้งสถานะ success หลังออกจาก context แม้ engine จะล้มเหลว โดย headless tests เดิมไม่ผ่าน branch นี้

**แก้:** ดัก exception/interrupt ภายใน Output context ก่อนที่ widget จะจัดการ ตรวจว่า engine คืนไฟล์ที่มีอยู่จริงก่อนรายงาน success และคืนสถานะปุ่มใน `finally`

**Regression:** `test_ipython_output_suppression_cannot_turn_failure_into_success`, `test_kernel_output_suppression_preserves_interrupt_and_validation_state`, `test_engine_returning_missing_archive_is_not_success`

### IG-02 — Error และ metadata อาจมี token ที่หลุดมากับข้อความ
**ความเสี่ยงข้อมูลส่วนตัวแบบมีเงื่อนไข**

ไม่มีการ serialize `HF_TOKEN` โดยตรง แต่ UI เดิมพิมพ์ `str(exception)` ทั้งก้อน และ metadata เก็บข้อความจากผู้ใช้ตามต้นฉบับ หาก exception มี token หรือผู้ใช้เผลอวาง token ลงใน prompt จะติดไปกับ output/ZIP ได้ ช่อง LoRA/config ยังยอมรับ URL ที่ฝัง credentials แม้ไม่ใช่รูปแบบ input ที่รองรับ

**แก้:** ปิดบัง token ที่ runtime รู้จัก (รวมรูป URL-encoded), รูปแบบ HF token ที่รู้จัก, URL credentials/query ที่เป็นความลับ และ secret-key fields ใน metadata แบบ recursive โดยไม่แก้ settings ที่ใช้ inference ปฏิเสธ URL ในช่องที่ควรเป็น repo ID/local path

**ขอบเขต:** ไม่ใช่ระบบค้นหาความลับทุกชนิด ไม่ครอบคลุมภาพที่โมเดลสร้าง, raw widget values, log ของ dependency/custom code หรือการเรียก engine/legacy cells โดยตรง ต้องไม่ใส่ secrets ลงใน forms และต้องตรวจ notebook outputs ก่อนแชร์

**Regression:** `test_ui_exception_output_does_not_echo_auth_secret`, `test_export_redacts_known_token_in_user_supplied_text`, `test_redaction_handles_nested_fields_and_does_not_mutate_input`

### IG-03 — ชื่อไฟล์หลัง URL decode และ LoRA filename ตรวจไม่ครบ
**Input hardening; ยังไม่ได้ยืนยัน arbitrary file-write exploit ใน dependency**

Filename เช่น `%2Ftmp/...` หรือ `..%2F...` ผ่าน validation เดิมแล้วถูกส่งต่อให้ Hugging Face downloader หลัง `unquote` ส่วน LoRA filename แบบ absolute/parent traversal ตรวจช้าเกินไปหรือไม่ถูกปฏิเสธ

**แก้:** ตรวจ decoded filename/revision ปฏิเสธ absolute paths, `.`/`..`, backslash, control characters และตรวจ LoRA filename ก่อนโหลด weights ยังคงรองรับ nested file และ branch ที่มี `/` เช่น `feature%2Fv2`

**ขอบเขต:** local model paths ที่ผู้ใช้เลือกเองยังรองรับตามปกติ ไม่ใช่ filesystem sandbox

**Regression:** `test_encoded_hf_filename_must_not_escape_repository`, `test_valid_nested_checkpoint_and_encoded_branch_still_work`, `test_lora_filename_validation_happens_before_weight_loading`

### IG-04 — ZIP รวมไฟล์ที่ไม่ใช่ผลลัพธ์และตาม symlink
**ความเสี่ยงข้อมูลส่วนตัว หากมีไฟล์อื่นปรากฏใน run directory**

การใช้ `run_dir.iterdir()` ทำให้ ZIP รวมทุกไฟล์ในโฟลเดอร์ โดย `ZipFile.write` ตาม symlink ด้วย จึงอาจมีไฟล์จาก callback/tool อื่นปะปนโดยผู้ใช้ไม่ตั้งใจ

**แก้:** ใช้รายการไฟล์ที่สร้างจริง (PNG/JPEG, metadata, README) เท่านั้น และตรวจว่าไฟล์เป็น regular file อยู่ใน run directory ไม่ใช่ symlink สร้าง run directory ด้วย permission `0700` บน POSIX

**ขอบเขต:** ไม่ได้ป้องกัน malicious code ใน kernel เดียวกันหรือ filesystem race จาก process ที่มีสิทธิ์เดียวกัน โค้ดดังกล่าวเข้าถึงไฟล์/secret ได้อยู่แล้ว

**Regression:** `test_zip_contains_only_generated_manifest_not_unrelated_files`, `test_symlink_replacing_generated_file_is_not_exported`

### IG-05 — Metadata เขียนไม่ครบแล้วกลบสาเหตุ error เดิม
**บั๊กความน่าเชื่อถือ**

เมื่อ disk write ล้มเหลวระหว่างบันทึก metadata โค้ดเดิมอ่าน JSON ที่อาจถูกตัดขาดซ้ำใน error handler แล้วเกิด `JSONDecodeError` กลบสาเหตุ เช่น disk full

**แก้:** เขียน metadata ลง temporary file แล้ว replace เมื่อสำเร็จ ใช้สถานะใน memory ใน failure handler และบันทึกสถานะ failure แบบ best-effort โดยไม่กลบ exception เดิม เพิ่ม cleanup เมื่อ loader ถูก interrupt โดยตรง

**Regression:** `test_metadata_write_failure_does_not_mask_original_disk_error`, `test_failed_metadata_replace_keeps_previous_complete_json`, `test_interrupted_loader_clears_allocated_pipeline`

### IG-06 — Extra kwargs JSON ไม่มีขอบเขตขนาด/ความลึก
**การเพิ่มความทนทาน ไม่ใช่ข้ออ้างว่าป้องกัน resource exhaustion ทั้งหมด**

**แก้:** จำกัด JSON เป็น object ขนาดไม่เกิน 32 KiB, ความลึกไม่เกิน 12 ชั้น และแปลง parser recursion errors เป็น validation error ที่อ่านได้

ค่าที่ถูกต้องตาม JSON เช่น resolution หรือพารามิเตอร์เฉพาะของโมเดลยังอาจใช้ RAM/VRAM สูง ต้องใช้ model-specific constraints และ runtime ที่เหมาะสมด้วย

**Regression:** `test_large_or_overdeep_extra_json_fails_cleanly`

## ตรวจซ้ำรอบที่สอง — จุดที่เพิ่มและแก้แล้ว

### IG-07 — การตรวจนามสกุลไม่ตรงกับตัวอ่าน checkpoint ของ dependency
**การปิดช่องโหว่การตรวจรูปแบบไฟล์; ไม่ได้ยืนยัน arbitrary code execution**

โค้ดเดิมใช้ `.lower().endswith(".safetensors")` จึงยอมรับ `.SAFETENSORS` แต่บางเส้นทางของ Diffusers เลือกตัวอ่านไฟล์แบบ case-sensitive ทำให้ไม่สามารถรับรองว่าการยอมรับชื่อไฟล์นั้นจะนำไปสู่ safetensors reader ได้ นอกจากนี้ `urlparse` แยก query/fragment และลบ control characters บางตัว แต่ local filesystem ใช้ชื่อไฟล์ตามตัวอักษร จึงอาจตรวจ suffix คนละตัวกับไฟล์ที่จะโหลด

**แก้:** รับ suffix `.safetensors` ตัวพิมพ์เล็กเท่านั้น ปฏิเสธ query/fragment ใน local checkpoint path และ control characters ก่อนส่งให้ตัวโหลด ตรวจ local LoRA file อีกครั้งก่อนโหลด base model ไม่แก้ชื่อหรือเปลี่ยนไฟล์ให้อัตโนมัติ

**Regression:** `test_uppercase_checkpoint_suffix_is_rejected_before_loader`, `test_local_checkpoint_suffix_cannot_hide_query_fragment_or_newline`, `test_local_lora_with_unsafe_suffix_is_rejected_before_base_load`

### IG-08 — LoRA autodiscovery ไม่ใช้ explicit token / อาจเลือกไฟล์ผิดโดยเดา
**บั๊กการเข้าถึงและความถูกต้องของโมเดล**

ใน Diffusers 0.36 ฟังก์ชัน `_best_guess_weight_name` เรียก `model_info` โดยไม่ส่ง explicit token การมี token ใน Colab Secrets ซึ่งไม่ได้บันทึกเป็น cached login จึงอาจไม่ช่วยในขั้นตอนนี้ แม้ notebook ส่ง token ให้ `load_lora_weights` แล้ว และเมื่อมีหลายไฟล์ dependency อาจเลือกไฟล์แรกให้เอง

**แก้:** notebook ใช้ `list_repo_files(..., token=HF_TOKEN)` เพื่อหาไฟล์ก่อนโหลด base model ส่ง `weight_name` ชัดเจนทุกครั้ง เลือกอัตโนมัติเฉพาะกรณีมี `.safetensors` หนึ่งไฟล์เท่านั้น หากมีหลายไฟล์/ไม่มีไฟล์ ให้ผู้ใช้ระบุ ไม่เดา รองรับ local file/directory โดยไม่ต้องเรียก Hub และไม่ค้น repo ซ้ำเมื่อใช้ model cache เดิม

บันทึก filename ที่เลือกจริงใน `metadata.json → runtime → loader` การ authenticate กับ repo จริงยังต้อง smoke-test บน Colab; tests ใช้ fake Hub API

**Regression:** `test_private_lora_autodiscovery_uses_explicit_token`, `test_multiple_or_missing_lora_files_fail_before_base_model_load`, `test_cached_lora_does_not_repeat_repo_listing`, local LoRA tests

### IG-09 — VAE tiling ถูกข้ามใน pipeline ที่ไม่มี wrapper แบบเก่า
**บั๊กการตั้งค่าหน่วยความจำ**

เช่น PixArt Sigma / Z-Image ใน Diffusers 0.36 มี VAE แต่ไม่มี `pipe.enable_vae_tiling()` การตรวจแค่ method บน pipeline จึงข้ามการเปิด tiling ทั้งที่ VAE รองรับ

**แก้:** เรียก `pipe.vae.enable_tiling()` ก่อน หากไม่มีจึงใช้ wrapper แบบเก่า บันทึกสถานะจริงเป็น `enabled`, `off` หรือ `unavailable` ใน loader metadata ข้อนี้ไม่ได้แปลว่าโมเดลจะพอดีกับ GPU ทุกรุ่น

**Regression:** `test_tiling_uses_vae_api_when_pipeline_wrapper_is_absent`, `test_modern_tiling_preferred_and_actual_status_recorded`, `test_legacy_tiling_still_supported_and_disabled_tiling_not_called`

### IG-10 — Cleanup ของ loader กลบ error ต้นเหตุได้
**บั๊ก error handling**

แม้ generation engine จะป้องกัน cleanup failure แล้ว แต่ `get_pipeline` ยังเรียก `unload_model` โดยตรงใน exception handler หาก GPU cache cleanup ล้มเหลวซ้ำ จะกลบ error ที่ทำให้โหลดโมเดลไม่สำเร็จ

**แก้:** cleanup แบบ best-effort และ re-raise error ต้นฉบับ โดยล้าง pipeline/cache references ตามเดิม

**Regression:** `test_loader_cleanup_error_does_not_mask_original_failure`

### IG-11 — ZIP อยู่นอก private run directory และใช้ permission ตาม umask
**Privacy hardening สำหรับ POSIX runtime ที่มีหลาย local accounts**

run directory เป็น `0700` แต่ ZIP ถูกสร้างเป็นไฟล์ข้างนอก directory นั้น จึงอาจเป็น `0644` ภายใต้ umask ปกติ และเปิดให้ local accounts อื่นอ่าน prompts/metadata ได้หาก parent directories อนุญาต ไม่ใช่การเผยแพร่ ZIP สู่อินเทอร์เน็ตโดยอัตโนมัติ

**แก้:** สร้าง temporary ZIP ด้วย `os.open(..., O_CREAT | O_EXCL, 0o600)` ก่อนเขียนข้อมูล แล้ว rename เป็น ZIP สุดท้าย ซึ่งรักษา permission นี้ไว้

**Regression:** `test_zip_is_private_even_with_permissive_umask`

## ผลตรวจสอบ

```bash
python -m pip install pillow ipywidgets nbformat
python -W error::DeprecationWarning -m unittest discover -s tests -p 'test_image_generator*.py' -v
```

- ผ่าน **75 tests** เมื่อมี optional test dependencies ครบ ไม่มีการโหลดโมเดลจริง
- Notebook ผ่าน `nbformat.validate` และ code cells ผ่าน syntax checks
- `git diff --check` ผ่าน
- Regression สำหรับปัญหาหลักถูกเพิ่มและรันให้พบ failure ก่อนแก้ ไม่ใช่เฉพาะ happy-path tests
- หากไม่ติดตั้ง Pillow/ipywidgets บาง tests จะถูก skip; ผล run ที่มี skip ไม่เท่ากับการตรวจครบ

## ความเสี่ยงที่ยังเหลือ / งานที่ควรทำต่อ

1. **Colab/GPU smoke test:** ต้องทดสอบ interaction ใน browser จริง, download, interruption, gated model authentication, OOM และ inference ของโมเดลเป้าหมาย
2. **Dependency / model supply chain:** `use_safetensors=True` และการไม่เปิด remote code ไม่ใช่ sandbox สำหรับ weights/config ที่ไม่เชื่อถือ ตรวจแหล่งที่มา/license และใช้ runtime แยกที่ไม่มี secrets สำคัญเมื่อทดลองโมเดลใหม่
3. **Single-file auxiliary components:** Diffusers 0.36 สามารถเรียก `from_pretrained` ของ component เสริมโดยไม่ส่ง `use_safetensors=True` ในบางเส้นทาง Notebook จึงไม่รับรองว่า **ทุก auxiliary component** จะเป็น safetensors เพียงเพราะ checkpoint หลักเป็น `.safetensors` ใช้ config/cache ที่เชื่อถือได้และ PyTorch/Transformers ที่อัปเดต ข้อนี้ไม่ได้ถูกแก้โดย monkey-patch dependency
4. **Custom loaders:** เป็น Python code ที่ผู้ใช้เชื่อถือและลงทะเบียนเอง มีสิทธิ์ของ kernel เต็มรูปแบบ ห้ามใช้รับ backend code จากบุคคลอื่นโดยไม่ตรวจ
5. **Version reproducibility:** dependency ranges และ repo revision ที่ไม่ pin เปลี่ยนได้ ควร pin เวอร์ชันและ commit หลังผ่าน smoke test ของ workflow จริง
6. **Privacy:** prompts และ metadata เป็นข้อมูลส่วนตัวแม้ไม่มี token อย่าแชร์ ZIP หรือ notebook outputs โดยไม่ตรวจ
