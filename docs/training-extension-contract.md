# Training extension manifest, schema 1

Install an optional extension under `extensions/training/` beside the application executable. The inference package never imports or executes a manifest while detecting it. Detection validates schema/API, extension version, backend, task and confined file paths.

Example `manifest.json` (illustrative; no AMD training implementation is supplied):

```json
{
  "schema_version": 1,
  "api_version": "1",
  "version": "0.1.0",
  "backends": ["cuda"],
  "tasks": ["sensevoice_singing", "gsv_tts", "campplus", "game"],
  "python": "runtime/python.exe",
  "entrypoint": "train.py"
}
```

Statuses: `extension_missing`, `version_mismatch`, `device_unsupported`, `task_unsupported`, `dependency_missing`, `invalid_manifest`, `ready`. Version compatibility currently means schema 1 and API 1; a nonempty extension version is mandatory. All executable paths must resolve inside the extension directory and exist. Merely adding a manifest does not enable AMD training. The frozen inference release exposes detection only and keeps training disabled; source CUDA/CPU adapters retain their existing behavior.
