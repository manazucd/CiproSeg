# How to add a new model

1. Implement your module under `src/ciproseg/models/`.
2. Register it in `registry.py` using `@register_model("to_name")`.
3. Point your YAML `model.name` to `"to_name"`.

