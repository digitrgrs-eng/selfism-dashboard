"""Ordered model-only LoRA rows and their prompt triggers for MiniMax workflows."""
import json
import math
from pathlib import Path, PureWindowsPath
import re


MAX_LORA_ROWS = 16
MAX_JSON_BYTES = 128 * 1024
MAX_TRIGGER_CHARACTERS = 4096
MAX_NAME_CHARACTERS = 1024


def _relative_name(name, row_number):
    if not isinstance(name, str) or len(name) > MAX_NAME_CHARACTERS:
        raise ValueError(f'LoRA {row_number}: choose a valid installed LoRA name.')
    name = name.strip().replace('\\', '/')
    if not name:
        return ''
    parts = name.split('/')
    if (name.startswith('/') or PureWindowsPath(name).drive or ':' in name
            or any(part in ('', '.', '..') for part in parts)
            or any(ord(character) < 32 or ord(character) == 127 for character in name)):
        raise ValueError(f'LoRA {row_number}: use a relative name from the installed LoRA list.')
    return name


def _reject_json_constant(value):
    raise ValueError(f'LoRA settings contain invalid JSON number {value}.')


def parse_lora_rows(loras_json):
    if not isinstance(loras_json, str) or len(loras_json.encode('utf-8')) > MAX_JSON_BYTES:
        raise ValueError('LoRA settings must be JSON text smaller than 128 KiB.')
    try:
        rows = json.loads(loras_json, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError('LoRA settings are not valid JSON. Choose your LoRAs again in the settings panel.') from error
    if not isinstance(rows, list) or len(rows) > MAX_LORA_ROWS:
        raise ValueError(f'LoRA settings must be a list with no more than {MAX_LORA_ROWS} rows.')
    normalized = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f'LoRA {index}: each row must be an object.')
        enabled = row.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError(f'LoRA {index}: enabled must be true or false.')
        name = _relative_name(row.get('name', ''), index)
        if enabled and not name:
            raise ValueError(f'LoRA {index}: choose an installed LoRA, or disable this row.')
        strength = row.get('strength', 1.0)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not -10 <= strength <= 10:
            raise ValueError(f'LoRA {index}: strength must be a finite number between -10 and 10.')
        triggers = row.get('triggers', '')
        if not isinstance(triggers, str) or len(triggers) > MAX_TRIGGER_CHARACTERS:
            raise ValueError(f'LoRA {index}: trigger words must be text with at most {MAX_TRIGGER_CHARACTERS} characters.')
        normalized.append({'enabled': enabled, 'name': name, 'strength': float(strength), 'triggers': triggers})
    return normalized


def resolve_enabled_loras(rows):
    """Resolve only enabled rows through Comfy's registered LoRA folders."""
    active = [(index, row) for index, row in enumerate(rows, start=1) if row['enabled']]
    if not active:
        return []
    import folder_paths
    installed = {name.replace('\\', '/'): name for name in folder_paths.get_filename_list('loras')}
    roots = [Path(root).resolve() for root in folder_paths.get_folder_paths('loras')]
    resolved = []
    for index, row in active:
        if row['name'] not in installed:
            raise ValueError(f'LoRA {index}: "{row["name"]}" is not installed. Refresh the list or upload the file.')
        canonical_name = installed[row['name']]
        filename = folder_paths.get_full_path('loras', canonical_name)
        if not filename:
            raise ValueError(f'LoRA {index}: the selected file is missing. Refresh the installed LoRA list.')
        path = Path(filename).resolve()
        if not path.is_file() or not any(path.is_relative_to(root) for root in roots):
            raise ValueError(f'LoRA {index}: the file must be inside a registered LoRA folder.')
        resolved.append((row, canonical_name, path))
    return resolved


def append_lora_triggers(prompt, rows):
    if not isinstance(prompt, str):
        raise ValueError('The LoRA stack needs the active generation prompt as text.')
    triggers, seen = [], set()
    for row in rows:
        if not row['enabled']:
            continue
        for part in re.split(r'[,\r\n]+', row['triggers']):
            text = part.strip()
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                triggers.append(text)
    if not triggers:
        return prompt
    # Keep the original prompt byte-for-byte as the prefix; neither Manual nor
    # Qwen output is rewritten. Duplicate trigger phrases appear only once.
    return prompt + '\n\nLoRA trigger words:\n' + ', '.join(triggers)


class H3StudioLoraStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL', {'tooltip': 'MiniMax H3 model before these LoRAs are applied.'}),
            'loras_json': ('STRING', {'multiline': True, 'default': '[]', 'tooltip': 'LoRA rows edited in the settings panel. Enabled rows apply in order; disabled rows skip both weights and triggers.'}),
            'prompt': ('STRING', {'forceInput': True, 'tooltip': 'Active Qwen or Manual prompt. Enabled LoRA trigger words are appended here.'}),
        }}

    RETURN_TYPES = ('MODEL', 'STRING')
    RETURN_NAMES = ('model', 'prompt_with_lora_triggers')
    FUNCTION = 'apply'
    CATEGORY = 'MiniMax H3 Studio/internal'
    DESCRIPTION = 'Apply installed MiniMax-compatible LoRAs in displayed order and append enabled trigger words to the active prompt.'

    @classmethod
    def VALIDATE_INPUTS(cls, loras_json):
        try:
            resolve_enabled_loras(parse_lora_rows(loras_json))
        except (ValueError, OSError) as error:
            return str(error)
        return True

    @classmethod
    def IS_CHANGED(cls, loras_json, **kwargs):
        resolved = resolve_enabled_loras(parse_lora_rows(loras_json))
        # Replacing an installed file should invalidate the node even if its
        # dropdown name, prompt and strength remain unchanged.
        return tuple((name, path.stat().st_size, path.stat().st_mtime_ns) for _, name, path in resolved)

    def apply(self, model, loras_json, prompt):
        rows = parse_lora_rows(loras_json)
        resolved = resolve_enabled_loras(rows)
        full_prompt = append_lora_triggers(prompt, rows)
        current = model
        if resolved:
            import nodes
            # The stock loader uses safe_load=True and its standard model patch
            # handling. Keep one short-lived loader, not sixteen cached copies.
            loader = nodes.LoraLoaderModelOnly()
            for row, filename, _ in resolved:
                if row['strength'] != 0:
                    current = loader.load_lora_model_only(current, filename, row['strength'])[0]
        return (current, full_prompt)


LORA_CONTROL_NODES = {'H3StudioLoraStack': H3StudioLoraStack}
