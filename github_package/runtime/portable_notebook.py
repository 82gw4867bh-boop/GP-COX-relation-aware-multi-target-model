"""Bind source notebooks to one of the four endpoint inputs and output paths."""
import ast
import json
from pathlib import Path

def bind(notebook, train_path, external_path, output_dir, profile):
    project = json.loads((Path(__file__).resolve().parent.parent / 'config/four_endpoint_project.json').read_text())
    matches = [ep for ep in project['endpoints'] if train_path.stem in (ep['id'] + '_eicu', ep['input_stem'] + '_eicu')]
    if len(matches) != 1:
        raise ValueError(f'Not a four-endpoint input: {train_path}')
    ep = matches[0]
    cfg = dict(endpoint_index=int(ep['id'].split('_')[1]), target_id=ep['id'], target_name=ep['name'],
               train_file=str(train_path), external_file=str(external_path))
    profile_path = output_dir / 'profile.json'
    profile_path.write_text(json.dumps({'profile_name': 'profile_007_high_mutation', 'profile_id': 7, 'params': profile}, indent=2), encoding='utf8')
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        text = ''.join(cell['source'])
        edits = []
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if name == 'ENDPOINT_RUN_CONFIGS':
                edits.append((node.lineno-1, node.end_lineno, 'ENDPOINT_RUN_CONFIGS = ' + repr([cfg]) + '\n'))
            elif name == 'PROFILE_PARAMS_PATH':
                edits.append((node.lineno-1, node.end_lineno, ' '*node.col_offset + 'PROFILE_PARAMS_PATH = Path(' + repr(str(profile_path)) + ')\n'))
        lines = text.splitlines(keepends=True)
        for start, end, replacement in sorted(edits, reverse=True):
            lines[start:end] = [replacement]
        cell['source'] = ''.join(lines).splitlines(keepends=True)
        cell['outputs'] = []
        cell['execution_count'] = None
    return notebook
