"""Execute source notebook cells using the invoking Python environment."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')

def main():
    source, destination = map(Path, sys.argv[1:3])
    notebook = json.loads(source.read_text(encoding='utf8'))
    namespace = {'__name__': '__main__', '__file__': str(source)}
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        print(f'Executing cell {index + 1}', flush=True)
        exec(compile(''.join(cell['source']), f'{source}:cell{index}', 'exec'), namespace)
        cell['execution_count'] = index + 1
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf8')

if __name__ == '__main__':
    main()
