"""Run the selected seed schedules and save newly computed result summaries."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import run_pipeline as pipeline

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',required=True,choices=['g2g','boundary','live_online','end_only'])
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--upstream-dir',type=Path,help='Formal G/resume2G output directory; required for exchanges.')
    parser.add_argument('--seed',type=int,help='Run just one seed instead of the complete selected schedule.')
    args=parser.parse_args()
    manifest=json.loads((pipeline.ROOT/'config/selected_seed_manifest.json').read_text())['selected_nonformal_seeds']
    seeds=[args.seed] if args.seed is not None else manifest['G' if args.method=='g2g' else args.method]
    if args.method!='g2g' and args.upstream_dir is None:
        parser.error('--upstream-dir is required for exchange seed runs')
    rows=[]
    for seed in seeds:
        output=args.output_dir.resolve()/args.method/f'seed_{seed}'
        common=['--data-dir',str(args.data_dir.resolve()),'--output-dir',str(output)]
        def run(extra):
            subprocess.run([sys.executable,str(pipeline.ROOT/'run_pipeline.py'),*extra,*common],check=True)
        if args.method=='g2g':
            for budget in ['G','resume2G']:
                run(['train','--budget',budget,'--seed',str(seed)])
                values=[]
                for ep in pipeline.ENDPOINTS:
                    r=json.loads((output/budget/'endpoints'/ep['id']/'result.json').read_text())
                    values.append(float(r['ABC_external_cindex']))
                rows.append({'method':'G' if budget=='G' else '2G','seed':seed,'external_macro_cindex':statistics.mean(values)})
        else:
            run(['exchange','--strategy',args.method,'--seed',str(seed),'--policy-seed',str(seed),
                 '--upstream-dir',str(args.upstream_dir.resolve())])
            if args.method=='end_only':
                with (output/'end_only/results/external_validation_4endpoint.csv').open(encoding='utf-8-sig') as stream:
                    values=[float(r['multitarget_exchange_gp_external_cindex']) for r in csv.DictReader(stream)]
            else:
                r=json.loads((output/args.method/'selected_profile/selected_profile_external.json').read_text())
                values=[float(x['external_cindex']) for x in r]
            rows.append({'method':args.method,'seed':seed,'external_macro_cindex':statistics.mean(values)})
        args.output_dir.mkdir(parents=True,exist_ok=True)
        with (args.output_dir/f'{args.method}_recomputed.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['method','seed','external_macro_cindex']);writer.writeheader();writer.writerows(rows)

if __name__=='__main__':main()
