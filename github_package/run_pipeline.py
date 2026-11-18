"""Portable four-endpoint training, exchange, final-model and statistics entry point."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime'
sys.path.insert(0, str(RUNTIME))
PROJECT = json.loads((ROOT / 'config/four_endpoint_project.json').read_text())
ENDPOINTS = PROJECT['endpoints']

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf8')

def preflight(data, output):
    import numpy as np
    import pandas as pd
    rows = []
    for ep in ENDPOINTS:
        for cohort in ['eicu', 'mimic']:
            path = data / f"{ep['input_stem']}_{cohort}.xlsx"
            frame = pd.read_excel(path)
            required = {'time', 'event', 'Total.Ca', 'Ionized.Ca'}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f'{path.name}: missing {sorted(missing)}')
            if not np.isfinite(frame['time']).all() or not (frame['time'] > 0).all() or not frame['event'].isin([0, 1]).all():
                raise ValueError(f'{path.name}: invalid survival outcome')
            expected = 313 if cohort == 'eicu' else 340
            if len(frame) != expected:
                raise ValueError(f'{path.name}: expected {expected} rows; received {len(frame)}')
            rows.append({'endpoint':ep['name'], 'cohort':cohort, 'rows':len(frame), 'events':int(frame['event'].sum()),
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'columns':list(frame.columns)})
    write(output / 'input_manifest.json', rows)
    return rows

def aliases(data, output):
    directory = output / 'inputs'
    directory.mkdir(parents=True, exist_ok=True)
    for ep in ENDPOINTS:
        for cohort in ['eicu', 'mimic']:
            src = data / f"{ep['input_stem']}_{cohort}.xlsx"
            dst = directory / f"{ep['id']}_{cohort}.xlsx"
            if not dst.exists() or hashlib.sha256(src.read_bytes()).digest() != hashlib.sha256(dst.read_bytes()).digest():
                shutil.copyfile(src, dst)
    return directory

def environment(output, seed):
    project = json.loads(json.dumps(PROJECT))
    config = {'endpoints':[dict(name=e['name'], id=e['id'], group='resource' if e['name']=='icu_los_gt7d' else 'terminal') for e in ENDPOINTS],
              'phaseA_matrix':project['phaseA_reference_matrix'], 'phaseC_matrix':project['phaseC_reference_matrix'],
              'exclusive_groups':[['death_hospice','institutional_nonhome','home']]}
    return {'MTGP_ENDPOINT_CONFIG_JSON':json.dumps(config), 'MTGP_DATA_DIR':str(output/'inputs'),
            'MTGP_G_CHECKPOINT_ROOT':str(output/'G/endpoints'), 'MTGP_G_KNOWLEDGE_ROOT':str(output/'G/knowledge'),
            'MTGP_G_EXPECTED_GENERATIONS_JSON':json.dumps({e['id']:[12]*5 for e in ENDPOINTS}),
            'MTGP_ENDPOINT_SEEDS_JSON':json.dumps({e['id']:seed for e in ENDPOINTS}),
            'MTGP_SINGLE_PROFILE_JSON':str(output/'G/profile.json'), 'NOTEBOOK_PYTHON':sys.executable,
            'MPLBACKEND':'Agg', 'PYTHONPATH':str(RUNTIME)+os.pathsep+os.environ.get('PYTHONPATH','')}

def knowledge(source, output, phase_a, phase_c):
    import build_knowledge_hcoxfix_trial010_from_single_gp as builder
    builder.SOURCE_ROOT = source
    builder.OUTPUT_ROOT = output
    builder.ENDPOINTS = [(e['id'],e['name'],int(e['id'].split('_')[1])) for e in ENDPOINTS]
    builder.PROFILE_PARAMS = parameters(phase_a, phase_c)
    builder.main()

def parameters(phase_a, phase_c):
    values = json.loads((ROOT/'config/single_target_G_defaults.json').read_text())
    values['N_GEN_X'] = values['FINAL_FULL_TRAIN_N_GEN_X'] = phase_a
    values['PHASEC_N_GEN'] = phase_c
    for key in ['MAIN_EFFECT_GP_PARAMS','GP_ONLY_PARAMS']:
        values[key]['N_GEN_X'] = phase_a
    return values

def command(script, args, env=None):
    merged = dict(os.environ)
    merged.update(env or {})
    subprocess.run([sys.executable, str(script), *map(str,args)], env=merged, check=True)

def train(data, output, seed, budget, endpoint=None, smoke=False):
    import run_multitarget_profile_optimization as legacy
    branch = output / budget
    phase_a, phase_c = (12,20) if budget=='G' else (24,40)
    if budget == 'resume2G':
        phase_a = 12
    params = parameters(phase_a, phase_c)
    if smoke:
        params.update(PHASEC_N_GEN=1, PHASEC_POP_SIZE=8, PHASEC_K_VALUES=[0], PHASEC_HOF_SIZE=3,
                      N_GEN_X=1,FINAL_FULL_TRAIN_N_GEN_X=1)
        for key in ['MAIN_EFFECT_GP_PARAMS','GP_ONLY_PARAMS']:
            params[key].update(N_GEN_X=1,POP_SIZE_X=8)
    request = {'seed':seed,'parameters':params,'data_dir':str(data)}
    request_path = branch / 'run_request.json'
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError('This output directory contains a different seed/profile/input. Use a new output directory.')
    write(request_path,request)
    write(branch/'profile.json', {'profile_name':'hcoxfix_trial010_forceB_softC','params':params})
    template = RUNTIME/'single_target_checkpoint.ipynb'
    if budget=='resume2G':
        template = output/'templates/resume2G.ipynb'
        env = environment(output,seed)
        env.update(HCA_RESUME_SOURCE_NOTEBOOK=str(RUNTIME/'single_target_checkpoint.ipynb'),HCA_RESUME_OUTPUT_NOTEBOOK=str(template))
        command(RUNTIME/'checkpoint_no_exchange_G_continuation/build_resume_notebook.py',[],env)
    legacy.NOTEBOOK_FILE = template
    for ep in ENDPOINTS:
        if endpoint and ep['name'] != endpoint:
            continue
        target = {'target_id':ep['id'],'target_name':ep['name'],
                  'train_file':data/f"{ep['id']}_eicu.xlsx", 'external_file':data/f"{ep['id']}_mimic.xlsx"}
        status = legacy.run_notebook(branch/'endpoints'/ep['id'], target, params, timeout=86400, force=False, seed=seed)
        if status['status'] != 'success':
            raise RuntimeError(f"{budget}/{ep['name']} failed: {status['error_message']}; see {status['log_path']}")
        print(f"Completed {budget}/{ep['name']}",flush=True)
    if not endpoint:
        knowledge(branch/'endpoints',branch/'knowledge',phase_a,phase_c)

def exchange(output, seed, strategy, policy_seed=2026080903, upstream=None):
    upstream = upstream or output
    env = environment(upstream,seed)
    if strategy in ['boundary','live_online']:
        mode = 'boundary' if strategy=='boundary' else 'online'
        template = output/f'templates/{strategy}.ipynb'
        env.update({f'MTGP_{"BOUNDARY" if mode=="boundary" else "ONLINE"}_SOURCE_NOTEBOOK':str(RUNTIME/'exchange_checkpoint.ipynb'),
                    f'MTGP_{"BOUNDARY" if mode=="boundary" else "ONLINE"}_OUTPUT_NOTEBOOK':str(template),
                    'HCA_RESUME_SOURCE_NOTEBOOK':str(RUNTIME/'exchange_checkpoint.ipynb')})
        command(RUNTIME/f'checkpoint_fair_multitarget_experiments/build_{mode}_resume_notebook.py',[],env)
        command(RUNTIME/'checkpoint_fair_multitarget_experiments/run_g_to_2g_exchange.py',
                ['--mode',mode,'--output-root',output/strategy,'--exchange-config',ROOT/f'config/{strategy}_defaults.json','--run-external-after-internal'],env)
    else:
        backend = output/'end_only/runtime'
        backend.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'config/eligible_predictor_manifest.json', backend/'eligible_predictor_manifest.json')
        cfg = json.loads((ROOT/'config/end_only_runtime.json').read_text())
        cfg.update(branch_root=str(backend),knowledge_root=str(upstream/'resume2G/knowledge'),
                   output_root=str(output/'end_only/results'),run_output_root=str(output/'end_only/results'))
        for ep in cfg['endpoints']:
            ep['train_file'] = str(output/f"inputs/{ep['folder']}_eicu.xlsx")
        cfg['smoke']['random_seed'] = policy_seed
        path=backend/'config.json';write(path,cfg)
        command(RUNTIME/'end_only_backend/exchange_policy_tree.py',['--config',path],env)

def locked(data, output):
    os.environ['COXGP_DATA_DIR'] = str(data)
    import four_endpoint_safe_endonly_core_20260814 as core
    models=core.load_all_models()
    write(output/'locked_model_audit.json',core.audit_payload(models))
    for ep,model in models.items():
        write(output/f'locked_models/{ep}.json',{'endpoint':ep,'external_cindex':model.external_cindex,
              'phaseA_coefficients':model.phasea_coefficients,'final_coefficients':model.final_coefficients,
              'expressions':model.gp_expressions})

def statistics_report(output):
    from scipy import stats
    def read(name):
        with (ROOT/'reproducibility'/name).open(encoding='utf-8-sig',newline='') as stream:
            return list(csv.DictReader(stream))
    g=read('selected_10seed_G_2G_results.csv');m=read('selected_10seed_multitarget_results.csv')
    a=[float(r['external_macro_cindex']) for r in g if r['method']=='2G']
    b=[float(r['external_macro_cindex']) for r in m if r['method']=='end_only']
    benchmark=.6587537502979848
    def interval(diff,se,df):
        critical=stats.t.ppf(.975,df)
        return [diff-critical*se,diff+critical*se]
    va,vb=statistics.variance(a)/len(a),statistics.variance(b)/len(b)
    difference=statistics.mean(b)-statistics.mean(a)
    df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
    write(output/'published_seed_statistics.json',{
        'source':'published selected seed-level records; training is a separate command',
        'two_g_vs_top12':{'difference':statistics.mean(a)-benchmark,'p':stats.ttest_1samp(a,benchmark).pvalue,
                         'ci95':interval(statistics.mean(a)-benchmark,va**.5,len(a)-1)},
        'end_only_vs_two_g':{'difference':difference,'p':stats.ttest_ind(b,a,equal_var=False).pvalue,
                            'ci95':interval(difference,(va+vb)**.5,df)}})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','train','exchange','locked','statistics','baselines','all'])
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    parser.add_argument('--seed',type=int,default=20260523)
    parser.add_argument('--policy-seed',type=int,default=2026080903)
    parser.add_argument('--upstream-dir',type=Path,help='Existing G and resumed-2G outputs for exchange-only runs.')
    parser.add_argument('--budget',choices=['G','2G','resume2G'],default='G')
    parser.add_argument('--strategy',choices=['boundary','live_online','end_only'],default='end_only')
    parser.add_argument('--endpoint',choices=[ep['name'] for ep in ENDPOINTS])
    parser.add_argument('--smoke',action='store_true',help='Explicit reduced-budget training; use a separate output directory.')
    args=parser.parse_args();data=args.data_dir.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.stage=='statistics':statistics_report(out);return
    if args.stage=='locked':locked(data,out);return
    preflight(data,out)
    if args.stage=='preflight':return
    inputs=aliases(data,out)
    os.environ.update(environment(out,args.seed))
    if args.stage=='baselines':
        import run_four_endpoint_deepsurv_rsf_20260812 as baseline
        for cfg,ep in zip(baseline.ENDPOINTS, [ENDPOINTS[0],ENDPOINTS[1],ENDPOINTS[2],ENDPOINTS[3]]):
            cfg['eicu_path']=str(data/f"{ep['input_stem']}_eicu.xlsx");cfg['mimic_path']=str(data/f"{ep['input_stem']}_mimic.xlsx")
        baseline.run(out/'baselines',None,False);return
    if args.stage=='train':train(inputs,out,args.seed,args.budget,args.endpoint,args.smoke)
    elif args.stage=='exchange':exchange(out,args.seed,args.strategy,args.policy_seed,args.upstream_dir.resolve() if args.upstream_dir else None)
    else:
        if args.smoke or args.endpoint:raise ValueError('Use train for a reduced or single-endpoint run.')
        train(inputs,out,args.seed,'G');train(inputs,out,args.seed,'resume2G')
        for strategy in ['boundary','live_online','end_only']:exchange(out,args.seed,strategy,args.policy_seed)
        locked(data,out);statistics_report(out)

if __name__=='__main__':main()
