"""Check the runtime dependency graph and generate all continuation templates."""
import ast
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import run_pipeline as pipeline

def main():
    with tempfile.TemporaryDirectory(prefix='coxgp_check_') as directory:
        output=Path(directory)
        os.environ.update(pipeline.environment(output,20260523))
        for name in ['run_multitarget_profile_optimization','multitarget_abc_runner','relation_aware_mtgps',
                     'single_endpoint_abc_pipeline','endpoint_relation_config','phaseC_relation_transfer',
                     'checkpoint_fair_multitarget_experiments.run_g_to_2g_exchange',
                     'four_endpoint_safe_endonly_core_20260814']:
            importlib.import_module(name)
        import endpoint_relation_config as relations
        assert len(relations.ENDPOINTS)==4
        assert relations.build_phaseA_transfer_matrix(relations.ENDPOINTS).shape==(4,4)
        import run_multitarget_profile_optimization as legacy
        target=pipeline.ENDPOINTS[0]
        template=pipeline.RUNTIME/'single_target_checkpoint.ipynb'
        patched=output/'patched.ipynb'
        legacy.patch_notebook_for_profile(template,patched,output/f"{target['id']}_eicu.xlsx",
                                         output/f"{target['id']}_mimic.xlsx",output,pipeline.parameters(12,20),20260523)
        import multitarget_abc_runner as mt
        mt.relation_patch_notebook()
        for mode in ['boundary','online']:
            env=pipeline.environment(output,20260523)
            env.update({f'MTGP_{mode.upper()}_SOURCE_NOTEBOOK':str(pipeline.RUNTIME/'exchange_checkpoint.ipynb'),
                        f'MTGP_{mode.upper()}_OUTPUT_NOTEBOOK':str(output/f'{mode}.ipynb')})
            pipeline.command(pipeline.RUNTIME/f'checkpoint_fair_multitarget_experiments/build_{mode}_resume_notebook.py',[],env)
            legacy.patch_notebook_for_profile(output/f'{mode}.ipynb',output/f'{mode}_bound.ipynb',
                output/f"{target['id']}_eicu.xlsx",output/f"{target['id']}_mimic.xlsx",output,pipeline.parameters(12,40),20260523)
        env=pipeline.environment(output,20260523)
        env.update(HCA_RESUME_SOURCE_NOTEBOOK=str(template),HCA_RESUME_OUTPUT_NOTEBOOK=str(output/'resume.ipynb'))
        pipeline.command(pipeline.RUNTIME/'checkpoint_no_exchange_G_continuation/build_resume_notebook.py',[],env)
        for path in output.glob('*.ipynb'):
            notebook=json.loads(path.read_text())
            for cell in notebook['cells']:
                if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        sys.path.insert(0,str(pipeline.RUNTIME/'end_only_backend'))
        from exchange_policy_tree import apply_current_acceptance
        cfg={'acceptance':{'minimum_delta_cindex':-.02,'positive_folds_min':2,'worst_fold_drop_min':-.03}}
        assert apply_current_acceptance({'status':'ok','delta':-.005,'fold_deltas':[.01,.01,-.02,-.02]},cfg)['accepted']
        assert not apply_current_acceptance({'status':'ok','delta':-.04,'fold_deltas':[.01,.01,-.09,-.09]},cfg)['accepted']
    print('Runtime imports, four-target matrices, notebook binding, continuation builders and acceptance checks passed.')

if __name__=='__main__':main()
