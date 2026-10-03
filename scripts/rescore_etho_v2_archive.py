"""Evaluate v2 additions on frozen MuJoCo forecasts without retraining.

The archive uses the standalone study layout (data/population_*/train and
results/evaluation/population_*). Old core rows are supplied as a CSV. Neither
the archive nor the original tables are modified.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nethobench.etho.extended import fit_etho_calibration,compute_extended_etho_metrics
from nethobench.etho.pipeline import CORE_V1,CORE_V2
from nethobench.utils.helpers import geometric_mean_scores

PARTS=['CENTER','HEAD','REAR','LEFT_TORSO','RIGHT_TORSO','FOOT_1','FOOT_2','FOOT_3','FOOT_4']


def frame(pose):
    n,t=pose.shape[:2]
    data={'sequenceId':np.repeat(np.arange(n),t),'itemPosition':np.tile(np.arange(t),n)}
    for i,part in enumerate(PARTS):
        for j,axis in enumerate(['X','Y']):data[f'{part}_{axis}']=pose[:,:,i,j].ravel()
    return pd.DataFrame(data)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--scores',type=Path,required=True);parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();out=args.output_root;out.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(args.scores);old=old[old.checkpoint=='best'];rows=[];support=[]
    with tarfile.open(args.archive) as archive:
        def load(path):
            with archive.extractfile(path) as stream:return np.load(io.BytesIO(stream.read()))
        for teacher,group in old.groupby('teacher'):
            cfg={'sampling_interval':.05,'center_part':'CENTER','duration_horizon_frames':20}
            calibration=fit_etho_calibration(frame(load(f'data/population_{teacher}/train/episodes.npz')['pose']),cfg)
            (out/f'calibration_{teacher}.json').write_text(json.dumps(calibration,indent=2))
            cfg['etho_calibration']=calibration
            for split,part in group.groupby('split'):
                reference=frame(load(f'data/population_{teacher}/{split}/episodes.npz')['pose'][:,40:200])
                for (model,seed),draws in part.groupby(['model','seed']):
                    tag='linear_svd' if model=='linear_ar' else ('controls' if model in ['identity','simulator_oracle','simulator_mean8'] else f'seed_{seed if seed else 101}')
                    name=f'results/evaluation/population_{teacher}/{tag}/{split}_{model}_{seed}_best.npz'
                    predictions=load(name)['prediction']
                    for _,row in draws.iterrows():
                        prediction=frame(predictions[int(row.draw)])
                        paired=reference.merge(prediction,on=['sequenceId','itemPosition'],suffixes=('_gt','_inf'),validate='one_to_one')
                        extra,details=compute_extended_etho_metrics(paired,cfg)
                        values=[row[k] for k in CORE_V1]+[extra[k] for k in CORE_V2[8:]]
                        value=geometric_mean_scores(values) if np.isfinite(values).all() else np.nan
                        rows.append(dict(teacher=teacher,split=split,model=model,seed=seed,draw=int(row.draw),
                            legacy_composite_score=row.composite_score,study_posture_score=row.posture_distribution_score,
                            study_bout_duration_score=row.bout_duration_score,**extra,composite_score_v2=value,
                            duration_reference_coverage=details['duration']['reference_coverage'],
                            duration_comparison_coverage=details['duration']['comparison_coverage']))
                        for state in details['duration']['states']:
                            support.append(dict(teacher=teacher,split=split,model=model,seed=seed,draw=int(row.draw),**state))
                    print('SCORED',teacher,split,model,seed,flush=True)
            pd.DataFrame(rows).to_csv(out/'rescored_rows.csv',index=False)
            pd.DataFrame(support).to_csv(out/'duration_support.csv',index=False)
    data=pd.DataFrame(rows)
    # Preserve unavailable rows: pandas' default mean would silently skip them.
    numeric=[c for c in data.select_dtypes('number') if c not in ['teacher','seed','draw']]
    strict=lambda x: x.mean(skipna=False)
    seeds=data.groupby(['teacher','split','model','seed'])[numeric].agg(strict).reset_index()
    populations=seeds.groupby(['teacher','split','model'])[numeric].agg(strict).reset_index()
    populations.to_csv(out/'population_summary.csv',index=False)
    populations.groupby(['split','model'])[numeric].agg(strict).to_csv(out/'model_summary.csv')
    print('COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
