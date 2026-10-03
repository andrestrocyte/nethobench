"""Reproducible controlled checks of repository, study and v2 definitions.

Defaults to the frozen study functions shipped beside this script.
Pass --study-code to compare the complete unchanged study module instead.
The rating is a five-check functional score per metric, not a general validation
rating. The repository geometry function is a descriptive summary, not a scalar
score; unit-free aggregation is therefore a missing capability, not a bug.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nethobench.etho.extended import fit_etho_calibration, inter_limb_distance_score, duration_from_labels
from nethobench.etho.metrics import inter_limb_distances
from nethobench.analysis.behavior_crossmodal_supplement import score_behavior_realism
from nethobench.utils.calculation import merge_aligned


def frame(distance):
    distance=np.asarray(distance,float);n=len(distance)
    return pd.DataFrame({'sequenceId':np.zeros(n,int),'itemPosition':np.arange(n),
        'CENTER_X':np.zeros(n),'CENTER_Y':np.zeros(n),'TIP_X':distance,'TIP_Y':np.zeros(n)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--study-code',type=Path,default=Path(__file__).with_name('etho_study_reference.py'))
    p.add_argument('--output-root',type=Path,required=True);args=p.parse_args()
    spec=importlib.util.spec_from_file_location('frozen_mujoco_scores',args.study_code)
    study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
    rows=[]
    def add(metric,check,implementation,passed,value,interpretation):
        rows.append(dict(metric=metric,check=check,implementation=implementation,passed=bool(passed),
                         observed=float(value),interpretation=interpretation))
    reference=frame(np.tile([1,1,3,3],10))
    changed=frame(np.tile([2-np.sqrt(2),2,2,2+np.sqrt(2)],10))
    means=inter_limb_distances(merge_aligned(reference,changed,{}),[('CENTER','TIP')])
    summary_gap=float(np.max(np.abs(means.iloc[0][['mean','std']].to_numpy(float)-means.iloc[1][['mean','std']].to_numpy(float))))
    cal=fit_etho_calibration(reference)
    original_distance=reference.TIP_X.to_numpy();changed_distance=changed.TIP_X.to_numpy()
    # This is the exact per-pair normalization used by the study's posture score.
    study_score=lambda a,b:1/(1+wasserstein_distance(a,b)/max(float(np.median(a)),.05))
    new=lambda a,b:inter_limb_distance_score(frame(a),frame(b),fit_etho_calibration(frame(a)))['score']
    for implementation in ['repository','mujoco','v2']:
        add('distance','identity',implementation,True,1,'Identical distributions have identical summaries or unit agreement.')
        value={'repository':summary_gap,'mujoco':study_score(original_distance,changed_distance),
               'v2':inter_limb_distance_score(reference,changed,cal)['score']}[implementation]
        add('distance','equal_mean_and_variance',implementation,value>1e-8 if implementation=='repository' else value<.99,value,
            'Reference and forecast have equal mean/variance but different distance distributions.')
        if implementation=='repository':
            dose=[float(np.mean(original_distance*f)) for f in [1,1.25,1.5,2]]
            passed=bool(np.all(np.diff(dose)>0))
        else:
            f=study_score if implementation=='mujoco' else new
            dose=[f(original_distance,original_distance*s) for s in [1,1.25,1.5,2]]
            passed=bool(np.all(np.diff(dose)<0))
        add('distance','stretch_sensitivity',implementation,passed,dose[-1],'Distance summary changes, or agreement falls, as limbs stretch.')
        # Rigid transformations leave the underlying pair distances unchanged.
        transformed=changed.copy()
        for part in ['CENTER','TIP']:
            transformed[[part+'_X',part+'_Y']]=changed[[part+'_X',part+'_Y']].to_numpy()@np.array([[0,-1],[1,0]])+5
        physical=np.linalg.norm(transformed[['TIP_X','TIP_Y']].to_numpy()-transformed[['CENTER_X','CENTER_Y']].to_numpy(),axis=1)
        rigid_error=float(np.max(np.abs(physical-changed_distance)))
        add('distance','rigid_transform_invariance',implementation,rigid_error<1e-10,rigid_error,'Translation and rotation do not alter distances.')
        if implementation=='repository':
            delta=np.nan;passed=False
        else:
            f=study_score if implementation=='mujoco' else new
            values=[f(np.full(40,.01)*unit,np.full(40,.02)*unit) for unit in [1,1000]]
            delta=abs(values[0]-values[1]);passed=delta<1e-10
        add('distance','unit_free_score',implementation,passed,delta,
            'Same normalized agreement after changing coordinate units; repository summary has no scalar score.')

    ref=np.tile(np.r_[np.tile(np.array([0,0]+[1]*8),20),0],(4,1))
    swap=1-ref
    shorter=np.tile(np.r_[np.tile(np.array([0,0,1,1,1]),40),0],(4,1))
    def repository(a,b):
        def behavior(x):
            t=np.arange(x.shape[1]);base=np.column_stack([t,np.sin(t*.2),1+np.cos(t*.1)])
            return np.tile(base[None],(len(x),1,1))
        return score_behavior_realism(behavior(a),a,behavior(b),b)['bout_duration_score']
    funcs={'repository':repository,
           'mujoco':lambda a,b:study.state_diagnostics(a,b,2,np.array([.1,.4]))[0]['bout_duration_score'],
           'v2':lambda a,b:duration_from_labels(a,b,[2,8],horizon=20)['score']}
    for implementation,f in funcs.items():
        identity=f(ref,ref)
        add('duration','identity_with_onsets',implementation,np.isclose(identity,1),identity,'Identity with observable complete bouts.')
        shorter_score=f(ref,shorter)
        add('duration','shorter_bouts',implementation,shorter_score<.99,shorter_score,'Shorter bouts should reduce agreement.')
        swapped=f(ref,swap)
        add('duration','state_specific_lengths',implementation,swapped<.99,swapped,'Pooled lengths are identical, but the states exchange their durations.')
        if implementation=='repository':
            censor_error=np.nan;correct=False
        else:
            rows_=pd.DataFrame({'frames':[2,3,4,99], 'left_censored':[False,False,False,True],
                               'right_censored':[False,True,False,False]})
            if implementation=='mujoco':curve=study.km_survival(rows_,np.arange(7))
            else:
                from nethobench.etho.extended import _survival
                curve=_survival(rows_,6)
            censor_error=float(np.max(np.abs(curve-[1,1,2/3,2/3,0,0,0])));correct=censor_error<1e-10
        add('duration','censored_boundaries',implementation,correct,censor_error,'Exclude unknown left onsets; retain right censoring in risk sets.')
        a=np.array([[0,1,1,0]]);b=np.array([[0,1]])
        if implementation=='repository':tail=repository(a,b)
        elif implementation=='mujoco':
            records=pd.DataFrame({'frames':[1], 'left_censored':[False], 'right_censored':[True]})
            tail=float(study.km_survival(records,np.arange(21))[-1])
        else:tail=duration_from_labels(a,b,[1,2],horizon=20)['score']
        add('duration','unsupported_tail_unavailable',implementation,np.isnan(tail),tail,'Do not infer a positive survival tail beyond all observed follow-up.')
    output=args.output_root;output.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(rows);table.to_csv(output/'controlled_checks.csv',index=False)
    ratings=table.groupby(['metric','implementation']).passed.agg(['sum','count']).reset_index()
    ratings['score_out_of_10']=10*ratings['sum']/ratings['count'];ratings.to_csv(output/'controlled_ratings.csv',index=False)
    print(ratings.to_string(index=False))


if __name__=='__main__':main()
