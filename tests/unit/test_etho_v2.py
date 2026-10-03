import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nethobench.etho.extended import (
    _features, _labels, _survival, bout_records, compute_extended_etho_metrics,
    duration_from_labels, fit_etho_calibration, inter_limb_distance_score,
)
from nethobench.etho.pipeline import CORE_V1, CORE_V2, compute_etho_scores
from nethobench.etho import legacy_pipeline
from nethobench.etho.metrics import inter_limb_distances
from nethobench.utils.calculation import merge_aligned


def poses(distances, scale=1):
    distances=np.asarray(distances,float)
    n=len(distances)
    x=np.cumsum(1+0.5*np.sin(np.arange(n)*.7))
    return pd.DataFrame({'sequenceId':np.zeros(n,dtype=int),'itemPosition':np.arange(n),
        'CENTER_X':x*scale,'CENTER_Y':np.zeros(n),'TIP_X':(x+distances)*scale,'TIP_Y':np.zeros(n)})


def test_distance_detects_equal_mean_and_variance_distributions():
    reference=poses(np.tile([1,1,3,3],10))
    prediction=poses(np.tile([2-np.sqrt(2),2,2,2+np.sqrt(2)],10))
    old=inter_limb_distances(merge_aligned(reference,prediction,{}),[('CENTER','TIP')])
    np.testing.assert_allclose(old.iloc[0][['mean','std']].astype(float),old.iloc[1][['mean','std']].astype(float),atol=1e-12)
    cal=fit_etho_calibration(reference)
    new=inter_limb_distance_score(reference,prediction,cal)
    assert 0<new['score']<.9
    assert inter_limb_distance_score(reference,reference,cal)['score']==1


def test_distance_units_and_rigid_transform():
    values=[]
    for units in [1e-5,1,1000]:
        a,b=poses(np.full(40,.01),units),poses(np.full(40,.02),units)
        cal=fit_etho_calibration(a)
        values.append(inter_limb_distance_score(a,b,cal)['score'])
        transformed=b.copy()
        for part in ['CENTER','TIP']:
            transformed[[part+'_X',part+'_Y']]=b[[part+'_X',part+'_Y']].to_numpy() @ np.array([[0,-1],[1,0]]) + 20*units
        assert inter_limb_distance_score(a,transformed,cal)['score']==pytest.approx(values[-1])
    np.testing.assert_allclose(values,.5,atol=1e-9)


def test_stretch_dose_decreases_without_moving_centroid():
    reference=poses(np.ones(60));cal=fit_etho_calibration(reference)
    values=[inter_limb_distance_score(reference,poses(np.ones(60)*factor),cal)['score'] for factor in [1,1.25,1.5,2]]
    assert all(a>b for a,b in zip(values,values[1:]))


def test_state_specific_duration_detects_pooled_length_swap():
    a=[np.r_[np.tile([0,0]+[1]*8,20),0]]*3
    b=[1-a[0]]*3
    ra,rb=bout_records(a),bout_records(b)
    np.testing.assert_equal(np.sort(ra.frames),np.sort(rb.frames))
    actual=duration_from_labels(a,b,[2,8],horizon=20)
    assert actual['score']<.7
    assert duration_from_labels(a,a,[2,8],horizon=20)['score']==1


def test_censoring_and_supported_tail():
    records=pd.DataFrame({'frames':[2,3,4,99], 'left_censored':[False,False,False,True],
                          'right_censored':[False,True,False,False]})
    curve=_survival(records,6)
    np.testing.assert_allclose(curve,[1,1,2/3,2/3,0,0,0])
    assert _survival(records.iloc[[1]],6) is None
    result=duration_from_labels([[0,1,1,0]],[[0,1]],[1,2],horizon=20)
    assert np.isnan(result['score'])
    assert result['states'][1]['status']=='prediction_tail_unavailable'


def test_missing_onsets_and_constant_reference_are_distinguished():
    reference=[np.tile([0,0,1,1],12)]
    result=duration_from_labels(reference,[np.zeros(48,int)],[2,2],horizon=8)
    assert result['score']==0
    undefined=duration_from_labels([np.zeros(48,int)],[np.zeros(48,int)],[2],horizon=8)
    assert np.isnan(undefined['score']) and undefined['reference_coverage']==0


def test_duration_unit_scaling_and_sequence_separation():
    a=[np.tile([0,0]+[1]*4,12)]*2
    b=[np.tile([0]*3+[1]*3,12)]*2
    x=duration_from_labels(a,b,[2,4],horizon=12)['score']
    y=duration_from_labels([np.repeat(v,5) for v in a],[np.repeat(v,5) for v in b],[10,20],horizon=60)['score']
    assert x==pytest.approx(y)
    runs=bout_records([[0,0],[0,0]])
    assert runs.frames.tolist()==[2,2]
    assert runs.left_censored.all() and runs.right_censored.all()


def test_gaps_and_invalid_identifiers():
    frame=poses(np.ones(12));frame.loc[6:,'itemPosition']+=4
    features=_features(frame,{})
    assert [len(f) for f in features]==[4,4]
    frame.loc[1,'itemPosition']=0
    with pytest.raises(ValueError,match='unique'):_features(frame,{})


def test_nearest_center_uses_squared_euclidean_distance():
    cal={'feature_mean':[0,0],'feature_scale':[1,1],'state_centers':[[10,-10],[1,1]]}
    assert _labels([np.array([[0,0]])],cal)[0][0]==1


def test_calibration_roundtrip_and_prediction_independence(tmp_path):
    ref=poses(np.ones(80));pred=poses(np.ones(80)*1.5)
    cal=fit_etho_calibration(ref)
    frozen=copy.deepcopy(cal)
    file=tmp_path/'calibration.json';file.write_text(json.dumps(cal))
    pair=merge_aligned(ref,pred,{})
    a,details=compute_extended_etho_metrics(pair,{'etho_calibration':cal})
    b,_=compute_extended_etho_metrics(pair,{'etho_calibration':file})
    assert a==b and cal==frozen and details['calibration_source']=='supplied'
    with pytest.raises(ValueError,match='conflicts'):
        compute_extended_etho_metrics(pair,{'etho_calibration':cal,'sampling_interval':.05})


@pytest.mark.parametrize('bad', [np.nan,np.inf])
def test_nonfinite_geometry_does_not_silently_skip_parts(bad):
    reference=poses(np.ones(30));pred=reference.copy();pred.loc[0,'TIP_X']=bad
    with pytest.raises(ValueError,match='Finite'):
        inter_limb_distance_score(reference,pred,fit_etho_calibration(reference))


def test_versioned_pipeline_preserves_all_legacy_components(tmp_path):
    root=Path(__file__).resolve().parents[1]/'resources/data/behavioural'
    gt,pred=root/'behav-ground-truth.csv',root/'behav-predictions.csv'
    old=legacy_pipeline.compute_etho_scores(gt,pred)
    legacy=compute_etho_scores(gt,pred,cfg={'etho_score_version':'legacy_v1'})
    assert json.dumps(old,sort_keys=True)==json.dumps(legacy,sort_keys=True)
    new=compute_etho_scores(gt,pred,cfg={'etho_details_path':tmp_path/'details.json'})
    for name in CORE_V1:assert new[0][name]==pytest.approx(old[0][name],nan_ok=True)
    assert new[0]['legacy_composite_score']==old[0]['composite_score']
    values=np.asarray([new[0][name] for name in CORE_V2])
    if np.isfinite(values).all():
        expected=np.exp(np.mean(np.log(np.clip(values,1e-6,1-1e-6))))
        assert new[0]['composite_score']==pytest.approx(expected)
    else:
        assert np.isnan(new[0]['composite_score'])
    assert json.dumps(new[1:],sort_keys=True)==json.dumps(old[1:],sort_keys=True)
    assert json.loads((tmp_path/'details.json').read_text())['protocol']=='etho-v2'


def test_finite_ten_component_identity():
    # Enough independent trajectories for the original shape component, plus
    # many internal kinematic transitions for the added duration component.
    frames=[]
    for i in range(12):
        f=poses(1+.1*np.sin(np.arange(100)*(.2+i*.01)))
        f['sequenceId']=i
        for part in ['CENTER','TIP']:
            f[part+'_Y']=np.sin(np.arange(100)*(.12+i*.02))
        f['NOSE_X']=f.TIP_X;f['NOSE_Y']=f.TIP_Y
        f['TAIL_BASE_X']=f.CENTER_X-.5;f['TAIL_BASE_Y']=f.CENTER_Y
        frames.append(f)
    reference=pd.concat(frames,ignore_index=True)
    scores,*_=compute_etho_scores(paired_df=merge_aligned(reference,reference,{}),
                                cfg={'duration_horizon_frames':2})
    assert np.isfinite([scores[k] for k in CORE_V2]).all()
    expected=np.exp(np.mean(np.log(np.clip([scores[k] for k in CORE_V2],1e-6,1-1e-6))))
    assert scores['composite_score']==pytest.approx(expected)
    assert scores['inter_limb_distance_score']==1
    assert scores['bout_duration_score']==1


def test_unknown_version_rejected():
    with pytest.raises(ValueError,match='etho_score_version'):
        compute_etho_scores(cfg={'etho_score_version':'typo'})


def test_degenerate_distance_calibration():
    ref=poses(np.zeros(30));cal=fit_etho_calibration(ref)
    assert inter_limb_distance_score(ref,ref,cal)['score']==1
    assert inter_limb_distance_score(ref,poses(np.ones(30)),cal)['score']==0
