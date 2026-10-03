"""Frozen study definitions for comparative validation, not public scoring defaults.

Only the four unmodified functions needed by the controlled comparison are
included. See docs/validation/etho_v2/provenance.json for the source digest.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

def bouts(labels,dt=.05):
    rows=[]
    for sequence,row in enumerate(np.asarray(labels)):
        edges=np.r_[0,np.flatnonzero(row[1:]!=row[:-1])+1,len(row)]
        for a,b in zip(edges[:-1],edges[1:]):
            rows.append(dict(sequence=sequence,state=int(row[a]),frames=int(b-a),seconds=(b-a)*dt,
                             left_censored=bool(a==0),right_censored=bool(b==len(row))))
    return pd.DataFrame(rows)

def km_survival(frame,grid):
    """Exclude left-censored runs; retain right-censored runs in risk sets."""
    frame=frame[~frame.left_censored]
    if frame.empty:return None
    duration=frame.frames.to_numpy();event=~frame.right_censored.to_numpy()
    survival=1.;times=[];values=[]
    for t in np.unique(duration):
        risk=np.sum(duration>=t);deaths=np.sum((duration==t)&event)
        survival*=1-deaths/risk;times.append(t);values.append(survival)
    indices=np.searchsorted(times,grid,side='right')-1
    return np.where(indices>=0,np.asarray(values)[np.maximum(indices,0)],1.0)

def js_score(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.sum()==0 or b.sum()==0:return 0.0
    return float(1-jensenshannon(a/a.sum(),b/b.sum(),base=2))

def state_diagnostics(reference_labels,prediction_labels,k,duration_scale,dt=.05):
    a=np.asarray(reference_labels);b=np.asarray(prediction_labels)
    assert a.shape==b.shape
    ca=np.bincount(a.ravel(),minlength=k);cb=np.bincount(b.ravel(),minlength=k)
    transitions=[]
    for x in [a,b]:
        transitions.append(np.bincount((x[:,:-1]*k+x[:,1:]).ravel(),minlength=k*k).reshape(k,k))
    ta,tb=transitions
    weights=ca/max(ca.sum(),1)
    transition=sum(weights[i]*js_score(ta[i],tb[i]) for i in range(k))
    ba,bb=bouts(a,dt),bouts(b,dt)
    grid=np.arange(0,a.shape[1]+1)
    scores=[];support=[]
    for state in range(k):
        ra=ba[ba.state==state];rb=bb[bb.state==state]
        sa=km_survival(ra,grid);sb=km_survival(rb,grid)
        if sa is None:
            value=np.nan
        elif sb is None:
            value=0.
        else:
            restricted_w1=float(np.sum(np.abs(sa[:-1]-sb[:-1]))*dt)
            value=1/(1+restricted_w1/max(duration_scale[state],dt))
        scores.append(value)
        support.append(dict(state=state,reference_bouts=len(ra),prediction_bouts=len(rb),
                            reference_complete=int((~ra.left_censored&~ra.right_censored).sum()),
                            prediction_complete=int((~rb.left_censored&~rb.right_censored).sum()),duration_score=value))
    mask=np.isfinite(scores)
    duration=float(np.average(np.asarray(scores)[mask],weights=weights[mask])) if weights[mask].sum()>0 else np.nan
    return {'state_occupancy_score':js_score(ca,cb),'state_transition_score':float(transition),
            'bout_duration_score':duration,'duration_supported_states':int(mask.sum())},pd.DataFrame(support),ba,bb
