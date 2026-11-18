"""Measure postoperative landmark error on saved images with one fixed alignment."""
import json
import cv2
import numpy as np
from hda_joint_data import ROOT, DATA, OUT, REGIONS, detector, read, points
from hda_joint_shape import anchors

if __name__=='__main__':
    rows=json.loads((DATA/'manifest.json').read_text())
    scores=[]
    with detector() as det:
        for r in rows:
            if r['split']!='test' or not r.get('aligned'):continue
            cat,i=r['category'],r['case']
            ids=sum(REGIONS[cat],[])
            original=points(read(ROOT/r['source']),det)
            post=points(read(ROOT/r['followup']),det)
            anchor=anchors(cat)
            mat,_=cv2.estimateAffinePartial2D(post[anchor],original[anchor],method=cv2.LMEDS)
            target=post[ids]@mat[:,:2].T+mat[:,2]
            span=np.linalg.norm(original[33]-original[263])
            def error(pts):
                return float(np.linalg.norm(pts[ids]-target,axis=1).mean()/span)
            case=dict(category=cat,case=i,original=error(original))
            for variant in ['frozen','trained','shape']:
                try:
                    case[variant]=error(points(read(OUT/variant/'test'/f'{cat}_{i}'/'simulation.png'),det))
                except ValueError:
                    case[variant]=None
            scores.append(case)
    (OUT/'landmark_test.json').write_text(json.dumps(scores,indent=2))
