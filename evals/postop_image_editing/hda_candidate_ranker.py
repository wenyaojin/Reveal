"""Select a local edit strength using paired supervision and source-only features."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import Ridge

from hda_joint_data import ROOT, DATA, OUT, CKPT, REGIONS, detector, points, read
from hda_joint_shape import anchors, normalized, render

STRENGTHS = (0., .5, 1., 1.5)


def describe(pts, category, shape):
    xy, _, _ = normalized(pts, category)
    centered = xy.ravel() - shape[category+'_center']
    return np.r_[centered, np.linalg.norm(centered)]


def candidate(im, pts, category, shape, strength):
    state = dict(shape)
    for key in ('_coef', '_mean'):
        state[category+key] = shape[category+key]*strength
    return render(im, pts, category, state)


def measure(row, det, shape):
    cat = row['category']
    im = read(ROOT/row['source'])
    pts = points(im, det)
    post = points(read(ROOT/row['followup']), det)
    anchor = anchors(cat)
    mat, _ = cv2.estimateAffinePartial2D(post[anchor], pts[anchor], method=cv2.LMEDS)
    if mat is None:
        raise ValueError('alignment failed')
    ids = sum(REGIONS[cat], [])
    target = post[ids]@mat[:,:2].T + mat[:,2]
    span = np.linalg.norm(pts[33]-pts[263])
    errors = []
    for strength in STRENGTHS:
        try:
            edited = pts if strength == 0 else points(candidate(im, pts, cat, shape, strength), det)
            errors.append(float(np.linalg.norm(edited[ids]-target, axis=1).mean()/span))
        except ValueError:
            errors.append(None)
    return describe(pts, cat, shape), errors


def select(features, model, category):
    scores = features @ model[category+'_coef'].T + model[category+'_bias']
    return int(np.argmin(np.r_[0., scores]))


def train(det, shape):
    manifest = json.loads((DATA/'manifest.json').read_text())
    output = OUT/'candidate_ranker'
    output.mkdir(parents=True, exist_ok=True)
    models, results = {}, []
    for cat in REGIONS:
        rows = [r for r in manifest if r['category']==cat and r.get('aligned')
                and r['split'] in ('val', 'test')]
        measurements = [(r, *measure(r, det, shape)) for r in rows]
        val = [(x,e) for r,x,e in measurements if r['split']=='val' and None not in e]
        X = np.stack([x for x,e in val])
        errors = np.asarray([e for x,e in val])
        reg = Ridge(alpha=.15).fit(X, errors[:,1:]-errors[:,:1])
        models[cat+'_coef'] = reg.coef_
        models[cat+'_bias'] = reg.intercept_
        fixed = int(errors.mean(0).argmin())
        for row, feats, errs in measurements:
            if row['split']!='test':
                continue
            chosen = select(feats, models, cat)
            results.append(dict(category=cat, case=row['case'], selected=chosen,
                                fixed=fixed, errors=errs))
            source = read(ROOT/row['source'])
            pts = points(source, det)
            result = candidate(source, pts, cat, shape, STRENGTHS[chosen])
            Image.fromarray(result).save(output/f"{cat}_{row['case']}.png")
    np.savez(CKPT/'candidate_ranker.npz', **models)
    (output/'metrics.json').write_text(json.dumps(results, indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--category', choices=REGIONS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if any((args.source, args.category, args.output)) and not all((args.source, args.category, args.output)):
        parser.error('single-image inference requires source, category and output')
    with np.load(CKPT/'shape.npz', allow_pickle=False) as stored:
        shape = dict(stored)
    with detector() as det:
        if args.source is None:
            train(det, shape)
        else:
            im = read(args.source)
            pts = points(im, det)
            with np.load(CKPT/'candidate_ranker.npz', allow_pickle=False) as ranker:
                index = select(describe(pts, args.category, shape), ranker, args.category)
            Image.fromarray(candidate(im, pts, args.category, shape, STRENGTHS[index])).save(args.output)
