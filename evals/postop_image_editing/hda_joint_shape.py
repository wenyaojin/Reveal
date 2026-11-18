"""Fit category-specific landmark displacement regressions on the same HDA manifest."""
import argparse
import json
import cv2
import numpy as np
from PIL import Image
from hda_joint_data import ROOT, DATA, CKPT, OUT, REGIONS, SIZE, crop, points, detector, read, crop_geometry


def anchors(cat):
    if cat=='Nose': return [33,133,362,263,61,291]
    if cat=='Eyelid': return [168,1,98,327,61,291]
    return [33,133,362,263,1,61,291]


def normalized(pts,cat):
    mid=(pts[33]+pts[263])/2
    span=pts[263]-pts[33]
    scale=np.linalg.norm(span)
    axis=span/scale
    basis=np.array([axis,[-axis[1],axis[0]]])
    ids=sum(REGIONS[cat],[])
    xy=(pts[ids]-mid)@basis.T/scale
    return xy,scale,basis


def fit(manifest,det):
    state={}
    for cat in REGIONS:
        X,Y=[],[]
        for r in manifest:
            if r['split']!='train' or not r.get('aligned') or r['category']!=cat:continue
            a=points(read(ROOT/r['source']),det)
            b=points(read(ROOT/r['followup']),det)
            ids=anchors(cat)
            M,_=cv2.estimateAffinePartial2D(b[ids],a[ids],method=cv2.LMEDS)
            xy,s,basis=normalized(a,cat)
            ids=sum(REGIONS[cat],[])
            offset=(b[ids]@M[:,:2].T+M[:,2]-a[ids])@basis.T/s
            X.append(xy.flatten())
            Y.append(np.clip(offset,-.075,.075).flatten())
        X,Y=np.array(X),np.array(Y)
        center=X.mean(0)
        # A fixed regularizer and displacement cap, independent of holdout data.
        Z=X-center
        coef=np.linalg.solve(Z.T@Z+np.eye(Z.shape[1])*.04,Z.T@(Y-Y.mean(0)))
        state[cat+'_center']=center
        state[cat+'_coef']=coef
        state[cat+'_mean']=Y.mean(0)
    np.savez(CKPT/'shape.npz',**state)
    return state


def render(im,pts,cat,state):
    xy,s,basis=normalized(pts,cat)
    delta=(xy.flatten()-state[cat+'_center'])@state[cat+'_coef']+state[cat+'_mean']
    delta=np.clip(delta.reshape(-1,2),-.06,.06)*s@basis
    mat,mask=crop_geometry(pts,cat)
    ids=sum(REGIONS[cat],[])
    centers=pts[ids]@mat[:,:2].T+mat[:,2]
    yy,xx=np.indices((SIZE,SIZE),dtype=np.float32)
    numer=np.zeros((SIZE,SIZE,2),np.float32)
    denom=np.full((SIZE,SIZE),.03,np.float32)
    sigma=.09*s*mat[0,0]
    for c,d in zip(centers,delta):
        weight=np.exp(-((xx-c[0])**2+(yy-c[1])**2)/(2*sigma**2))
        numer+=weight[:,:,None]*d
        denom+=weight
    inverse=cv2.invertAffineTransform(mat)
    disp=cv2.warpAffine((numer/denom[:,:,None])*(mask[:,:,None]/255).astype('float32'),inverse,(im.shape[1],im.shape[0]))
    y,x=np.indices(im.shape[:2],dtype=np.float32)
    output=cv2.remap(im,x-disp[:,:,0],y-disp[:,:,1],cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    assert np.array_equal(output[(disp==0).all(2)],im[(disp==0).all(2)])
    return output


def evaluate(rows,det,state,split):
    results=[]
    for r in rows:
        if r['split']!=split or not r.get('aligned'):continue
        cat,i=r['category'],r['case']
        im=read(ROOT/r['source'])
        pts=points(im,det)
        out=render(im,pts,cat,state)
        dest=OUT/'shape'/split/f'{cat}_{i}'
        dest.mkdir(exist_ok=True,parents=True)
        Image.fromarray(out).save(dest/'simulation.png')
        m,mask=crop_geometry(pts,cat)
        orig,pred=crop(im,m),crop(out,m)
        target=read(DATA/cat/str(i)/'target.png')
        def err(a,b):
            return float(abs(cv2.GaussianBlur(a.astype('float32'),(0,0),3)-cv2.GaussianBlur(b.astype('float32'),(0,0),3))[mask>128].mean())
        results.append(dict(category=cat,case=i,original=err(orig,target),edited=err(pred,target),change=err(pred,orig)))
    (OUT/'shape'/split/'metrics.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='cmd',required=True)
    subs.add_parser('train')
    i=subs.add_parser('infer')
    i.add_argument('source')
    i.add_argument('category',choices=list(REGIONS))
    i.add_argument('output')
    args=p.parse_args()
    with detector() as det:
        if args.cmd=='train':
            rows=json.loads((DATA/'manifest.json').read_text())
            model=fit(rows,det)
            for split in ['val','test']: evaluate(rows,det,model,split)
        else:
            im=read(args.source)
            out=render(im,points(im,det),args.category,np.load(CKPT/'shape.npz'))
            Image.fromarray(out).save(args.output,format='PNG')
