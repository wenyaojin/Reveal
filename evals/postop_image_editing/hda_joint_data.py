"""Build aligned, masked supervision for the five-domain HDA research pilot."""
import hashlib
import json
import os
import re
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps

ROOT=Path(os.environ.get('POSTOP_WORKSPACE', 'downloads/postop')).resolve()
DATA=ROOT/'training_adapters/hda_joint_v1'
OUT=ROOT/'validation_outputs/hda_joint_v1'
CKPT=ROOT/'checkpoints/hda_joint_v1'
SIZE=384
REGIONS={'Eyelid':[[33,133,159,145],[362,263,386,374]],'Eyebrow':[[70,63,105,66,107],[336,296,334,293,300]],'Nose':[[168,98,97,2,326,327,4]],'FacialBones':[[172,136,150,149],[397,365,379,378],[152,148,377]],'Facelift':[[132,172,136,205],[361,397,365,425]]}
PROMPTS={k:f'Make a realistic subtle postoperative {v} edit. Preserve the same person, expression and skin texture.' for k,v in zip(REGIONS,['eyelid','eyebrow','nose shape','lower facial contour','lower cheek lift'])}


def read(p):
    with Image.open(p) as im:
        return np.array(ImageOps.exif_transpose(im).convert('RGB'))


def detector():
    path=ROOT/'checkpoints/eyelid_shape_v1/face_landmarker.task'
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),num_faces=2))


def points(im,det):
    faces=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=im)).face_landmarks
    if len(faces)!=1:
        raise ValueError('face count')
    return np.asarray([(p.x*im.shape[1],p.y*im.shape[0]) for p in faces[0]],np.float32)


def crop_geometry(pts,cat):
    groups=[pts[ids] for ids in REGIONS[cat]]
    eye_dist=np.linalg.norm(pts[33]-pts[263])
    if eye_dist<25:
        raise ValueError('insufficient resolution')
    shapes=[]
    for g in groups:
        lo,hi=g.min(0),g.max(0)
        margin={'Eyelid':(.05,.06),'Eyebrow':(.05,.05),'Nose':(.06,.07),'FacialBones':(.11,.12),'Facelift':(.10,.12)}[cat]
        shapes.append(((lo+hi)/2,(hi-lo)*.65+eye_dist*np.array(margin)))
    lo=np.min([c-r for c,r in shapes],0)
    hi=np.max([c+r for c,r in shapes],0)
    center=(lo+hi)/2
    scale=SIZE/(max(hi-lo)*1.28)
    mat=np.array([[scale,0,SIZE/2-scale*center[0]],[0,scale,SIZE/2-scale*center[1]]])
    yy,xx=np.mgrid[:SIZE,:SIZE]
    mask=np.zeros((SIZE,SIZE))
    for c,r in shapes:
        cx,cy=scale*c+mat[:,2]
        radius=((xx-cx)/(r[0]*scale))**2+((yy-cy)/(r[1]*scale))**2
        mask=np.maximum(mask,np.clip((1-radius)/.4,0,1))
    return mat,np.uint8(mask*255)


def crop(im,mat):
    return cv2.warpAffine(im,mat,(SIZE,SIZE),borderMode=cv2.BORDER_REFLECT_101)


def prepare():
    files={}
    reserved={}
    for cat in REGIONS:
        for p in sorted((ROOT/'HDA-PlasticSurgery'/cat).iterdir()):
            m=re.fullmatch(r'(\d+)_([ab])(?:\.\w+)+',p.name)
            if m:
                files.setdefault((cat,int(m[1])),{})[m[2]]=p
            else:
                m=re.fullmatch(r'(.+)_([ab])(?:\.\w+)+',p.name)
                if m: reserved.setdefault((cat,m[1]),{})[m[2]]=p
    parent={k:k for k in files}
    def group(k):
        while k!=parent[k]:
            k=parent[k]
        return k
    hashes=[]
    for k,pair in files.items():
        for path in pair.values():
            small=cv2.resize(cv2.cvtColor(read(path),cv2.COLOR_RGB2GRAY),(32,32))
            vals=cv2.dct(small.astype('float32'))[:8,:8].ravel()[1:]
            fp=int(''.join('1' if v else '0' for v in vals>np.median(vals)),2)
            for key,other in hashes:
                if (fp^other).bit_count()<=7:
                    a,b=sorted((group(k),group(key)))
                    parent[b]=a
            hashes.append((k,fp))
    old=json.loads(Path(__file__).with_name('eyelid_prior_split.json').read_text())
    oldsplit={int(i):split for i,g,split in old['splits']}
    partitions={}
    for k in files:
        g=group(k)
        hint=oldsplit.get(k[1]) if k[0]=='Eyelid' else ('train' if k[1]<=16 else None)
        if hint: partitions.setdefault(g,set()).add(hint)
    DATA.mkdir(exist_ok=True,parents=True)
    rows=[]
    with detector() as det:
        for (cat,i),pair in files.items():
            g=group((cat,i))
            prior=partitions.get(g,set())
            seed=int(hashlib.sha256(str(g).encode()).hexdigest()[:8],16)%10
            split='excluded' if len(prior)>1 else next(iter(prior),('train' if seed<7 else 'val' if seed==7 else 'test'))
            row=dict(category=cat,case=i,group=list(g),split=split,source=str(pair['b'].relative_to(ROOT)),followup=str(pair['a'].relative_to(ROOT)))
            rows.append(row)
            if split=='excluded': continue
            try:
                src,tgt=read(pair['b']),read(pair['a'])
                a,b=points(src,det),points(tgt,det)
                anchors=[33,133,362,263,1,61,291] if cat!='Nose' else [33,133,362,263,61,291]
                if cat=='Eyelid': anchors=[168,1,98,327,61,291]
                M,_=cv2.estimateAffinePartial2D(b[anchors],a[anchors],method=cv2.LMEDS)
                if M is None: raise ValueError('alignment')
                err=np.linalg.norm(b[anchors]@M[:,:2].T+M[:,2]-a[anchors],axis=1)
                if np.max(err)>.14*np.linalg.norm(a[33]-a[263]): raise ValueError('pose or expression')
                mat,mask=crop_geometry(a,cat)
                src_crop=crop(src,mat)
                target=crop(tgt,(np.vstack([mat,[0,0,1]])@np.vstack([M,[0,0,1]]))[:2])
                bg=mask==0
                offset=np.median(src_crop[bg].astype(float)-target[bg],axis=0)
                alpha=mask[:,:,None]/255
                label=np.uint8(np.clip(src_crop*(1-alpha)+(target+offset)*alpha,0,255))
                dest=DATA/cat/str(i)
                dest.mkdir(exist_ok=True,parents=True)
                for name,img in [('pre',src_crop),('target',label),('mask',mask)]:
                    Image.fromarray(img).save(dest/(name+'.png'))
                row['aligned']=True
            except ValueError as e:
                row['reason']=str(e)
    for (cat,label),pair in reserved.items():
        rows.append(dict(category=cat,case=label,group=[cat,label],split='excluded',
                         reason='non-numeric identifier pending identity review',
                         source=str(pair['b'].relative_to(ROOT)),followup=str(pair['a'].relative_to(ROOT))))
    (DATA/'manifest.json').write_text(json.dumps(rows,indent=1))
    print('saved',len(rows))

if __name__=='__main__':
    prepare()
