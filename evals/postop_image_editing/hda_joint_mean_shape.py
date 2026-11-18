"""Fit a fixed mean local optical-flow field per surgery domain as a baseline."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from hda_joint_data import ROOT, DATA, REGIONS, SIZE, OUT, CKPT, detector, read, points, crop_geometry, crop


def gray(im):
    return cv2.equalizeHist(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY))


def compute():
    manifest=json.loads((DATA/'manifest.json').read_text())
    CKPT.mkdir(parents=True,exist_ok=True)
    result=[]
    for cat in REGIONS:
        fields=[]
        for row in manifest:
            if row['category']!=cat or row['split']!='train' or not row.get('aligned'): continue
            d=DATA/cat/str(row['case'])
            src,target=[gray(read(d/name)) for name in ['pre.png','target.png']]
            flow=cv2.calcOpticalFlowFarneback(target,src,None,.5,3,25,3,7,1.5,0)
            fields.append(cv2.resize(flow,(48,48),interpolation=cv2.INTER_AREA))
        field=np.median(fields,axis=0)
        np.save(CKPT/f'mean_flow_{cat}.npy',field)
        yy,xx=np.indices((SIZE,SIZE),dtype=np.float32)
        mx,my=cv2.resize(field,(SIZE,SIZE)).transpose(2,0,1)
        for r in manifest:
            if r['category']!=cat or r['split']!='test' or not r.get('aligned'):continue
            d=DATA/cat/str(r['case'])
            src,target=[read(d/name) for name in ['pre.png','target.png']]
            mask=cv2.imread(str(d/'mask.png'),0)/255
            pred=cv2.remap(src,(xx+mx*mask).astype('float32'),(yy+my*mask).astype('float32'),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
            def err(im):
                return float(abs(cv2.GaussianBlur(im.astype('float32'),(0,0),3)-cv2.GaussianBlur(target.astype('float32'),(0,0),3))[mask>.5].mean())
            result.append(dict(category=cat,case=r['case'],unchanged=err(src),mean_flow=err(pred)))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'mean_flow_test.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    compute()
