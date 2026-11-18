"""Train or run the five-project HDA research LoRA using local FLUX.2 Klein Base."""
import argparse
import json
import os
import random
import time
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from diffusers import Flux2KleinPipeline
from peft import LoraConfig
from hda_joint_data import ROOT, DATA, CKPT, OUT, SIZE, REGIONS, PROMPTS, read, detector, points, crop_geometry, crop

BASE=Path(os.environ.get('POSTOP_FLUX_BASE', ROOT.parent/'models/FLUX.2-klein-base-4B'))
DT=torch.bfloat16


def pipe():
    p=Flux2KleinPipeline.from_pretrained(BASE,torch_dtype=DT,local_files_only=True).to('cuda')
    p.set_progress_bar_config(disable=True)
    return p


def manifest(split):
    return [r for r in json.loads((DATA/'manifest.json').read_text()) if r['split']==split and r.get('aligned')]


def train(p,steps):
    random.seed(107)
    torch.manual_seed(107)
    tr=p.transformer
    tr.requires_grad_(False)
    tr.add_adapter(LoraConfig(r=8,lora_alpha=8,target_modules=['to_q','to_v','to_qkv_mlp_proj']))
    params=[v for v in tr.parameters() if v.requires_grad]
    for v in params: v.data=v.data.float()
    opt=torch.optim.AdamW(params,lr=7e-5)
    cache={cat:[] for cat in REGIONS}
    rows=manifest('train')
    with torch.no_grad():
        prompts={c:p.encode_prompt(PROMPTS[c],max_sequence_length=128) for c in REGIONS}
        for r in rows:
            d=DATA/r['category']/str(r['case'])
            def encode(name):
                with Image.open(d/name) as im:
                    tensor=p.image_processor.preprocess(im).to('cuda',DT)
                return p._encode_vae_image(tensor,None)
            src,target=encode('pre.png'),encode('target.png')
            h,w=target.shape[-2:]
            mask=cv2.resize(cv2.imread(str(d/'mask.png'),0),(w,h),interpolation=cv2.INTER_AREA)/255
            weight=torch.tensor(.15+mask,device='cuda',dtype=torch.float32).reshape(1,-1,1)
            cache[r['category']].append((p._pack_latents(src),p._pack_latents(target),weight))
        ids=torch.cat([p._prepare_latent_ids(target),p._prepare_image_ids([src])],dim=1).to('cuda')
    for items in cache.values(): random.shuffle(items)
    ck=CKPT
    ck.mkdir(exist_ok=True,parents=True)
    log=[]
    start=time.monotonic()
    tr.train()
    for step in range(steps):
        # One patient per category per update. Shuffle each category between cycles.
        opt.zero_grad()
        losses=[]
        for cat,items in cache.items():
            idx=step%len(items)
            if idx==0 and step: random.shuffle(items)
            cond,z,w=items[idx]
            emb,txt=prompts[cat]
            eps=torch.randn_like(z)
            sigma=torch.randn(1,device='cuda').sigmoid()
            noisy=((1-sigma)*z+sigma*eps).to(DT)
            with torch.autocast('cuda',dtype=DT):
                v=tr(hidden_states=torch.cat([noisy,cond],dim=1),timestep=sigma.to(DT),guidance=None,
                     encoder_hidden_states=emb,txt_ids=txt,img_ids=ids,return_dict=False)[0][:,:z.shape[1]]
                loss=(((v.float()-(eps.float()-z.float())).square().mean(-1,keepdim=True)*w).sum()/w.sum())
            if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
            (loss/len(cache)).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(params,1.)
        opt.step()
        log.append(float(np.mean(losses)))
        if (step+1)%10==0: print(step+1,log[-1],flush=True)
    tr.save_lora_adapter(ck,weight_name='adapter.safetensors')
    (ck/'training.json').write_text(json.dumps(dict(steps=steps,loss=log,seconds=time.monotonic()-start,
        counts={k:len(v) for k,v in cache.items()},parameters=sum(v.numel() for v in params),
        peak_gib=torch.cuda.max_memory_allocated()/2**30,seed=107,base=str(BASE)),indent=2))


def render(p,im,landmarks,cat):
    mat,mask=crop_geometry(landmarks,cat)
    source=crop(im,mat)
    result=p(image=Image.fromarray(source),prompt=PROMPTS[cat],height=SIZE,width=SIZE,num_inference_steps=20,
             guidance_scale=4,max_sequence_length=128,generator=torch.Generator('cuda').manual_seed(107)).images[0]
    delta=np.asarray(result).astype('float32')-source
    alpha=mask.astype('float32')/255
    inv=cv2.invertAffineTransform(mat)
    dims=(im.shape[1],im.shape[0])
    correction=cv2.warpAffine(delta*alpha[:,:,None],inv,dims)
    out=np.uint8(np.clip(np.rint(im.astype('float32')+correction),0,255))
    support=cv2.warpAffine(alpha,inv,dims)>0
    assert np.array_equal(out[~support],im[~support])
    return out


def evaluate(p,split,frozen,limit):
    rows=manifest(split)
    folder=OUT/('frozen' if frozen else 'trained')/split
    folder.mkdir(exist_ok=True,parents=True)
    panels=[]
    scores=[]
    with detector() as det:
        for cat in REGIONS:
            subset=[r for r in rows if r['category']==cat]
            for r in subset[:limit]:
                im=read(ROOT/r['source'])
                lm=points(im,det)
                out=render(p,im,lm,cat)
                dest=folder/f"{cat}_{r['case']}"
                dest.mkdir(exist_ok=True)
                Image.fromarray(out).save(dest/'simulation.png')
                mat,mask=crop_geometry(lm,cat)
                d=DATA/cat/str(r['case'])
                before,pred,target=crop(im,mat),crop(out,mat),read(d/'target.png')
                region=mask>128
                def error(a,b):
                    return float(abs(cv2.GaussianBlur(a.astype('float32'),(0,0),3)-cv2.GaussianBlur(b.astype('float32'),(0,0),3))[region].mean())
                scores.append(dict(category=cat,case=r['case'],unchanged=error(before,target),
                                   edited=error(pred,target),edit_size=error(pred,before)))
                panel=Image.new('RGB',(720,268),'white')
                draw=ImageDraw.Draw(panel)
                for j,(label,pic) in enumerate(zip(['Before','Prediction','Observed follow-up'],[before,pred,read(ROOT/r['followup'])])):
                    draw.text((j*240+4,4),f"{cat} {r['case']} {label}",fill='black')
                    if j==2:
                        # Display the actual follow-up in its own landmark-based crop.
                        m,_=crop_geometry(points(pic,det),cat)
                        pic=crop(pic,m)
                    panel.paste(ImageOps.contain(Image.fromarray(pic),(240,240)),(j*240,24))
                panel.save(dest/'comparison.jpg',quality=94)
                panels.append(panel)
    (folder/'metrics.json').write_text(json.dumps(scores,indent=2))
    sheet=Image.new('RGB',(1440,268*((len(panels)+1)//2)),'white')
    for i,panel in enumerate(panels): sheet.paste(panel,(720*(i%2),268*(i//2)))
    sheet.save(folder/'gallery.jpg',quality=92)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    sub=a.add_subparsers(dest='cmd',required=True)
    t=sub.add_parser('train')
    t.add_argument('--steps',type=int,default=110)
    e=sub.add_parser('eval')
    e.add_argument('--split',choices=['val','test'],default='val')
    e.add_argument('--limit',type=int,default=2)
    e.add_argument('--frozen',action='store_true')
    i=sub.add_parser('infer')
    i.add_argument('source')
    i.add_argument('category',choices=list(REGIONS))
    i.add_argument('output')
    args=a.parse_args()
    p=pipe()
    if args.cmd=='train':
        train(p,args.steps)
    else:
        if not getattr(args,'frozen',False):
            p.transformer.load_lora_adapter(CKPT,weight_name='adapter.safetensors',prefix=None)
        if args.cmd=='eval': evaluate(p,args.split,args.frozen,args.limit)
        else:
            im=read(args.source)
            with detector() as det: out=render(p,im,points(im,det),args.category)
            Image.fromarray(out).save(args.output,format='PNG')
