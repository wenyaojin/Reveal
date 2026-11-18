"""Make reproducibly shuffled source-versus-output trials for independent reviewers."""
import csv
import json
import random
from PIL import Image,ImageDraw,ImageOps
from hda_joint_data import ROOT, DATA, OUT, REGIONS, read

if __name__=='__main__':
    dest=OUT/'blind_review'
    dest.mkdir(exist_ok=True)
    rows=json.loads((DATA/'manifest.json').read_text())
    rng=random.Random(314)
    trials=[]
    for cat in REGIONS:
        pool=[r for r in rows if r['category']==cat and r['split']=='test' and r.get('aligned')]
        for r in rng.sample(pool,3):
            for method in ('trained','shape'):
                trials.append((r,method))
        trials.append((pool[0],'unchanged'))
    rng.shuffle(trials)
    answers=[]
    for n,(r,method) in enumerate(trials,1):
        cat=r['category']
        src=ROOT/r['source']
        pred=src if method=='unchanged' else OUT/method/'test'/f"{cat}_{r['case']}"/'simulation.png'
        swap=rng.choice([True,False])
        panel=Image.new('RGB',(880,520),'white')
        draw=ImageDraw.Draw(panel)
        for i,path in enumerate([pred,src] if swap else [src,pred]):
            im=ImageOps.contain(Image.fromarray(read(path)),(435,480))
            draw.text((440*i+8,4),f"Trial {n:02} | {cat} | {'AB'[i]}",fill='black')
            panel.paste(im,(440*i+(440-im.width)//2,30))
        panel.save(dest/f'{n:02}.png')
        answers.append([n,cat,r['case'],method,('same' if method=='unchanged' else 'A' if swap else 'B')])
    for filename,header,body in [('answer_key.csv',['trial','category','case','method','edited_side'],answers),
            ('ratings_blank.csv',['trial','visibly_edited_A_B_or_same','looks_natural_A_B_both_neither','same_identity_yes_no','artifact_location'],[[a[0],'','','',''] for a in answers])]:
        with (dest/filename).open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(header)
            writer.writerows(body)
    (dest/'README.md').write_text('每位评审独立查看 01–35.png，在 ratings_blank.csv 的个人副本记录变化侧、自然的一侧、是否同人、伪影部位；看不出变化可填 same。提交后再查看 answer_key.csv。35 组含每类 3 位病例的两种模型及每类一组相同原图。此题测可见性和自然度，不评诊疗效果。\n')
