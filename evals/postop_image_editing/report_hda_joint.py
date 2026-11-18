"""Report all paired outcomes and create a predetermined five-row full-portrait panel."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from hda_joint_data import CKPT, DATA, OUT, REGIONS, ROOT, read


def load(path):
    return json.loads(path.read_text())


def indexed(rows):
    return {(r['category'],r['case']):r for r in rows}


def make_report():
    manifest=load(DATA/'manifest.json')
    holdout=[r for r in manifest if r.get('aligned') and r['split']=='test']
    model=indexed(load(OUT/'trained/test/metrics.json'))
    frozen=indexed(load(OUT/'frozen/test/metrics.json'))
    geom=indexed(load(OUT/'mean_flow_test.json'))
    shape=indexed(load(OUT/'shape/test/metrics.json'))
    assert set(model)==set(frozen)==set(geom)==set(shape)==set(indexed(holdout))
    sheet=Image.new('RGB',(5*225,5*265),'white')
    draw=ImageDraw.Draw(sheet)
    rows=[]
    for idx,cat in enumerate(REGIONS):
        group=[r for r in holdout if r['category']==cat]
        first=group[0]
        key=f"{cat}_{first['case']}"
        for j,(path,label) in enumerate(zip([ROOT/first['source'],OUT/'frozen/test'/key/'simulation.png',
                                         OUT/'trained/test'/key/'simulation.png',OUT/'shape/test'/key/'simulation.png',ROOT/first['followup']],
                                        ['Before','Frozen base','Trained LoRA','Learned geometry','Observed follow-up'])):
            im=ImageOps.contain(Image.fromarray(read(path)),(222,239))
            draw.text((j*225+4,idx*265+3),f'{key} | {label}',fill='black')
            sheet.paste(im,(225*j+(225-im.width)//2,265*idx+21))
        metrics=[]
        for r in group:
            k=(cat,r['case'])
            f,m,g=frozen[k],model[k],geom[k]
            metrics.append([m['unchanged'],g['mean_flow'],f['edited'],m['edited'],shape[k]['edited'],m['edit_size']])
        arr=np.array(metrics)
        vals=arr.mean(0)
        rows.append([cat,len(arr),*vals.tolist(),int((arr[:,3]<arr[:,0]).sum()),int((arr[:,3]<arr[:,2]).sum())])
    sheet.save(OUT/'five_projects.jpg',quality=94)
    (OUT/'summary.json').write_text(json.dumps(dict(columns=['project','n','no_edit','mean_flow','frozen','trained','learned_geometry','edit_size','wins_vs_no_edit','wins_vs_frozen'],rows=rows),indent=2))
    info=load(CKPT/'training.json')
    md=['# HDA 五项目联合训练实测','',
        '首轮内部研究结果。总览每类取测试清单的第一例，五列依次为术前、冻结底座、共享 LoRA、学习形变模型、真实术后。五类使用同一组适配器权重。',
        '', '[五项目全脸对照](five_projects.jpg) · [全部测试病例局部对照](trained/test/gallery.jpg) · [逐例指标](trained/test/metrics.json)','',
        '|项目|例数|直接保留原图|项目平均形变|冻结底座|共享 LoRA|学习形变|LoRA 区内变化量|优于原图例数|优于底座例数|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cat,n,*v in rows:
        md.append('|'+ '|'.join([cat,str(n),*[f'{a:.2f}' for a in v[:6]],*[str(a) for a in v[6:]]])+'|')
    md+=['','误差越小越好。将结果映射回术前局部坐标，与稳定点配准、粗略颜色校正后的术后监督做比较。对 RGB 做 3 像素标准差的高斯低通，再计算掩膜核心区域的平均绝对误差。变化量使用相同算法比较输出与输入。指标受姿态、皮肤颜色、配准及照片水印影响，不能代替身份与自然度盲评。总览中的真实术后列使用原始实拍；全库局部对照中的真实术后列按自身关键点裁剪。个别裁剪边缘出现镜像延拓，完整 PNG 保留原图画幅。','',
        f"实际训练 {info['steps']} 步，每步五类各 1 个病例；LoRA rank 8，可训练参数 {info['parameters']:,}。局部输入为 384 × 384，BF16，固定种子 107。单卡更新耗时 {info['seconds']:.0f} 秒（不含预处理和潜变量缓存），训练进程峰值分配 {info['peak_gib']:.1f} GiB。另一张卡并行评测冻结底座。每类训练病例数 {info['counts']}。较大类别在本轮预算内未遍历全部病例。",'',
        '639 对中，6 对非数字编号保留待核对身份，2 对未通过稳定点配准。631 对进入训练/验证/测试（487/46/98）。跨类做了近重复聚类并保留既有眼睑划分；独立身份尚未经完整核验，98 对属于内部留出集，不能当作新机构独立患者。HDA 与相关权重限研究使用。','',
        '复现：在项目目录、mllmam 环境依次运行 scripts/hda_joint_data.py、scripts/hda_joint_lora.py train、scripts/hda_joint_mean_shape.py、scripts/hda_joint_shape.py train、scripts/hda_joint_lora.py eval --split test --limit 999（冻结对照另加 --frozen），最后运行 scripts/report_hda_joint.py。单图入口：','',
        '```bash','python scripts/hda_joint_lora.py infer HDA-PlasticSurgery/Nose/103_b.jpg Nose /tmp/nose_simulation.png','```','',
        '项目参数可选 Eyelid、Eyebrow、Nose、FacialBones、Facelift。输入照片无需掩膜或术后图。内部条件指定项目，未训练按审美偏好自动决定项目的选择器。']
    if (OUT/'landmark_test.json').exists():
        marks=load(OUT/'landmark_test.json')
        md+=['','## 结构变化与验收结论','','将真实术后的目标关键点按非目标锚点变换到术前坐标，计算目标点平均距离，以双眼外眼角间距归一化。所有结果重新检测关键点，检测成功只代表可以测量。形变模型的训练与测量使用同一关键点检测器，存在评价偏好。','',
             '|方法|成功检测|平均关键点误差（眼距百分比）|优于术前原图|','|---|---:|---:|---:|']
        for label,key in [('术前','original'),('冻结底座','frozen'),('共享 LoRA','trained'),('学习形变','shape')]:
            values=[r for r in marks if r[key] is not None]
            wins=sum(r[key]<r['original'] for r in values)
            md.append(f"|{label}|{len(values)}|{100*np.mean([r[key] for r in values]):.2f}%|{wins if key!='original' else '—'}|")
        md+=['','LoRA 在 98 例中有 54 例的局部低频像素误差低于冻结底座，12 例低于直接保留原图。部分眉部有突兀的深色眉段，部分脸缘有块状色差。以当前输出作为术后理想图尚未通过验收。下一轮优先验证低自由度形变、眼球/唇部保护区及配准质检，保留全部失败例。新配置应另设外部测试集。','',
            '单图使用形变模型可运行 python scripts/hda_joint_shape.py infer 输入图.png Nose 输出图.png。35 组左右顺序随机的匿名对照见 [评审目录](blind_review/README.md)，当前没有独立评审打分。重建关键点记录和盲评包分别使用 scripts/eval_hda_joint_landmarks.py 和 scripts/review_hda_joint.py。']
    (OUT/'README.md').write_text('\n'.join(md)+'\n')


if __name__=='__main__':
    make_report()
