#!/usr/bin/env python3
"""KID + IC-LPIPS for ZJU fabric generated sets (adapted from tools/eval_t1_generation_quality.py)."""
from __future__ import annotations
import argparse, csv, glob, json, math, os, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image

IMG_EXTS = ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.JPEG","*.PNG")

def list_images(d):
    fs=[]
    for pat in IMG_EXTS: fs += glob.glob(os.path.join(d,pat))
    return sorted(set(fs))

def load_u8(path,size=299):
    im=Image.open(path).convert("RGB").resize((size,size),Image.BILINEAR)
    return torch.from_numpy(np.asarray(im,dtype=np.uint8)).permute(2,0,1).contiguous()

def stack(paths,size,cap,seed):
    if cap and len(paths)>cap: paths=random.Random(seed).sample(paths,cap)
    return torch.stack([load_u8(p,size) for p in paths]) if paths else torch.empty(0,3,size,size,dtype=torch.uint8)

def lpips_pre():
    from torchvision import transforms
    return transforms.Compose([transforms.Resize([256,256]),transforms.ToTensor(),
        transforms.Normalize(mean=[.5]*3,std=[.5]*3)])

def load_lpips(paths,pre,dev):
    return torch.stack([pre(Image.open(p).convert("RGB")) for p in paths]).to(dev)

@torch.no_grad()
def ic_lpips(gen_paths,support_paths,lpips_fn,dev,cap=50,seed=0):
    if len(gen_paths)<2 or not support_paths: return None,{}
    pre=lpips_pre(); clusters=[[] for _ in support_paths]
    support=load_lpips(support_paths,pre,dev)
    for start in range(0,len(gen_paths),16):
        batch=gen_paths[start:start+16]; img=load_lpips(batch,pre,dev)
        # pairwise LPIPS query x support
        q=img.shape[0]; s=len(support_paths)
        lhs=img.repeat_interleave(s,0); rhs=support.repeat(q,1,1,1)
        d=lpips_fn(lhs,rhs).reshape(q,s).cpu()
        for path,k in zip(batch,d.argmin(1).tolist()): clusters[k].append(path)
    rng=random.Random(seed); scores=[]; used=0; sizes=[]
    for c in clusters:
        sel=list(c); rng.shuffle(sel); sel=sel[:cap]; sizes.append(len(sel))
        if len(sel)<2: continue
        used+=len(sel); vals=[]
        # one reference image vs a batch of later images; LPIPS needs equal batch sizes
        for i in range(len(sel)-1):
            a=load_lpips([sel[i]],pre,dev)
            for j in range(i+1,len(sel),16):
                right=sel[j:j+16]
                b=load_lpips(right,pre,dev)
                ar=a.expand(b.shape[0],-1,-1,-1)
                vals += lpips_fn(ar,b).reshape(-1).cpu().tolist()
        if vals: scores.append(float(np.mean(vals)))
    return (float(np.mean(scores)) if scores else None), {
        "centers":len(support_paths),"nonempty_clusters":len(scores),"images_used":used,"cluster_sizes":sizes}

def kid(real_u8,gen_u8,device,subset=50,subsets=100,seed=0):
    from torchmetrics.image.kid import KernelInceptionDistance
    ss=min(subset,real_u8.shape[0],gen_u8.shape[0])
    if ss<2: return None,None
    m=KernelInceptionDistance(feature=2048,subset_size=ss,subsets=subsets).to(device)
    for i in range(0,len(real_u8),32): m.update(real_u8[i:i+32].to(device),real=True)
    for i in range(0,len(gen_u8),32): m.update(gen_u8[i:i+32].to(device),real=False)
    with torch.random.fork_rng(devices=[device.index] if device.type=="cuda" and device.index is not None else []):
        torch.manual_seed(seed)
        if device.type=="cuda": torch.cuda.manual_seed_all(seed)
        mean,std=m.compute()
    return float(mean),float(std)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--real_root",default="/home/cc/Documents/anomalydiffusion/anomalydiffusion/zju_leaper_native512/fabric/test")
    ap.add_argument("--gen_root",default="/media/cc/U2-Server/crisp-storage/b300-repo/datasets/generated/zju512_sdxl_10k_native512/fabric")
    ap.add_argument("--patterns",default="pattern16,pattern17,pattern18,pattern19")
    ap.add_argument("--n_gen",type=int,default=500)
    ap.add_argument("--kid_subset",type=int,default=50)
    ap.add_argument("--kid_subsets",type=int,default=100)
    ap.add_argument("--ic_cap",type=int,default=50)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--out_csv",default="explore/kid_iclpips_sdxl10k.csv")
    a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev=torch.device(a.device if torch.cuda.is_available() else "cpu")
    import lpips
    lpips_fn=lpips.LPIPS(net="vgg").to(dev).eval()
    rows=[]
    for pat in [x.strip() for x in a.patterns.split(",") if x.strip()]:
        real=list_images(os.path.join(a.real_root,pat))
        gen=list_images(os.path.join(a.gen_root,pat,"image"))
        if not gen: print("skip",pat,"no generated"); continue
        gen=random.Random(a.seed).sample(gen,min(a.n_gen,len(gen)))
        real_u8=stack(real,299,0,a.seed); gen_u8=stack(gen,299,0,a.seed)
        km,ks=kid(real_u8,gen_u8,dev,a.kid_subset,a.kid_subsets,a.seed)
        support=real[:max(1,len(real)//3)]
        ic,meta=ic_lpips(gen,support,lpips_fn,dev,a.ic_cap,a.seed)
        rows.append({"pattern":pat,"n_real":len(real),"n_gen":len(gen),
                     "KID_x1e3":km*1e3 if km is not None else "","KID_std_x1e3":ks*1e3 if ks is not None else "",
                     "IC_LPIPS":ic if ic is not None else "",**meta})
        print(rows[-1],flush=True)
    if rows:
        keys=[]
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
        out=Path(a.out_csv); out.parent.mkdir(parents=True,exist_ok=True)
        with open(out,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
        kid_vals=[r["KID_x1e3"] for r in rows if isinstance(r["KID_x1e3"],float)]
        ic_vals=[r["IC_LPIPS"] for r in rows if isinstance(r["IC_LPIPS"],float)]
        print("mean KIDx1e3",float(np.mean(kid_vals)),"mean IC-LPIPS",float(np.mean(ic_vals)),"wrote",out)
if __name__=="__main__": main()
