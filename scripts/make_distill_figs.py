"""Figures + GIF for the MoE->dense router-swap / reconstruction-pretraining writeup.
Grounded in the real V1 reconstruction metrics and the densify_layer source.
"""
import json, os, random
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"figure.dpi":140,"font.family":"DejaVu Sans"})

REPO="/home/ubuntu/laguna-xs2-expert-coactivation-scheduling"
OUT=f"{REPO}/docs/reports/figures"
os.makedirs(OUT,exist_ok=True)
random.seed(0); np.random.seed(0)

# ---- load real metrics ----
recs=[json.loads(l) for l in open(f"{REPO}/runs/reconstruction/opencode_v1_5k/metrics.jsonl")]
steps=np.array([r["step"] for r in recs])
loss=np.array([r["loss"] for r in recs])
L=39
M=np.array([[recs[i]["per_layer"][str(l)]["mse"] for l in range(1,L+1)] for i in range(len(recs))])  # [T,39]
COS=np.array([[recs[i]["per_layer"][str(l)]["cosine_loss"] for l in range(1,L+1)] for i in range(len(recs))])
deep=M[:,27:].mean(1); mid=M[:,10:27].mean(1); shallow=M[:,:10].mean(1)
print(f"loaded {len(recs)} steps; loss {loss[0]:.3f}->{loss[-1]:.3f}; deep MSE {deep[0]:.3f}->{deep[-1]:.3f}")

# real C4 expert load for routing flicker
load=np.load("/home/ubuntu/expert_stats.npz")["counts"].sum(0); p=load/load.sum()

# =========================================================
# FIG 1 — router swap schematic (MoE block  ->  dense FFN)
# =========================================================
fig,ax=plt.subplots(figsize=(11,5.2)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,100)
def box(x,y,w,h,txt,fc,ec="#333",fs=10,tc="#111"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.6,rounding_size=2",fc=fc,ec=ec,lw=1.4))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,fontweight="bold")
# LEFT: MoE
ax.text(20,95,"MoE block (teacher)",ha="center",fontsize=12,fontweight="bold")
box(4,74,32,9,"sigmoid router  +  e_score_bias\ntop-8 of 256, renorm",  "#fde68a",fs=9)
# 256 expert minigrid
gx,gy=6,30;
for i in range(256):
    r,c=i//32,i%32
    lit = i in [216,114,204,133,183,214,179,161]  # busiest (real)
    ax.add_patch(plt.Rectangle((gx+c*0.9, gy+ (7-r)*4.2 +1.5), 0.8, 3.6,
                 fc=("#ef4444" if lit else "#dbeafe"), ec="white", lw=0.2))
ax.text(20,29,"256 routed experts  (8 active / token, sparse)",ha="center",fontsize=9)
box(4,18,32,8,"shared expert (always on)","#bbf7d0",fs=9)
# RIGHT: Dense
ax.text(80,95,"Dense block (student)",ha="center",fontsize=12,fontweight="bold")
box(64,52,32,30,"Dense SwiGLU FFN\n\nintermediate = 4096\n(= K8 × 512)\n\ngate | up | down","#c7d2fe",fs=11)
box(64,18,32,8,"shared expert (kept verbatim)","#bbf7d0",fs=9)
# arrow + transform label
ax.add_patch(FancyArrowPatch((37,50),(63,60),arrowstyle="-|>",mutation_scale=22,lw=2.2,color="#111"))
ax.text(50,70,"swap the router out",ha="center",fontsize=11,fontweight="bold",color="#b91c1c")
ax.text(50,64,"DO-ACP select K=8 of 256\nconcat gate/up · Σ→down\nfold ×2.5 routed-scale · α",
        ha="center",fontsize=8.5,color="#374151")
ax.text(50,8,"no router · all-active · one matmul instead of a 256-way gather",
        ha="center",fontsize=9,style="italic",color="#555")
fig.tight_layout(); fig.savefig(f"{OUT}/router_swap.png"); plt.close(fig)

# =========================================================
# FIG 2 — reconstruction loss curves (real)
# =========================================================
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
axes[0].plot(steps,loss,color="#4f46e5",lw=2)
axes[0].set_title("Total reconstruction loss",fontweight="bold")
axes[0].set_xlabel("step"); axes[0].set_ylabel("mean_l(MSE/‖y‖² + 0.05·(1−cos))"); axes[0].grid(alpha=.25)
axes[0].annotate(f"{loss[0]:.2f} → {loss[-1]:.2f}",(steps[-1],loss[-1]),xytext=(-90,30),
                 textcoords="offset points",fontsize=9,fontweight="bold",
                 arrowprops=dict(arrowstyle="->",color="#777"))
axes[1].plot(steps,deep,color="#dc2626",lw=2,label="deep  L28–39")
axes[1].plot(steps,mid,color="#f59e0b",lw=1.6,label="mid   L11–27")
axes[1].plot(steps,shallow,color="#10b981",lw=1.6,label="shallow L1–10")
axes[1].set_yscale("log"); axes[1].set_title("Per-layer-group raw MSE (log)",fontweight="bold")
axes[1].set_xlabel("step"); axes[1].set_ylabel("MSE"); axes[1].grid(alpha=.25,which="both"); axes[1].legend()
axes[1].text(0.98,0.95,"deep layers carry the big magnitudes\n→ biggest drop",transform=axes[1].transAxes,
             ha="right",va="top",fontsize=8.5,color="#555",style="italic")
fig.tight_layout(); fig.savefig(f"{OUT}/recon_loss_curves.png"); plt.close(fig)

# =========================================================
# FIG 3 — per-layer MSE heatmap (layer x step), real
# =========================================================
fig,ax=plt.subplots(figsize=(11,4.4))
H=M.T.clip(1e-5,None)  # [39, T]
im=ax.imshow(H,aspect="auto",origin="lower",cmap="inferno",norm=LogNorm(vmin=1e-4,vmax=H.max()),
             extent=[steps[0],steps[-1],1,L])
ax.set_xlabel("training step"); ax.set_ylabel("layer (1=shallow → 39=deep)")
ax.set_title("Reconstruction MSE per layer over training — deep layers collapse first",fontweight="bold")
fig.colorbar(im,fraction=0.025,pad=0.02,label="MSE (log)")
fig.tight_layout(); fig.savefig(f"{OUT}/recon_mse_heatmap.png"); plt.close(fig)

# =========================================================
# GIF — router swap: sparse MoE routing (flickers) vs dense student reconstructing (MSE collapses)
# =========================================================
nf=48
sidx=np.linspace(0,len(recs)-1,nf).astype(int)
fig=plt.figure(figsize=(9,5.0))
gL=fig.add_axes([0.04,0.12,0.40,0.74]); gR=fig.add_axes([0.54,0.12,0.43,0.74])
ymax=M.max()*1.1
def draw(fi):
    si=sidx[fi]; gL.clear(); gR.clear()
    # LEFT: teacher MoE routing — 16x16, 8 experts lit (sampled by real load) → sparse flicker
    chosen=np.random.choice(256,size=8,replace=False,p=p)
    grid=np.zeros(256); grid[chosen]=1.0
    gL.imshow(grid.reshape(16,16),cmap="Reds",vmin=0,vmax=1.4)
    gL.set_xticks([]); gL.set_yticks([])
    gL.set_title("MoE teacher\n8 of 256 experts / token (sparse, flickers)",fontsize=10,fontweight="bold")
    # RIGHT: student per-layer MSE bars collapsing (real)
    col=plt.cm.viridis(np.linspace(0,1,L))
    gR.bar(range(1,L+1),M[si],color=col)
    gR.set_ylim(1e-4,ymax); gR.set_yscale("log")
    gR.set_xlabel("layer (1→39)"); gR.set_ylabel("reconstruction MSE (log)")
    gR.set_title("Dense student\nreconstructs each layer's output",fontsize=10,fontweight="bold")
    gR.grid(alpha=.2,which="both",axis="y")
    fig.suptitle(f"Swapping routers out → distillation by feature reconstruction   ·   step {steps[si]:>4d}   ·   loss {loss[si]:.3f}",
                 fontsize=11,fontweight="bold",y=0.99)
anim=animation.FuncAnimation(fig,draw,frames=nf,interval=110)
anim.save(f"{OUT}/distillation.gif",writer=animation.PillowWriter(fps=10))
plt.close(fig)
print("wrote: router_swap.png, recon_loss_curves.png, recon_mse_heatmap.png, distillation.gif")
for f in ["router_swap.png","recon_loss_curves.png","recon_mse_heatmap.png","distillation.gif"]:
    print(" ",f,os.path.getsize(f"{OUT}/{f}")//1024,"KB")
