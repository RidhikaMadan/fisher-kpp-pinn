import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

torch.manual_seed(0)
np.random.seed(0)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = 0.01
r = 1.0
xmin, xmax = -10.0, 10.0
tmin, tmax = 0.0, 5.0

# Analytical Solution

def analytic(x, t, D=D, r=r):
    lam = np.sqrt(r / (4 * D))
    speed = 2 * np.sqrt(D * r)
    return 1.0 / (1.0 + np.exp(lam * (x - speed * t))) ** 2


def rel_err(pred, ref):
    return np.linalg.norm(pred - ref) / np.linalg.norm(ref)


Nx, Nt = 200, 1000
xs = np.linspace(xmin, xmax, Nx)
ts = np.linspace(tmin, tmax, Nt)
dx = xs[1] - xs[0]
dt = ts[1] - ts[0]


# Finite difference solver
def run_fd():
    u = np.zeros((Nt, Nx))
    u[0] = analytic(xs, 0.0)

    for n in range(Nt - 1):
        un = u[n]
        lap = np.zeros(Nx)
        lap[1:-1] = (un[2:] - 2*un[1:-1] + un[:-2]) / dx**2

        u[n+1, 1:-1] = un[1:-1] + dt*D*lap[1:-1] + dt*r*un[1:-1]*(1 - un[1:-1])
        u[n+1, 0]  = analytic(xmin, ts[n])
        u[n+1, -1] = analytic(xmax, ts[n])
        u[n+1] = np.clip(u[n+1], 0.0, 1.0)

    return u

u_fd = run_fd()


# Forward net
class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def residual(net, x, t, D_val, r_val):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = net(x, t)
    ut  = torch.autograd.grad(u, t,  torch.ones_like(u), create_graph=True)[0]
    ux  = torch.autograd.grad(u, x,  torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    return ut - D_val*uxx - r_val*u*(1 - u)


# Training data
N_PDE = 8000

def sample_pts():
    # IC points
    xi = torch.FloatTensor(200, 1).uniform_(xmin, xmax)
    ti = torch.zeros(200, 1)
    ui = torch.FloatTensor(analytic(xi.numpy(), 0.0)).reshape(-1, 1)

    # BC points
    tL = torch.FloatTensor(100, 1).uniform_(tmin, tmax)
    tR = torch.FloatTensor(100, 1).uniform_(tmin, tmax)
    uL = torch.FloatTensor(analytic(xmin, tL.numpy())).reshape(-1, 1)
    uR = torch.FloatTensor(analytic(xmax, tR.numpy())).reshape(-1, 1)
    xbc = torch.cat([torch.full((100,1), xmin), torch.full((100,1), xmax)])
    tbc = torch.cat([tL, tR])
    ubc = torch.cat([uL, uR])

    # Collocation points
    xc = torch.FloatTensor(N_PDE, 1).uniform_(xmin, xmax)
    tc = torch.FloatTensor(N_PDE, 1).uniform_(tmin, tmax)

    def to_dev(*args):
        return [a.to(dev) for a in args]

    return to_dev(xi, ti, ui, xbc, tbc, ubc, xc, tc)


# Training for forward problem
def train_forward():
    net = MLP().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.5)
    xi, ti, ui, xbc, tbc, ubc, xc, tc = sample_pts()

    log = []
    for ep in range(6000):
        opt.zero_grad()
        lic  = torch.mean((net(xi, ti) - ui) ** 2)
        lbc  = torch.mean((net(xbc, tbc) - ubc) ** 2)
        lpde = torch.mean(residual(net, xc, tc, D, r) ** 2)
        loss = lic + lbc + 0.1*lpde
        loss.backward()
        opt.step()
        sched.step()
        log.append((loss.item(), lic.item(), lbc.item(), lpde.item()))

    print(f"Final loss for forward network: {log[-1][0]:.2e}")
    return net, log


net_fwd, fwd_log = train_forward()


def predict(net, t_val):
    net.eval()
    with torch.no_grad():
        xin = torch.FloatTensor(xs).reshape(-1, 1).to(dev)
        tin = torch.full((Nx, 1), t_val).to(dev)
        return net(xin, tin).cpu().numpy().flatten()


snaps = [0.0, 1.25, 2.5, 3.75, 5.0]
fwd_preds = {t: predict(net_fwd, t) for t in snaps}

print("Relative L2 errors for forward network:")
for t in snaps:
    print(f"  t={t:.2f}  {rel_err(fwd_preds[t], analytic(xs, t)):.2e}")


# Inverse net
class InvMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.log_D = nn.Parameter(torch.tensor(np.log(0.05)))
        self.log_r = nn.Parameter(torch.tensor(np.log(0.5)))

    @property
    def D_est(self): return torch.exp(self.log_D)

    @property
    def r_est(self): return torch.exp(self.log_r)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def run_inverse(noise=0.0, n_obs=500, n_ep=10000):
    n_front = n_obs // 2
    n_uni   = n_obs - n_front

    ti_u = np.random.randint(0, Nt, n_uni)
    xi_u = np.random.randint(0, Nx, n_uni)
    ti_f = np.random.randint(0, Nt, n_front)
    xi_f = np.random.randint(0, Nx, n_front)

    ti_all = np.concatenate([ti_u, ti_f])
    xi_all = np.concatenate([xi_u, xi_f])

    u_meas = u_fd[ti_all, xi_all] + noise * np.random.randn(n_obs)
    u_meas = np.clip(u_meas, 0, 1)

    xobs = torch.FloatTensor(xs[xi_all]).reshape(-1, 1).to(dev)
    tobs = torch.FloatTensor(ts[ti_all]).reshape(-1, 1).to(dev)
    uobs = torch.FloatTensor(u_meas).reshape(-1, 1).to(dev)

    xc = torch.FloatTensor(8000, 1).uniform_(xmin, xmax).to(dev)
    tc = torch.FloatTensor(8000, 1).uniform_(tmin, tmax).to(dev)

    model = InvMLP().to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2500, gamma=0.5)

    D_hist, r_hist = [], []
    for _ in range(n_ep):
        opt.zero_grad()
        l_data = torch.mean((model(xobs, tobs) - uobs) ** 2)
        l_pde  = torch.mean(residual(model, xc, tc, model.D_est, model.r_est) ** 2)
        (l_data + 0.1*l_pde).backward()
        opt.step()
        sched.step()
        D_hist.append(model.D_est.item())
        r_hist.append(model.r_est.item())

    return model.D_est.item(), model.r_est.item(), D_hist, r_hist


noise_levels = [0.00, 0.01, 0.05, 0.10]
inv_results = []
for sig in noise_levels:
    D_rec, r_rec, Dh, rh = run_inverse(noise=sig)
    eD = abs(D_rec - D) / D * 100
    er = abs(r_rec - r) / r * 100
    inv_results.append((sig, D_rec, r_rec, eD, er, Dh, rh))
    print("\nResults for inverse problem")
    print(f"  noise={sig:.2f}  D={D_rec:.4f} ({eD:.1f}%)  r={r_rec:.4f} ({er:.1f}%)")


# Plots
cmap = plt.cm.viridis
cols = cmap(np.linspace(0.1, 0.9, len(snaps)))

def style(ax):
    ax.set_facecolor("#f9f9f7")
    ax.spines[["top", "right"]].set_visible(False)

# Exact solution vs PINN
fig1, ax = plt.subplots(figsize=(6, 4))
fig1.patch.set_facecolor("#f9f9f7")
style(ax)
for i, t in enumerate(snaps):
    ax.plot(xs, analytic(xs, t), "--", color=cols[i], lw=1.2, alpha=0.6)
    ax.plot(xs, fwd_preds[t],    "-",  color=cols[i], lw=2.0, label=f"t={t:.2f}")
ax.set_title("PINN vs Exact Solution (- exact,  -- PINN)", fontsize=10)
ax.set_xlabel("x");  ax.set_ylabel("u")
ax.legend(fontsize=7, ncol=2)
fig1.tight_layout()
plt.savefig("exact_vs_forward_PINN.png", dpi=150, bbox_inches="tight")
plt.show()

# Training loss evolution
fig2, ax = plt.subplots(figsize=(6, 4))
fig2.patch.set_facecolor("#f9f9f7")
style(ax)
tots = [h[0] for h in fwd_log]
ax.semilogy(tots, color="#3a7abf", lw=1.5)
ax.set_title("training loss (forward)", fontsize=10)
ax.set_xlabel("epoch");  ax.set_ylabel("loss")
ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all"))
fig2.tight_layout()
plt.savefig("training_loss_evolution.png", dpi=150, bbox_inches="tight")
plt.show()

# L2 error evolution for the forward solution
fig3, ax = plt.subplots(figsize=(6, 4))
fig3.patch.set_facecolor("#f9f9f7")
style(ax)
errs = [rel_err(fwd_preds[t], analytic(xs, t)) for t in snaps]
ax.plot(snaps, errs, "o-", color="#c0522b", lw=1.5, ms=5)
ax.set_title("relative L2 error vs. time", fontsize=10)
ax.set_xlabel("t");  ax.set_ylabel("rel. L2")
fig3.tight_layout()
plt.savefig("forward_L2_error.png", dpi=150, bbox_inches="tight")
plt.show()

# Estimates of parameters D,r against epochs completed
fig4, ax = plt.subplots(figsize=(6, 4))
fig4.patch.set_facecolor("#f9f9f7")
style(ax)
Dh = inv_results[0][5]
rh = inv_results[0][6]
ax.plot(Dh, lw=1.4, label=r"$D$ (true = 0.01)")
ax.plot(rh, lw=1.4, label=r"$r$ (true = 1.0)")
ax.axhline(D, ls="--", lw=0.8, color="C0", alpha=0.5)
ax.axhline(r, ls="--", lw=0.8, color="C1", alpha=0.5)
ax.set_title("Inverse PINN Parameter Estimates for σ=0", fontsize=10)
ax.set_xlabel("epoch");  ax.legend(fontsize=8)
fig4.tight_layout()
plt.savefig("inv_PINN_estimates.png", dpi=150, bbox_inches="tight")
plt.show()