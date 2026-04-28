import torch
import torch.nn as nn
import numpy as np
import skfmm
import matplotlib.pyplot as plt
import time
from scipy.interpolate import griddata
import skimage.filters
from tqdm import tqdm
from scipy.ndimage import median_filter
import math

# --- 1. 环境、数据加载和参数设置 ---
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# --- 参数设置 ---
N_SOURCES = 8
N_SURFACE_RECEIVERS_PER_SOURCE = 128
N_COLLOCATION = 4096
EPOCHS = 30000
GAUSSIAN_SIGMA = 1.0

# --- 损失权重 ---
LAMBDA_DATA = 10.0
LAMBDA_PHYSICS = 10.0

print("步骤 1: 从 model10.npy 加载速度模型...")
try:
    fwi_data = np.load('model10.npy')
except FileNotFoundError:
    print("错误: model10.npy 未找到! 创建虚拟文件...")
    fwi_data = np.ones((70, 70)) * 2000
    for i in range(70): fwi_data[i, :] = 1500 + i * 30

if fwi_data.ndim == 4:
    vp_true_ms = fwi_data[0, 0, :, :].T  # 加上 .T
elif fwi_data.ndim == 2:
    vp_true_ms = fwi_data.T              # 加上 .T

print(f"创建合成Vs模型并应用高斯平滑 (Sigma={GAUSSIAN_SIGMA})...")
vs_true_ms = vp_true_ms / 1.73
vp_true_ms = skimage.filters.gaussian(vp_true_ms, GAUSSIAN_SIGMA)
vs_true_ms = skimage.filters.gaussian(vs_true_ms, GAUSSIAN_SIGMA)

NX, NZ = vp_true_ms.shape
print(f"模型加载完毕. 网格维度: {NX} x {NZ} (X, Z)")

vp_true_kms = vp_true_ms / 1000.0
vs_true_kms = vs_true_ms / 1000.0
VP_MIN, VP_MAX = np.min(vp_true_kms), np.max(vp_true_kms)
VS_MIN, VS_MAX = np.min(vs_true_kms), np.max(vs_true_kms)

DX, DZ = 0.01, 0.01  # km
X = np.arange(0, NX * DX, DX)
Z = np.arange(0, NZ * DZ, DZ)
X_MAX, Z_MAX = X.max(), Z.max()
WELL_X_POSITIONS = [0.0, X_MAX]

# ==========================================================
# 数据生成
# ==========================================================
def generate_travel_time_data(velocity_model_kms, num_sources, num_receivers):
    print(f"步骤 2: 生成走时数据...")
    src_x = np.linspace(X.min() + DX, X.max() - DX, num_sources)
    src_locs = np.array([[x, Z.min()] for x in src_x])
    rec_x = np.linspace(X.min(), X.max(), num_receivers)
    all_pairs, all_times = [], []
    coords_x, coords_z = np.meshgrid(X, Z, indexing='ij')
    grid_pts = np.vstack([coords_x.ravel(), coords_z.ravel()]).T

    for src in src_locs:
        phi = -1 * np.ones_like(velocity_model_kms)
        phi[np.argmin(np.abs(X - src[0])), np.argmin(np.abs(Z - src[1]))] = 0.
        tt_field = skfmm.travel_time(phi, velocity_model_kms, dx=[DX, DZ])

        rec_locs = np.vstack((
            np.array([[x, Z.min()] for x in rec_x]),
            np.array([[X.min(), z] for z in Z]),
            np.array([[X.max(), z] for z in Z])
        ))
        times = griddata(grid_pts, tt_field.ravel(), rec_locs, method='linear')
        for r_pos, t in zip(rec_locs, times):
            if not np.isnan(t):
                all_pairs.append([r_pos[0], r_pos[1], src[0], src[1]])
                all_times.append(t)
    return np.array(all_pairs), np.array(all_times).reshape(-1, 1), src_locs

print("\n--- 生成 P 波数据 ---")
tp_coords_np, tp_times_np, src_pos_np = generate_travel_time_data(vp_true_kms, N_SOURCES,
                                                                  N_SURFACE_RECEIVERS_PER_SOURCE)
print("\n--- 生成 S 波数据 ---")
ts_coords_np, ts_times_np, _ = generate_travel_time_data(vs_true_kms, N_SOURCES, N_SURFACE_RECEIVERS_PER_SOURCE)

# 转换为 GPU 张量
tp_coords_gpu = torch.from_numpy(tp_coords_np).float().to(device)
ts_coords_gpu = torch.from_numpy(ts_coords_np).float().to(device)
tp_times_gpu = torch.from_numpy(tp_times_np).float().to(device)
ts_times_gpu = torch.from_numpy(ts_times_np).float().to(device)
src_pos_gpu = torch.from_numpy(src_pos_np).float().to(device)

norm_xy = torch.tensor([X_MAX, Z_MAX], dtype=torch.float32, device=device)
norm_4d = torch.tensor([X_MAX, Z_MAX, X_MAX, Z_MAX], dtype=torch.float32, device=device)

# ==========================================================
# 模型架构
# ==========================================================
class JointMLP(nn.Module):
    def __init__(self, in_dim, out_dim, layers, neurons, is_tau=False):
        super().__init__()
        linears = [nn.Linear(in_dim, neurons), nn.SiLU()]
        for _ in range(layers - 1): linears.extend([nn.Linear(neurons, neurons), nn.SiLU()])
        self.body = nn.Sequential(*linears)
        self.h1 = nn.Linear(neurons, out_dim)
        self.h2 = nn.Linear(neurons, out_dim)
        self.is_tau = is_tau
        self.sig = nn.Sigmoid()

    def forward(self, x):
        f = self.body(x)
        return (self.h1(f), self.h2(f)) if self.is_tau else (self.sig(self.h1(f)), self.sig(self.h2(f)))

velocity_model = JointMLP(2, 1, 8, 128, False).to(device)
travel_time_net = JointMLP(4, 1, 8, 64, True).to(device)

optimizer = torch.optim.Adam(list(velocity_model.parameters()) + list(travel_time_net.parameters()), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5000, 0.5)

vis_x, vis_z = torch.from_numpy(X).float().to(device), torch.from_numpy(Z).float().to(device)
grid_x, grid_z = torch.meshgrid(vis_x, vis_z, indexing='ij')
grid_coords_2d = torch.stack([grid_x.flatten(), grid_z.flatten()], dim=1)

# ==========================================================
# 训练循环
# ==========================================================
print("\n========== 开始最终训练 (PINN 反演，无 LSQR 约束) ==========")
start_time = time.time()

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # 1. Data Loss
    rec_norm_p = tp_coords_gpu / norm_4d
    R0_rp = torch.norm(rec_norm_p[:, :2] * norm_xy - rec_norm_p[:, 2:] * norm_xy, dim=1, keepdim=True) + 1e-8
    tau_p, _ = travel_time_net(rec_norm_p)
    loss_data_p = torch.mean((tau_p * R0_rp - tp_times_gpu) ** 2)

    rec_norm_s = ts_coords_gpu / norm_4d
    R0_rs = torch.norm(rec_norm_s[:, :2] * norm_xy - rec_norm_s[:, 2:] * norm_xy, dim=1, keepdim=True) + 1e-8
    _, tau_s = travel_time_net(rec_norm_s)
    loss_data_s = torch.mean((tau_s * R0_rs - ts_times_gpu) ** 2)
    loss_data = loss_data_p + loss_data_s

    # 2. Physics Loss (Eikonal equation)
    coll_2d = (torch.rand(N_COLLOCATION, 2, device=device) * norm_xy).requires_grad_(True)
    src_rand = src_pos_gpu[torch.randint(0, N_SOURCES, (N_COLLOCATION,), device=device)]
    coll_4d_norm = torch.cat([coll_2d, src_rand], dim=1) / norm_4d

    vp_n, vs_n = velocity_model(coll_2d / norm_xy)
    vp = VP_MIN + vp_n * (VP_MAX - VP_MIN)
    vs = VS_MIN + vs_n * (VS_MAX - VS_MIN)
    tau_pn, tau_sn = travel_time_net(coll_4d_norm)
    R0_c = torch.norm(coll_2d - src_rand, dim=1, keepdim=True) + 1e-8

    def get_grad(y, x):
        return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

    dt_p = get_grad(tau_pn, coll_2d) * R0_c + tau_pn * (coll_2d - src_rand) / R0_c
    dt_s = get_grad(tau_sn, coll_2d) * R0_c + tau_sn * (coll_2d - src_rand) / R0_c

    loss_phys = torch.mean(((dt_p ** 2).sum(1, keepdim=True) * vp ** 2 - 1) ** 2) + \
                torch.mean(((dt_s ** 2).sum(1, keepdim=True) * vs ** 2 - 1) ** 2)

    loss = LAMBDA_DATA * loss_data + LAMBDA_PHYSICS * loss_phys
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 2000 == 0 or epoch == EPOCHS - 1:
        print(
            f"Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.4e}, Data: {loss_data.item():.4e}, Phys: {loss_phys.item():.4e}")

print(f"训练完成，耗时 {time.time() - start_time:.2f} 秒")

# ==========================================================
# 结果可视化 (只展示真实模型和 PINN 反演结果)
# ==========================================================
print("生成可视化图表...")
with torch.no_grad():
    vp_n, vs_n = velocity_model(grid_coords_2d / norm_xy)
    vp_final = (VP_MIN + vp_n * (VP_MAX - VP_MIN)).cpu().numpy().reshape(NX, NZ)
    vs_final = (VS_MIN + vs_n * (VS_MAX - VS_MIN)).cpu().numpy().reshape(NX, NZ)

# 准备显示数组（转置为 (NZ, NX) 以适应 imshow 的行列顺序）
vp_true_disp = vp_true_kms.T
vs_true_disp = vs_true_kms.T
vp_pred_disp = vp_final.T
vs_pred_disp = vs_final.T

# 获取真实模型的颜色范围
vmin_vp, vmax_vp = vp_true_kms.min(), vp_true_kms.max()
vmin_vs, vmax_vs = vs_true_kms.min(), vs_true_kms.max()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Vp
im1 = axes[0, 0].imshow(vp_true_disp, origin='upper', extent=[0, X_MAX, 0, Z_MAX],
                        cmap='jet', vmin=vmin_vp, vmax=vmax_vp)
axes[0, 0].set_title("True Vp")
plt.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(vp_pred_disp, origin='upper', extent=[0, X_MAX, 0, Z_MAX],
                        cmap='jet', vmin=vmin_vp, vmax=vmax_vp)
axes[0, 1].set_title("PINN Vp")
plt.colorbar(im2, ax=axes[0, 1])

# Vs
im3 = axes[1, 0].imshow(vs_true_disp, origin='upper', extent=[0, X_MAX, 0, Z_MAX],
                        cmap='jet', vmin=vmin_vs, vmax=vmax_vs)
axes[1, 0].set_title("True Vs")
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(vs_pred_disp, origin='upper', extent=[0, X_MAX, 0, Z_MAX],
                        cmap='jet', vmin=vmin_vs, vmax=vmax_vs)
axes[1, 1].set_title("PINN Vs")
plt.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("inversion_results_no_lsqr_optimized.png", dpi=300)
plt.show()