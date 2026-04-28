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
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
import math
import gc

# --- 1. 环境、数据加载和参数设置 ---
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# --- 参数设置 ---
N_SOURCES = 12
N_SURFACE_RECEIVERS_PER_SOURCE = 128
N_COLLOCATION = 4096
EPOCHS = 20000
GAUSSIAN_SIGMA = 1.0

# --- 损失权重 ---
LAMBDA_DATA = 10.0
LAMBDA_PHYSICS = 10.0
LAMBDA_LSQR_INIT = 13.0   # 初始 LSQR 权重
LAMBDA_LSQR_FINAL = 3.0   # 最终 LSQR 权重
DECAY_START = 10000        # 开始衰减的 epoch

# --- LSQR 参数 ---
N_TOP_POINTS_LSQR = 100
N_SIDE_POINTS_LSQR = 100
LSQR_ITER = 200
LSQR_DAMP = 1e-2

print("步骤 1: 从 model10.npy 加载速度模型...")
try:
    fwi_data = np.load('model10.npy')
except FileNotFoundError:
    print("错误: model10.npy 未找到! 创建虚拟文件...")
    fwi_data = np.ones((70, 70)) * 2000
    for i in range(70): fwi_data[i, :] = 1500 + i * 30

if fwi_data.ndim == 4:
    vp_true_ms = fwi_data[0, 0, :, :].T
elif fwi_data.ndim == 2:
    vp_true_ms = fwi_data.T

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
# 辅助函数
# ==========================================================
def straight_ray_path_jf(sources_idx, receivers_idx, nx_grid, nz_grid, dx_grid, dz_grid):
    # 防御性类型转换：确保输入是元组
    if not isinstance(sources_idx, (tuple, list)):
        sources_idx = tuple(sources_idx)
    if not isinstance(receivers_idx, (tuple, list)):
        receivers_idx = tuple(receivers_idx)

    src_ix, src_iz = sources_idx
    rec_ix, rec_iz = receivers_idx
    x1, z1 = src_ix * dx_grid + dx_grid / 2, src_iz * dz_grid + dz_grid / 2
    x2, z2 = rec_ix * dx_grid + dx_grid / 2, rec_iz * dz_grid + dz_grid / 2
    ray_len = math.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)

    if ray_len < 1e-6:
        return [src_ix * nz_grid + src_iz], [0.0]

    dir_x, dir_z = (x2 - x1) / ray_len, (z2 - z1) / ray_len
    n_steps = int(ray_len / (min(dx_grid, dz_grid) / 4)) + 1
    segment_len = ray_len / n_steps
    cell_len_map = {}

    for step in range(n_steps):
        s = (step + 0.5) * segment_len
        curr_x = x1 + s * dir_x
        curr_z = z1 + s * dir_z
        ix = int(max(0, min(curr_x / dx_grid, nx_grid - 1)))
        iz = int(max(0, min(curr_z / dz_grid, nz_grid - 1)))
        cell_idx = ix * nz_grid + iz
        cell_len_map[cell_idx] = cell_len_map.get(cell_idx, 0.0) + segment_len

    return list(cell_len_map.keys()), list(cell_len_map.values())

def generate_unified_boundary_data(n_top, n_side_per_well):
    top_x = np.linspace(0, X_MAX, n_top)
    top_idx = [(int(max(0, min(x / DX, NX - 1))), 0) for x in top_x]
    left_z = np.linspace(0, Z_MAX, n_side_per_well)
    left_idx = [(int(max(0, min(WELL_X_POSITIONS[0] / DX, NX - 1))), int(max(0, min(z / DZ, NZ - 1)))) for z in left_z]
    right_z = np.linspace(0, Z_MAX, n_side_per_well)
    right_idx = [(int(max(0, min(WELL_X_POSITIONS[1] / DX, NX - 1))), int(max(0, min(z / DZ, NZ - 1)))) for z in right_z]
    return {'top': top_idx, 'left': left_idx, 'right': right_idx}

def run_lsqr_init(true_velocity_kms, nx, nz, dx, dz, save_path, lsqr_indices, v_limits):
    v_min, v_max = v_limits
    print(f"--- 执行 LSQR 初始化 ({save_path}) ---")
    # 强制将边界点转换为元组，避免 range 等类型错误
    all_idx = [tuple(pt) for pt in (lsqr_indices['top'] + lsqr_indices['left'] + lsqr_indices['right'])]

    src_rec_pairs = []
    for s_idx in all_idx:
        for r_idx in all_idx:
            # 再次确保是元组
            s_tup = tuple(s_idx)
            r_tup = tuple(r_idx)
            if math.sqrt((s_tup[0] - r_tup[0]) ** 2 + (s_tup[1] - r_tup[1]) ** 2) > 2:
                src_rec_pairs.append((s_tup, r_tup))

    n_rays = len(src_rec_pairs)
    rows, cols, data_vals = [], [], []

    print(f"LSQR: 构建稀疏射线矩阵 A, 射线数: {n_rays}...")
    for i in tqdm(range(n_rays), desc="  Building A matrix"):
        c_idx, lens = straight_ray_path_jf(src_rec_pairs[i][0], src_rec_pairs[i][1], nx, nz, dx, dz)
        for ci, ln in zip(c_idx, lens):
            if ln > 1e-6:
                rows.append(i)
                cols.append(ci)
                data_vals.append(ln)

    A = coo_matrix((data_vals, (rows, cols)), shape=(n_rays, nx * nz)).tocsr()
    b = A.dot((1.0 / true_velocity_kms).flatten()) + 0.005 * np.random.randn(n_rays)

    slowness_lsqr = lsqr(A, b, damp=LSQR_DAMP, iter_lim=LSQR_ITER)[0]

    velocity_lsqr = (1.0 / np.maximum(slowness_lsqr, 1e-8)).reshape((nx, nz))
    velocity_lsqr_f = np.clip(median_filter(velocity_lsqr, size=3), v_min, v_max)

    np.save(save_path, velocity_lsqr_f)

    # 释放大型临时变量
    del A, rows, cols, data_vals, src_rec_pairs, slowness_lsqr, velocity_lsqr
    gc.collect()

    return velocity_lsqr_f

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

        # 释放本源点生成的临时数组
        del tt_field, phi

    del coords_x, coords_z, grid_pts
    gc.collect()
    return np.array(all_pairs), np.array(all_times).reshape(-1, 1), src_locs

print("\n--- 生成 P 波数据 ---")
tp_coords_np, tp_times_np, src_pos_np = generate_travel_time_data(vp_true_kms, N_SOURCES,
                                                                  N_SURFACE_RECEIVERS_PER_SOURCE)
print("\n--- 生成 S 波数据 ---")
ts_coords_np, ts_times_np, _ = generate_travel_time_data(vs_true_kms, N_SOURCES, N_SURFACE_RECEIVERS_PER_SOURCE)

lsqr_indices = generate_unified_boundary_data(N_TOP_POINTS_LSQR, N_SIDE_POINTS_LSQR)
lsqr_vp_np = run_lsqr_init(vp_true_kms, NX, NZ, DX, DZ, "lsqr_vp.npy", lsqr_indices, (VP_MIN, VP_MAX))
lsqr_vs_np = run_lsqr_init(vs_true_kms, NX, NZ, DX, DZ, "lsqr_vs.npy", lsqr_indices, (VS_MIN, VS_MAX))

lsqr_vp_norm = torch.from_numpy((lsqr_vp_np - VP_MIN) / (VP_MAX - VP_MIN + 1e-8)).float().to(device)
lsqr_vs_norm = torch.from_numpy((lsqr_vs_np - VS_MIN) / (VS_MAX - VS_MIN + 1e-8)).float().to(device)

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
# 训练循环（分阶段加权）
# ==========================================================
print("\n========== 开始最终训练 (PINN 反演，动态 LSQR 权重) ==========")
print(f"LSQR 权重将从 {LAMBDA_LSQR_INIT} 线性衰减至 {LAMBDA_LSQR_FINAL}，从第 {DECAY_START} epoch 开始")
start_time = time.time()

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # 动态计算当前 LSQR 权重
    if epoch < DECAY_START:
        lambda_lsqr = LAMBDA_LSQR_INIT
    else:
        decay_ratio = (epoch - DECAY_START) / (EPOCHS - DECAY_START)
        lambda_lsqr = LAMBDA_LSQR_INIT - (LAMBDA_LSQR_INIT - LAMBDA_LSQR_FINAL) * decay_ratio
        lambda_lsqr = max(lambda_lsqr, LAMBDA_LSQR_FINAL)

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

    # 2. Physics Loss
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

    # 3. LSQR Loss
    vp_g, vs_g = velocity_model(grid_coords_2d / norm_xy)
    loss_lsqr = torch.mean((vp_g.view(NX, NZ) - lsqr_vp_norm) ** 2) + torch.mean(
        (vs_g.view(NX, NZ) - lsqr_vs_norm) ** 2)

    loss = LAMBDA_DATA * loss_data + LAMBDA_PHYSICS * loss_phys + lambda_lsqr * loss_lsqr
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 2000 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.4e}, Data: {loss_data.item():.4e}, "
              f"Phys: {loss_phys.item():.4e}, LSQR: {loss_lsqr.item():.4e}, λ_LSQR: {lambda_lsqr:.4f}")

print(f"训练完成，耗时 {time.time() - start_time:.2f} 秒")

# 释放训练过程中占用显存的大张量
del coll_2d, src_rand, coll_4d_norm, vp_n, vs_n, vp, vs, tau_pn, tau_sn, dt_p, dt_s, vp_g, vs_g
if device.type == 'cuda':
    torch.cuda.empty_cache()
gc.collect()

# ==========================================================
# 结果可视化 (保留必要的图像矩阵转换 .T)
# ==========================================================
print("生成可视化图表...")
with torch.no_grad():
    vp_n, vs_n = velocity_model(grid_coords_2d / norm_xy)
    vp_final = (VP_MIN + vp_n * (VP_MAX - VP_MIN)).cpu().numpy().reshape(NX, NZ)
    vs_final = (VS_MIN + vs_n * (VS_MAX - VS_MIN)).cpu().numpy().reshape(NX, NZ)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# Vp
im1 = axes[0, 0].imshow(vp_true_kms.T, origin='upper', extent=[0, X_MAX, Z_MAX, 0], cmap='jet')
axes[0, 0].set_title("True Vp")
plt.colorbar(im1, ax=axes[0, 0])
im2 = axes[0, 1].imshow(lsqr_vp_np.T, origin='upper', extent=[0, X_MAX, Z_MAX, 0], cmap='jet')
axes[0, 1].set_title("LSQR Vp")
plt.colorbar(im2, ax=axes[0, 1])
im3 = axes[0, 2].imshow(vp_final.T, origin='upper', extent=[0, X_MAX, Z_MAX, 0], cmap='jet')
axes[0, 2].set_title("PINN Vp")
plt.colorbar(im3, ax=axes[0, 2])
# Vs
im4 = axes[1, 0].imshow(vs_true_kms.T, origin='upper', extent=[0, X_MAX, Z_MAX, 0], cmap='jet')
axes[1, 0].set_title("True Vs")
plt.colorbar(im4, ax=axes[1, 0])
im5 = axes[1, 1].imshow(lsqr_vs_np.T, origin='upper', extent=[0, X_MAX, Z_MAX, 0], cmap='jet')
axes[1, 1].set_title("LSQR Vs")
plt.colorbar(im5, ax=axes[1, 1])
im6 = axes[1, 2].imshow(vs_final.T, origin='upper', extent=[0, X_MAX, Z_MAX, 0], cmap='jet')
axes[1, 2].set_title("PINN Vs")
plt.colorbar(im6, ax=axes[1, 2])

plt.tight_layout()
plt.savefig("ultimate_inversion_results.png", dpi=300)
plt.show()

print("程序全部执行完毕！图片已保存为 ultimate_inversion_results.png")

# 最终清理
if device.type == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()