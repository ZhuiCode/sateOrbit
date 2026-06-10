
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 参数设置 ====================
# 地球参数
R_EARTH = 6371.0  # 地球半径 (km)
MU_EARTH = 398600.4418  # 地球引力常数 (km^3/s^2)

# 卫星轨道参数
ORBIT_ALTITUDE = 550.0  # 轨道高度 (km)
ORBIT_RADIUS = R_EARTH + ORBIT_ALTITUDE  # 轨道半径 (km)
ORBIT_PERIOD = 2 * np.pi * np.sqrt(ORBIT_RADIUS**3 / MU_EARTH)  # 轨道周期 (s)
ORBIT_PERIOD_MIN = ORBIT_PERIOD / 60  # 轨道周期 (分钟)

# 地面站位置（例如：北京，纬度北正，经度东正）
GS_LAT = 39.9042   # 地面站纬度 (度)
GS_LON = 116.4074  # 地面站经度 (度)

# 模拟参数
NUM_ORBITS = 45   # 模拟的轨道圈数（约3天，约45圈）
POINTS_PER_ORBIT = 360  # 每圈采样点数

print(f"轨道周期: {ORBIT_PERIOD_MIN:.1f} 分钟")
print(f"模拟天数: {NUM_ORBITS * ORBIT_PERIOD_MIN / 1440:.1f} 天")

# ==================== 核心计算函数 ====================
def eci_to_ecef(raan, arg_lat, orbit_radius):
    """
    将ECI坐标（地心惯性系）转换为ECEF坐标（地心地固系）
    简化模型：假设轨道为圆形，未考虑地球扁率摄动
    """
    # 卫星在轨道平面内的位置（ECI）
    x_eci = orbit_radius * np.cos(arg_lat)
    y_eci = orbit_radius * np.sin(arg_lat)
    z_eci = 0
    
    # 考虑RAAN（升交点赤经）旋转
    cos_raan = np.cos(np.radians(raan))
    sin_raan = np.sin(np.radians(raan))
    
    x_rot = x_eci * cos_raan - y_eci * sin_raan
    y_rot = x_eci * sin_raan + y_eci * cos_raan
    z_rot = z_eci
    
    return np.array([x_rot, y_rot, z_rot])

def geodetic_to_ecef(lat, lon, alt):
    """大地坐标转ECEF坐标"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # WGS-84椭球参数
    a = 6378.137  # 长半轴 (km)
    f = 1 / 298.257223563
    e2 = 2*f - f*f
    
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N*(1-e2) + alt) * np.sin(lat_rad)
    
    return np.array([x, y, z])

def compute_satellite_ground_range_and_angle(sat_ecef, gs_ecef):
    """
    计算卫星到地面站的距离和从卫星看地面站的方向角
    返回: (距离, 方位角)
    """
    # 卫星到地面站的矢量
    r_vec = gs_ecef - sat_ecef
    distance = np.linalg.norm(r_vec)
    
    # 在卫星本体坐标系中的矢量（简化：将卫星的z轴指向地心）
    # 这里采用更精确的计算：方位角从真北顺时针测量
    sat_radius = np.linalg.norm(sat_ecef)
    sat_lat = np.degrees(np.arcsin(sat_ecef[2] / sat_radius))
    sat_lon = np.degrees(np.arctan2(sat_ecef[1], sat_ecef[0]))
    
    gs_lat = np.degrees(np.arcsin(gs_ecef[2] / R_EARTH))
    gs_lon = np.degrees(np.arctan2(gs_ecef[1], gs_ecef[0]))
    
    # 计算从卫星看向地面站的方位角（球面三角公式）
    # 使用正弦和余弦定理计算方位角
    lat1_rad = np.radians(sat_lat)
    lat2_rad = np.radians(gs_lat)
    delta_lon_rad = np.radians(gs_lon - sat_lon)
    
    # 计算方位角（从真北顺时针）
    y = np.sin(delta_lon_rad) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon_rad)
    azimuth = np.degrees(np.arctan2(y, x))
    azimuth = (azimuth + 360) % 360
    
    return distance, azimuth

# ==================== 生成数据 ====================
# 地球自转角速度 (度/秒)
EARTH_ROTATION_RATE = 360 / 86400  # 度/秒

# 轨道进动速率（简化：考虑J2摄动）
J2 = 1.08262668e-3
INCLINATION = 53.0  # 轨道倾角 (度)
RAAN_DOT = -2.06474e14 * J2 / (ORBIT_RADIUS**3.5) * np.cos(np.radians(INCLINATION))
RAAN_DOT_DEG = np.degrees(RAAN_DOT)  # 度/秒

print(f"升交点赤经进动速率: {RAAN_DOT_DEG*3600*24:.2f} 度/天")

# 地面站ECEF坐标
gs_ecef = geodetic_to_ecef(GS_LAT, GS_LON, 0)

# 存储所有轨道的数据
all_ranges = []
all_azimuths = []
orbit_min_distances = []

# 模拟多个轨道周期
for orbit_num in range(NUM_ORBITS):
    # 当前轨道的升交点赤经（考虑进动和初始相位）
    RAAN = 0 + RAAN_DOT_DEG * orbit_num * ORBIT_PERIOD
    
    # 当前轨道的时间点
    time_points = np.linspace(0, ORBIT_PERIOD, POINTS_PER_ORBIT)
    
    ranges = []
    azimuths = []
    min_distance = float('inf')
    min_azimuth = None
    
    for t in time_points:
        # 轨道平面内的幅角（平近点角，简化）
        arg_lat = 360 * t / ORBIT_PERIOD  # 度
        
        # 考虑地球自转对地面站位置的影响
        earth_rotation = EARTH_ROTATION_RATE * t  # 地球自转角度（度）
        current_gs_ecef = np.array([
            gs_ecef[0] * np.cos(np.radians(earth_rotation)) - gs_ecef[1] * np.sin(np.radians(earth_rotation)),
            gs_ecef[0] * np.sin(np.radians(earth_rotation)) + gs_ecef[1] * np.cos(np.radians(earth_rotation)),
            gs_ecef[2]
        ])
        
        # 计算卫星位置
        sat_ecef = eci_to_ecef(RAAN, arg_lat, ORBIT_RADIUS)
        
        # 计算距离和方位角
        dist, az = compute_satellite_ground_range_and_angle(sat_ecef, current_gs_ecef)
        
        ranges.append(dist)
        azimuths.append(az)
        
        if dist < min_distance:
            min_distance = dist
            min_azimuth = az
    
    # 归一化距离（除以最小距离）
    ranges_normalized = [r / min_distance for r in ranges]
    
    all_ranges.append(ranges_normalized)
    all_azimuths.append(azimuths)
    orbit_min_distances.append((min_distance, min_azimuth))

# ==================== 绘图 ====================
# 图5(a): 单次旋转
fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': 'polar'})

# 子图5(a)
ax1 = axes[0]
# 选择第10个轨道作为典型示例
orbit_idx = NUM_ORBITS // 3
ranges_single = all_ranges[orbit_idx]
azimuths_single = all_azimuths[orbit_idx]

# 在极坐标中绘图
az_rad = np.radians(azimuths_single)
ax1.plot(az_rad, ranges_single, 'b-', linewidth=2)
ax1.scatter(np.radians(orbit_min_distances[orbit_idx][1]), 1.0, 
           color='red', s=100, zorder=5, label='最近点')

# 设置极坐标属性
ax1.set_theta_zero_location('N')  # 0度在真北
ax1.set_theta_direction(-1)       # 顺时针为正
ax1.set_rmax(1.5)
ax1.set_rticks([1.0, 1.2, 1.4])
ax1.set_rlabel_position(45)
ax1.grid(True)
ax1.set_title('(a) 单次轨道过境', fontsize=14, pad=20)
ax1.legend(loc='upper right')

# 标记方向
ax1.text(np.radians(0), 1.6, 'N', fontsize=12, ha='center')
ax1.text(np.radians(90), 1.6, 'E', fontsize=12, ha='center')
ax1.text(np.radians(180), 1.6, 'S', fontsize=12, ha='center')
ax1.text(np.radians(270), 1.6, 'W', fontsize=12, ha='center')

# 子图5(b): 三天完整轨迹
ax2 = axes[1]
# 使用颜色映射区分不同轨道
colors = plt.cm.viridis(np.linspace(0, 1, NUM_ORBITS))

for i in range(NUM_ORBITS):
    az_rad = np.radians(all_azimuths[i])
    # 只绘制每个轨道的部分线段避免过于密集
    ax2.plot(az_rad[::5], all_ranges[i][::5], color=colors[i], linewidth=0.8, alpha=0.6)

# 标记最近点位置
min_az_list = [az for _, az in orbit_min_distances]
min_dist_list = [dist for dist, _ in orbit_min_distances]
# 找到所有最近点并标记
recent_points_az = [az for _, az in orbit_min_distances if az is not None]
recent_points_dist = [1.0] * len(recent_points_az)

ax2.scatter(np.radians(recent_points_az), recent_points_dist, 
           color='red', s=30, alpha=0.7, zorder=5, label='各轨道最近点')

# 设置相同的极坐标属性
ax2.set_theta_zero_location('N')
ax2.set_theta_direction(-1)
ax2.set_rmax(1.5)
ax2.set_rticks([1.0, 1.2, 1.4])
ax2.set_rlabel_position(45)
ax2.grid(True)
ax2.set_title('(b) 三天完整距离矢量', fontsize=14, pad=20)

# 标记方向
ax2.text(np.radians(0), 1.6, 'N', fontsize=12, ha='center')
ax2.text(np.radians(90), 1.6, 'E', fontsize=12, ha='center')
ax2.text(np.radians(180), 1.6, 'S', fontsize=12, ha='center')
ax2.text(np.radians(270), 1.6, 'W', fontsize=12, ha='center')

# 添加图例
ax2.legend(loc='upper right', fontsize=10)

# 总标题
fig.suptitle('卫星视角下的地面站距离变化', fontsize=16, fontweight='bold', y=1.05)

plt.tight_layout()
plt.savefig('satellite_ground_range_polar.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 补充信息输出 ====================
print("\n=== 统计信息 ===")
print(f"最小距离范围: {min(orbit_min_distances)[0]:.1f} - {max(orbit_min_distances)[0]:.1f} km")
print(f"最近点方位角范围: {min(recent_points_az):.1f}° - {max(recent_points_az):.1f}°")

# 分析最近点主要分布（南半球方向）
angles_mod_180 = [(az + 180) % 360 - 180 for az in recent_points_az]
if np.mean(angles_mod_180) > 0:
    print("最近点主要分布在东半球方向")
else:
    print("最近点主要分布在西半球方向")

# 绘制三维轨迹图（可选）
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# 绘制地球
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_earth = R_EARTH * np.outer(np.cos(u), np.sin(v))
y_earth = R_EARTH * np.outer(np.sin(u), np.sin(v))
z_earth = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
ax_3d.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3)

# 绘制卫星轨迹样本
for i in range(0, NUM_ORBITS, 5):
    orbit_points = []
    for t in np.linspace(0, 360, 100):
        sat_pos = eci_to_ecef(0 + RAAN_DOT_DEG * i * ORBIT_PERIOD, t, ORBIT_RADIUS)
        orbit_points.append(sat_pos)
    orbit_points = np.array(orbit_points)
    ax_3d.plot(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2], 
              color=colors[i], alpha=0.5, linewidth=0.5)

# 标记地面站
ax_3d.scatter([gs_ecef[0]], [gs_ecef[1]], [gs_ecef[2]], 
             color='red', s=100, label='地面站')

ax_3d.set_xlabel('X (km)')
ax_3d.set_ylabel('Y (km)')
ax_3d.set_zlabel('Z (km)')
ax_3d.set_title('卫星轨道与地面站三维位置')
ax_3d.legend()

plt.savefig('satellite_3d_trajectory.png', dpi=300, bbox_inches='tight')
#plt.show()