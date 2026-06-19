import numpy as np
from scipy import stats

# # fixed_field_x = np.array([0.9541, 0.8569, 1.0354, 0.8747, 1.2017])
# # pid_control_x = np.array([0.7813, 0.8339, 0.7706, 0.6545, 0.7933])
# fixed_field_y = np.array([1.2999, 0.8449, 1.0293, 0.8183, 1.4291])
# pid_control_y = np.array([0.6327, 0.7163, 0.7628, 0.5787, 0.6535])
# # fixed_field_z = np.array([3.1972, 1.7358, 2.0391, 2.1190, 3.2041])
# pid_control_z = np.array([1.7497, 1.3150, 1.2760, 1.4383, 1.5156])

# ウェルチのt検定を実行
t_stat, p_value = stats.ttest_ind(fixed_field_y, pid_control_y, equal_var=False)

print(f"t統計量: {t_stat:.4f}")
print(f"p値: {p_value:.4f}")

# 有意判定（有意水準 5%）
if p_value < 0.05:
    print("有意差あり：フィードバック制御により安定性が有意に向上しました。")
else:
    print("有意差なし：統計的な差は認められませんでした。")