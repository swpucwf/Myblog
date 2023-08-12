import numpy as np

# Input Informations (Angles, Depths and Resistivities)
# 输入信息(角度、深度和电阻率)
Alfa, Beta, Gamma = np.float64(0.0), np.float64(0.0), np.float64(0.0)
TvdStart, TvdStep, TvdEnd = np.float64(0.0), np.float64(0.0), np.float64(0.0)
NumLayer = 0
# Tool and Formation Parmameters (Frequecies, Spacing, Permeabilities)4
# Freq和Spacing是用于存储频率和间距的变量。EssnH1和EssnV1是用于存储水平和垂直方向的电场敏感度的数组变量。
# - Freq：频率，用于计算电磁场的频率成分。
# - Spacing：间距，用于计算电磁场的探测器之间的距离。
# - EssnH1：水平方向的电场敏感度，用于计算电磁场中水平方向的响应。
# - EssnV1：垂直方向的电场敏感度，用于计算电磁场中垂直方向的响应。
Freq, Spacing = np.float64(0.0), np.float64(0.0)
EssnH1, EssnV1 = np.zeros(NumLayer), np.zeros(NumLayer)
# Parameters Assignment
Freq = 12 * 1e3
Spacing = 10 / 0.0254 * 0.0254

# Input
with open('data/MCIL_INPUT.txt', 'r') as file:
    TvdStart, TvdStep, TvdEnd = map(float, file.readline().split()[:3])
    Alfa, Beta, Gamma = map(float, file.readline().split()[:3])
    NumLayer = int(file.readline().split()[0])
    Thickness = np.zeros(NumLayer)
    print("Thickness:", Thickness)
    Rh = np.zeros(NumLayer)
    Rv = np.zeros(NumLayer)
    ZLayer = np.zeros(NumLayer)
    EssnH1 = np.zeros(NumLayer)
    EssnV1 = np.zeros(NumLayer)
    for i in range(1, NumLayer + 1):  # Adjusted the range to start from 1
        ZLayer[i - 1], Rh[i - 1], Rv[i - 1] = map(float, file.readline().split()[:3])

Z1Refer = ZLayer[0]
EssnH1 = np.ones(NumLayer)
EssnV1 = np.ones(NumLayer)

# Calculate the thickness of each layer
for i in range(1, NumLayer):  # Adjusted the range to start from 1
    if i > 1 and i < NumLayer:
        Thickness[i-1] = ZLayer[i-1] - ZLayer[i - 2]  # Adjusted the index
    else:
        Thickness[0] = 100
        Thickness[NumLayer - 1] = 100

# Determine the record points
NumSample = int((TvdEnd - TvdStart) / TvdStep) + 1
Tvd = np.zeros(NumSample)

# 打印一下变量
print("TvdStart:",TvdStart)
print("TvdStep:",TvdStep)
print("TvdEnd:",TvdEnd)
print("NumLayer:",NumLayer)
print("Z1Refer:",Z1Refer)
print("ZLayer:",ZLayer)
print("EssnH1:",EssnH1)
print("EssnV1:",EssnV1)
print("Thickness:",Thickness)
#  偏移矩阵 建立
RotMat = np.zeros((3, 3))
RotMat[0, 0] = np.cos(np.deg2rad(Alfa)) * np.cos(np.deg2rad(Beta))
RotMat[0, 1] = -np.cos(np.deg2rad(Alfa)) * np.sin(np.deg2rad(Beta))
RotMat[0, 2] = np.sin(np.deg2rad(Alfa))
RotMat[1, 0] = np.sin(np.deg2rad(Beta))
RotMat[1, 1] = np.cos(np.deg2rad(Beta))
RotMat[1, 2] = 0
RotMat[2, 0] = -np.sin(np.deg2rad(Alfa)) * np.cos(np.deg2rad(Beta))
RotMat[2, 1] = np.sin(np.deg2rad(Alfa)) * np.sin(np.deg2rad(Beta))
RotMat[2, 2] = np.cos(np.deg2rad(Alfa))
#  rotMot   矩阵的成立
print(RotMat)
print(NumSample)

