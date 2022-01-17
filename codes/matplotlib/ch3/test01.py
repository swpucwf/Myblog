import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   #用来正常显示负号
# ①获取axs 对象
# fig, axs = plt.subplots(2, 5, figsize=(10, 4), sharex=True, sharey=True)
# fig.suptitle('样例1', size=20)
# for i in range(2):
#     for j in range(5):
#         axs[i][j].scatter(np.random.randn(10), np.random.randn(10))
#         axs[i][j].set_title('第%d行，第%d列'%(i+1,j+1))
#         # 设置
#         axs[i][j].set_xlim(-5,5)
#         axs[i][j].set_ylim(-5,5)
#         if i==1: axs[i][j].set_xlabel('横坐标')
#         if j==0: axs[i][j].set_ylabel('纵坐标')
# fig.tight_layout()
# plt.show()
# ②
# plt.figure()
# # 子图1
# plt.subplot(2,2,1)
# plt.plot([1,2], 'r')
# # 子图2
# plt.subplot(2,2,2)
# plt.plot([1,2], 'b')
# #子图3
# plt.subplot(224)  # 当三位数都小于10时，可以省略中间的逗号，这行命令等价于plt.subplot(2,2,4)
# plt.plot([1,2], 'g')
# plt.show()

# ③
# N = 150
# r = 2 * np.random.rand(N)
# theta = 2 * np.pi * np.random.rand(N)
# area = 200 * r**2
# colors = theta
# plt.subplot(projection='polar')
# plt.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.25)
# plt.show()

# # ④
# data = [805, 598, 831, 586, 357, 562, 692, 623, 575, 605, 623, 585, 573,
#             323, 805, 873, 773, 500, 396, 744, 892, 795, 598, 494, 469, 373]
# data.sort()
#
# theta = np.linspace(0, 2*np.pi, len(data))    # 等分极坐标系
#
# # 设置画布
# fig = plt.figure(figsize=(12, 6),    # 画布尺寸
#                  facecolor='lightyellow'    # 画布背景色
#                 )
#
# # 设置极坐标系
# ax = plt.axes(polar=True)   # 实例化极坐标系
# ax.set_theta_direction(-1)  # 顺时针为极坐标正方向
# ax.set_theta_zero_location('N')     # 极坐标 0° 方向为 N
#
#
# # 在极坐标系中画柱形图
# ax.bar(x=theta,    # 柱体的角度坐标
#        height=data,    # 柱体的高度, 半径坐标
#        width=0.33,    # 柱体的宽度
#        color=np.random.random((len(data),3))
#       )
# ## 绘制中心空白
# ax.bar(x=theta,    # 柱体的角度坐标
#        height=130,    # 柱体的高度, 半径坐标
#        width=0.33,    # 柱体的宽度
#        color='white'
#       )
# # 添加数据标注
# for angle, data, lab in zip(theta, data, data):
#     ax.text(angle + 0.03, data-100, str("西班牙"))
#     ax.text(angle+0.03, data+100, str(data) )
#     ax.set_title("综合趋势图")
#
# ax.set_axis_off()
# plt.show()