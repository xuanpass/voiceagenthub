import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np

# 中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 数据：6月1日-15日主力资金净流入（亿元）
dates = ['6/1','6/2','6/3','6/4','6/5','6/6','6/7','6/8','6/9','6/10',
         '6/11','6/12','6/13','6/14','6/15']
x = np.arange(len(dates))

semiconductor = np.array([12, 8, 22, 38, 5, -8, -3, -5, -15, -68, -20, 15, -8, 10, 45])
optical_comm  = np.array([8, 5, 15, 10, 3, -5, -2, -3, -10, -45, -12, 22, -5, 18, 52])
ai_server     = np.array([5, 3, 8, 12, 2, -3, -1, -2, -8, -35, -8, 8, -3, 5, 18])
total = semiconductor + optical_comm + ai_server

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#F7F6F3')
ax.set_facecolor('#F7F6F3')

# 合计面积
ax.fill_between(x, total, 0, alpha=0.15, color='#042C53')
ax.plot(x, total, color='#042C53', linewidth=2.5, marker='o', markersize=5, zorder=3, label='科技板块合计')

# 子板块
ax.plot(x, semiconductor, color='#534AB7', linewidth=1.3, linestyle='--', alpha=0.7, label='半导体/芯片')
ax.plot(x, optical_comm,  color='#0F6E56', linewidth=1.3, linestyle='--', alpha=0.7, label='光模块/通信')
ax.plot(x, ai_server,     color='#993C1D', linewidth=1.0, linestyle=':',  alpha=0.6, label='AI算力/服务器')

# 关键事件标注
ann = [
    (9,  total[9],  '6/10\n主力出逃\n-158亿',  '#A32D2D'),
    (11, total[11], '6/12 指数调整\n~1000亿被动流入', '#085041'),
    (14, total[14], '6/15 科技股\n净流入+182亿', '#085041'),
]
for xi, y, text, color in ann:
    ax.annotate('', xy=(xi, y), xytext=(xi, y + (25 if y > 0 else -35)),
                 arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    ax.text(xi, y + (34 if y > 0 else -48), text,
           ha='center', va='center', fontsize=8.5, color=color,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=0.8))

ax.axhline(y=0, color='#444441', linewidth=0.8, linestyle='-', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(dates, fontsize=10)
ax.set_ylabel('主力净流入（亿元）', fontsize=11, color='#2C2C2A')
ax.set_title('A股科技板块资金流向趋势  2026年6月1日—15日',
            fontsize=13, fontweight='500', color='#2C2C2A', pad=12)
ax.grid(axis='y', color='#D3D1C7', linewidth=0.5, alpha=0.6)
ax.set_xlim(-0.5, len(dates)-0.5)

ax.legend(loc='upper left', fontsize=9.5,
          framealpha=0.9, edgecolor='#B4B2A9', borderaxespad=0.5)

fig.text(0.02, 0.01, '数据来源：基于公开报道关键节点还原趋势，非精确逐日数据',
         fontsize=8.5, color='#888780')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
out = 'D:/workbuddy/Claw/科技板块资金流向_2026年6月.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('SAVED:' + out)
