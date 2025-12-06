# Changelog

## v0.1.0

- 前端：现代暗色 UI，后台线程避免卡顿；左侧参数调节（像素、DoG 尺寸、背景阈值、最小有效数）；左侧日志；图像叠加有效/无效圈与编号。
- 算法：DoG 生成候选 + 半径/峰值过滤；2D 椭圆高斯拟合筛选（轴比 1–3 且纵向拉长、幅值/SNR 阈值、残差/FWHM 物理范围）；纵向积分 profile + 离散 FWHM；统计与警告。
- 工具与测试：`fit_explorer.py` 用于拟合参数分布与阈值探索；`tests/test_analysis_v1.py` 冒烟测试；`docs/project_documentation.md` 汇总需求/设计/测试/迭代。
