## process_medical_data.py
- 功能：体检原始数据清洗与预处理
- 操作：去重、异常值过滤（$3\delta$ 原则）、缺失值处理
- 依赖：`pandas`、`openpyxl`
- 输入：原始体检Excel文件
- 输出：清洗后可直接用于建模的数据集
