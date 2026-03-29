import pandas as pd
import numpy as np
import os

def process_medical_data(input_file, output_file):
    """
    处理体检数据：格式统一、异常值处理、缺失值处理和去重
    
    参数:
    input_file: 输入文件路径
    output_file: 输出处理后文件路径
    """
    # 读取数据
    print(f"读取文件: {input_file}")
    
    # 用于存储处理后的块
    processed_chunks = []
    
    if input_file.endswith('.csv'):
        # 使用分块读取处理大型CSV文件
        chunksize = 10000  # 每块处理10000行
        total_rows = 0
        
        try:
            # 尝试使用UTF-8编码
            reader = pd.read_csv(input_file, encoding='utf-8', low_memory=False, chunksize=chunksize)
        except UnicodeDecodeError:
            print("UTF-8编码失败，尝试使用GBK编码...")
            reader = pd.read_csv(input_file, encoding='gbk', low_memory=False, chunksize=chunksize)
        
        print(f"开始分块处理，每块 {chunksize} 行...")
        
        for i, chunk in enumerate(reader):
            print(f"处理第 {i+1} 块数据...")
            total_rows += len(chunk)
            
            # 1. 格式统一
            numeric_cols = ['年龄', '检查结果']
            for col in numeric_cols:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
            # 2. 异常值处理
            # 删除年龄负数
            if '年龄' in chunk.columns:
                chunk = chunk[chunk['年龄'] >= 0]
            
            # 删除ALT上千的离谱错误值
            if '项目名称' in chunk.columns and '检查结果' in chunk.columns:
                chunk = chunk[~((chunk['项目名称'].str.contains('ALT', case=False, na=False)) & (chunk['检查结果'] > 1000))]
            
            # 3. 缺失值处理
            # 数值列用均值填充
            for col in numeric_cols:
                if col in chunk.columns:
                    mean_val = chunk[col].mean()
                    chunk[col] = chunk[col].fillna(mean_val)
            
            # 非数值列用众数或合适的值填充
            for col in chunk.columns:
                if col not in numeric_cols and chunk[col].isnull().sum() > 0:
                    try:
                        # 安全地获取众数
                        mode_vals = chunk[col].mode()
                        if len(mode_vals) > 0:
                            mode_val = mode_vals.iloc[0]
                            chunk[col] = chunk[col].fillna(mode_val)
                        else:
                            # 如果没有众数，根据数据类型选择填充值
                            if pd.api.types.is_numeric_dtype(chunk[col]):
                                chunk[col] = chunk[col].fillna(0)
                            else:
                                chunk[col] = chunk[col].fillna('')
                    except Exception as e:
                        # 根据数据类型选择填充值
                        if pd.api.types.is_numeric_dtype(chunk[col]):
                            chunk[col] = chunk[col].fillna(0)
                        else:
                            chunk[col] = chunk[col].fillna('')
            
            # 4. 去重（块内去重）
            if '病人ID' in chunk.columns and '项目代码' in chunk.columns:
                chunk = chunk.drop_duplicates(subset=['病人ID', '项目代码'])
            
            # 5. 3σ原则标记（暂时跳过，最后统一计算）
            
            processed_chunks.append(chunk)
        
        print(f"分块处理完成，总处理行数: {total_rows}")
        
        # 合并所有处理后的块
        df = pd.concat(processed_chunks, ignore_index=True)
        print(f"合并后数据形状: {df.shape}")
        
    else:
        # 对于Excel文件，直接读取
        df = pd.read_excel(input_file)
        print(f"原始数据形状: {df.shape}")
        
        # 1. 格式统一
        numeric_cols = ['年龄', '检查结果']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 2. 异常值处理
        if '年龄' in df.columns:
            df = df[df['年龄'] >= 0]
        if '项目名称' in df.columns and '检查结果' in df.columns:
            df = df[~((df['项目名称'].str.contains('ALT', case=False, na=False)) & (df['检查结果'] > 1000))]
        
        # 3. 缺失值处理
        for col in numeric_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
        for col in df.columns:
            if col not in numeric_cols and df[col].isnull().sum() > 0:
                try:
                    mode_vals = df[col].mode()
                    if len(mode_vals) > 0:
                        mode_val = mode_vals.iloc[0]
                        df[col] = df[col].fillna(mode_val)
                    else:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(0)
                        else:
                            df[col] = df[col].fillna('')
                except Exception as e:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna('')
        
        # 4. 去重
        if '病人ID' in df.columns and '项目代码' in df.columns:
            df = df.drop_duplicates(subset=['病人ID', '项目代码'])
    
    # 全局去重（跨块去重）
    print("\n4. 全局去重处理...")
    if '病人ID' in df.columns and '项目代码' in df.columns:
        before_duplicates = len(df)
        df = df.drop_duplicates(subset=['病人ID', '项目代码'])
        after_duplicates = len(df)
        print(f"删除重复记录: {before_duplicates - after_duplicates} 条")
    
    # 5. 使用3σ原则处理其他异常值
    print("\n5. 使用3σ原则处理其他异常值...")
    if '检查结果' in df.columns:
        # 计算均值和标准差
        mean_val = df['检查结果'].mean()
        std_val = df['检查结果'].std()
        
        # 定义3σ范围
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
        
        # 统计异常值数量
        outlier_count = ((df['检查结果'] < lower_bound) | (df['检查结果'] > upper_bound)).sum()
        print(f"3σ范围内异常值数量: {outlier_count}")
        print(f"3σ范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # 标记异常值
        df['3σ异常标记'] = ((df['检查结果'] < lower_bound) | (df['检查结果'] > upper_bound)).astype(int)
    
    # 保存处理后的数据
    print(f"\n保存处理后的数据到: {output_file}")
    # 对于大型数据，使用CSV格式避免Excel行数限制
    if output_file.endswith('.xlsx'):
        # 如果指定了xlsx格式，改为csv格式
        output_file = output_file.replace('.xlsx', '.csv')
        print(f"由于数据量较大，自动改为保存为CSV格式: {output_file}")
    
    # 保存为CSV文件
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"处理后数据形状: {df.shape}")
    print("处理完成！")

if __name__ == "__main__":
    # 默认输入输出文件
    input_file = "C:\\Users\\39734\\PycharmProjects\\checkdata\\最终版_完全去重列.csv"
    output_file = "processed_medical_data.xlsx"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在！")
        print("请检查文件路径是否正确")
    else:
        process_medical_data(input_file, output_file)