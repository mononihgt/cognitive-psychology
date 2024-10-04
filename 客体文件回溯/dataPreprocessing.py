#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class RemoveList:
    List = ['申伊航', '李嘉宁']
    def add(self, nameList):
        for name in nameList:
            self.List.append(name)


class EncodingList:
    List  = ['gb18030', 'utf-8', 'ansi', 'GB2312', 'GBK', 'utf-16', 'utf-32', 'utf-8-sig', 'utf-16-le', 'utf-16-be', 'utf-32-le', 'utf-32-be']
    def __init__(self, encoding = 'gb18030'):
        self.encoding = encoding


class CalculateAccuracy:
    def __init__(self, repeated_times_col = '', target_col = '', response_col = ''):
        self.repeated_times_col = repeated_times_col
        self.target_col = target_col
        self.response_col = response_col

    def __call__(self, dataFrame):
        if not isinstance(dataFrame, pd.DataFrame):
            raise TypeError("输入必须是 pandas DataFrame 类型")
        
        # 如果是repeated_times_col
        if self.repeated_times_col:
            if self.repeated_times_col not in dataFrame.columns:
                raise ValueError(f"DataFrame 中必须包含 '{self.repeated_times_col}' 列")
            else:
                correct = 0
                total = len(dataFrame[self.repeated_times_col])
                for index, row in dataFrame.iterrows():
                    if row[self.repeated_times_col] == 0:
                        correct += 1
                accuracy = correct / total if total > 0 else 0
                return accuracy
        
        # 如果是target_col和response_col
        if self.target_col not in dataFrame.columns or self.response_col not in dataFrame.columns :
            raise ValueError(f"DataFrame 中必须包含 '{self.target_col}' 和 '{self.response_col}' 列")

        correct = 0
        total = len(dataFrame)
        for index, row in dataFrame.iterrows():
            if row[self.target_col] == row[self.response_col]:
                correct += 1
        accuracy = correct / total if total > 0 else 0
        return accuracy

class DataPackage:
    dataMerged = pd.DataFrame()
    
    def __init__(self, dataFolderName = ''):
        self.dataDir = os.path.join(os.path.abspath('.'), dataFolderName)

    def readData(self, needed_num_row = 0, accuracy_threshold = None, gender_col = '', maxNum = 1000):
        if maxNum % 2 != 0:
            raise ValueError("maxNum 必须是偶数")
        maxofMale = maxNum // 2
        maxofFemale = maxNum // 2  
        numofMale = 0
        numofFemale = 0
        # 尝试encoding
        encodingFlag = 0
        for file in os.listdir(self.dataDir):
            if accuracy_threshold:
                # 如果已经达到最大人数，则跳过
                if numofMale + numofFemale >=  maxNum:
                    break
            
            # 读取file
            filePath = os.path.join(self.dataDir, file)
            print(file)
            # 删除包含removeList中名字的文件
            deleted = 0
            for name in removeList.List:
                if name in file:
                    os.remove(filePath)
                    print(f"{file} 包含 {name}，已删除")
                    deleted = 1
                    break
            if deleted: continue
            
            # 选择encoding并读取
            df = None
            if not encodingFlag:
                for encoding in encodingList.List:
                    try:
                        df = pd.read_csv(filePath, sep=',', encoding=encoding)
                        encodingFlag = True
                        print(f"使用编码 {encoding} 成功")
                        encodingList.encoding = encoding
                        break
                    except UnicodeDecodeError as e:
                        print(f"使用编码 {encoding} 失败: {e}")
            else:
                try:
                    df = pd.read_csv(filePath, sep=',', encoding = encodingList.encoding)
                except UnicodeDecodeError as e:
                    print(f"{file} 使用编码{encodingList.encoding} 失败: {e}")
            
            # 如果使用当前编码失败，尝试剩余的编码
            if df is None:
                remaining_encodings = encodingList.List[encodingList.List.index(encodingList.encoding)+1:]
                for encoding in remaining_encodings:
                    try:
                        df = pd.read_csv(filePath, sep=',', encoding=encoding)
                        print(f"使用编码 {encoding} 成功")
                        encodingList.encoding = encoding
                        break
                    except UnicodeDecodeError as e:
                        print(f"使用编码 {encoding} 失败: {e}")
                
                # 如果所有编码都尝试过仍然无法读取，跳过此文件
                if df is None:
                    print(f"无法读取文件 {file}，跳过")
                    continue
                
            # 需要的行
            df = df.iloc[:needed_num_row]

            # 判断性别
            if not gender_col:
                raise ValueError("gender_col 不能为空")
            if 'Male' in df[gender_col].unique():
                if (numofMale+1) > maxofMale:
                    continue
                numofMale += 1
            elif 'Female' in df[gender_col].unique():
                if (numofFemale+1) > maxofFemale:
                    continue
                numofFemale += 1
            else:
                raise ValueError("gender_col 必须是 'Male' 或 'Female'")
            
            # 判断正确率
            accuracy = calculateAccuracy(df)
            if 'accuracy' not in file:
                newFilename = f"accuracy={accuracy:.2f}_{file}"
                os.rename(os.path.join(self.dataDir, file), os.path.join(self.dataDir, newFilename))
                file = newFilename
            if accuracy_threshold:
                if accuracy > accuracy_threshold:
                    self.dataMerged = pd.concat([self.dataMerged, df],axis=0)
                    print(f"已合并{file}")
                else:
                    os.remove(filePath)
                    print(f"{file} 准确率低于{accuracy_threshold}，已删除")
        print(f"已读取 {numofMale} 名男性数据和 {numofFemale} 名女性数据")
        self.dataMerged = self.dataMerged.dropna(axis=1, how='all')  # 删除全为 NaN 的列
        self.dataMerged = self.dataMerged.loc[:, ~self.dataMerged.columns.str.contains('Unnamed')]  # 删除列名中包含 "Unnamed" 的列
        print("已删除dataMerged中包含 NaN 值的列和列名中包含 'Unnamed' 的列。")
        return self.dataMerged

class DataGenerate:
    def __init__(self, dataFrame):
        if not isinstance(dataFrame, pd.DataFrame):
            raise TypeError("输入必须是 pandas DataFrame 类型")
        self.dataFrame = dataFrame

    def __call__(self, verticalAxis, horizontalAxis, seperatedLines='', seperatedPlots='', subNameCol=''):
        if not verticalAxis:
            raise ValueError("verticalAxis 不能为空")
        if not horizontalAxis:
            raise ValueError("horizontalAxis 不能为空")
        if not subNameCol:
            raise ValueError("subNameCol 不能为空")

        # 将输入转换为列表形式，组合列表并移除空字符串
        def to_list(item):
            return [item] if isinstance(item, str) else item if isinstance(item, list) else []
        horizontalAxis = to_list(horizontalAxis)
        seperatedLines = to_list(seperatedLines)
        seperatedPlots = to_list(seperatedPlots)
        groupedList = seperatedPlots + seperatedLines + horizontalAxis
        groupedList = [subNameCol]+[item for item in groupedList if item]
        
        # 按照groupedList中的顺序，将dataFrame中的verticalAxis进行group
        groupedDataFrame = self.dataFrame.groupby(groupedList).agg(
            **{verticalAxis: (verticalAxis, 'mean')},
            N=(verticalAxis, 'size')
        ).reset_index()
        
        # 将multiIndex的元素保存为不同的列，而不是元组保存为excel中的一个单元
        groupedDataFrame.columns = groupedDataFrame.columns.map(lambda x: '_'.join(x) if isinstance(x, tuple) else x)
        
        # 将groupedDataFrame的数据保存为excel
        filename = f'groupedData, YAxis={verticalAxis}, XAxis={horizontalAxis}, Lines={seperatedLines}, Plots={seperatedPlots}'
        filename = filename.replace('[', '').replace(']', '')
        groupedDataFrame.to_excel(f'{filename}.xlsx', index=False)

        
        print("数据已按照指定列进行分组，multiIndex已展开为单独的列，并保存为Excel文件。")
        return groupedDataFrame

