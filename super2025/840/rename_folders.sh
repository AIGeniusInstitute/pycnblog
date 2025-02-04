#!/bin/bash

# 初始化计数器
count=1

# 遍历当前目录中的所有项目
for item in *; do
    # 检查是否为目录
    if [ -d "$item" ]; then
        # 构建新目录名
        new_name=$(printf "%02d" "$count")
        
        # 重命名目录
        mv "$item" "$new_name"
        
        # 增加计数器
        ((count++))
    fi
done

echo "重命名完成"