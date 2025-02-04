#!/bin/bash

# 检查参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

source_dir="$1"
destination_dir="$2"

# 检查源目录是否存在
if [ ! -d "$source_dir" ]; then
    echo "Source directory does not exist: $source_dir"
    exit 1
fi

# 创建目标目录（如果不存在）
mkdir -p "$destination_dir"

# 初始化计数器
file_count=0
group_count=1

# 递归遍历源目录
find "$source_dir" -type f | while read -r file; do
    # 创建新的组目录（如果需要）
    group_dir=$(printf "%s/%03d" "$destination_dir" $group_count)
    mkdir -p "$group_dir"

    # 移动文件
    mv "$file" "$group_dir/"

    # 增加文件计数
    ((file_count++))

    # 如果达到 15 个文件，增加组计数并重置文件计数
    if [ $file_count -eq 15 ]; then
        ((group_count++))
        file_count=0
    fi
done

echo "整理完成。最后一组是 $(printf "%03d" $group_count)"