#!/bin/bash

# 定义源目录和目标目录
source_dir="a"
target_dir="java"

# 确保目标目录存在
mkdir -p "$target_dir"

# 初始化计数器
file_count=0
dir_count=1

# 使用find命令递归查找所有文件，并用while循环处理每个文件
find "$source_dir" -type f | while read -r file; do
    # 增加文件计数
    ((file_count++))
    
    # 如果是第一个文件或者文件数量是15的倍数，创建新的目标子目录
    if [ $file_count -eq 1 ] || [ $((file_count % 15)) -eq 1 ]; then
        current_target_dir="${target_dir}/${dir_count}"
        mkdir -p "$current_target_dir"
        ((dir_count++))
    fi
    
    # 移动文件到当前目标子目录
    mv "$file" "$current_target_dir/"
    
    # 打印进度信息
    echo "Moved: $file to $current_target_dir/"
done

echo "File organization complete. Total files processed: $file_count"
echo "Total directories created: $((dir_count - 1))"