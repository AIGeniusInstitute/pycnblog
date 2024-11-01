#!/bin/bash

# 检查是否提供了目录参数
if [ $# -eq 0 ]; then
    echo "请提供一个目录路径作为参数"
    exit 1
fi

# 设置目标目录
target_dir="$1"

# 检查目录是否存在
if [ ! -d "$target_dir" ]; then
    echo "错误: 目录 '$target_dir' 不存在"
    exit 1
fi

# 进入目标目录
cd "$target_dir" || exit

# 初始化计数器
counter=1

# 遍历所有子目录
for dir in */; do
    # 检查是否为目录
    if [ -d "$dir" ]; then
        # 构造新的目录名
        new_name=$(printf "%03d" $counter)
        
        # 重命名目录
        if mv "$dir" "$new_name"; then
            echo "重命名: $dir -> $new_name"
            ((counter++))
        else
            echo "错误: 无法重命名 $dir"
        fi
    fi
done

echo "重命名完成。共处理 $((counter-1)) 个目录。"