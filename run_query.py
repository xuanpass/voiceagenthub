#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import os

# 添加脚本目录到路径
sys.path.insert(0, r"C:\Users\wangxuan\.workbuddy\plugins\marketplaces\cb_teams_marketplace\plugins\finance-data\skills\neodata-financial-search\scripts")

try:
    from query import main as query_main
    import argparse
    
    # 模拟命令行参数
    sys.argv = ["query.py", "--query", "查询近期宏观经济政策动向"]
    
    # 执行查询并捕获输出
    # 由于query.py直接打印结果，我们需要捕获stdout
    from io import StringIO
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result
    
    try:
        query_main()
    except SystemExit:
        pass
    finally:
        sys.stdout = old_stdout
    
    output = result.getvalue()
    
    # 解析JSON并保存
    data = json.loads(output)
    
    # 保存到文件
    output_file = r"D:\workbuddy\Claw\macro_policy_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")
    
except Exception as e:
    print(f"错误: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
