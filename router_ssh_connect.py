#!/usr/bin/env python3
"""
路由器 SSH 自动连接脚本
使用 paramiko 实现非交互式 SSH 连接
"""

import paramiko
import sys

def connect_router(host='192.168.123.1', username='admin', password='zte123'):
    """
    连接路由器并执行基本命令
    """
    try:
        # 创建 SSH 客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # 连接路由器
        print(f"正在连接 {username}@{host} ...")
        ssh.connect(host, username=username, password=password, timeout=10)
        print("✅ SSH 连接成功！\n")
        
        # 执行基本命令
        commands = [
            ('uname -a', '系统信息'),
            ('cat /proc/version', '内核版本'),
            ('df -h', '磁盘使用情况'),
            ('free', '内存使用情况'),
            ('ip addr show', '网络接口信息'),
        ]
        
        for cmd, desc in commands:
            print(f"📋 {desc} ({cmd}):")
            stdin, stdout, stderr = ssh.exec_command(cmd, timeout=5)
            output = stdout.read().decode('utf-8', errors='ignore')
            error = stderr.read().decode('utf-8', errors='ignore')
            
            if output:
                print(output)
            if error:
                print(f"错误: {error}")
            print("-" * 50)
        
        # 关闭连接
        ssh.close()
        print("\n✅ 连接已关闭")
        return True
        
    except paramiko.AuthenticationException:
        print("❌ 认证失败：用户名或密码错误")
        return False
    except paramiko.SSHException as e:
        print(f"❌ SSH 连接失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return False

if __name__ == '__main__':
    connect_router()
