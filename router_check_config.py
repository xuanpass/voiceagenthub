#!/usr/bin/env python3
"""
路由器配置查看脚本
自动识别固件类型，并查看关键配置
"""

import paramiko
import sys

def get_router_config(host='192.168.123.1', username='admin', password='zte123'):
    """
    连接路由器并查看配置
    """
    try:
        # 创建 SSH 客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # 连接路由器
        print(f"📡 正在连接 {username}@{host} ...")
        ssh.connect(host, username=username, password=password, timeout=10)
        print("✅ SSH 连接成功！\n")
        
        # 定义要执行的命令
        commands = [
            # 识别固件类型
            ('cat /etc/os-release 2>/dev/null || echo "No os-release"', '固件信息'),
            ('nvram show 2>/dev/null | head -50', 'NVRAM 配置（前50行）'),
            ('cat /etc/storage/base_setting.json 2>/dev/null || echo "File not found"', '基础配置 (Padavan)'),
            ('uci show 2>/dev/null | head -50', 'UCI 配置（OpenWrt，如果适用）'),
            ('cat /etc/config/network 2>/dev/null || echo "File not found"', '网络配置 (OpenWrt)'),
            ('ifconfig', '网络接口状态'),
            ('cat /etc/wlan/ra0.dat 2>/dev/null || iwconfig ra0 2>/dev/null || echo "WiFi config not found"', 'WiFi 配置'),
            ('ps', '运行中的进程'),
            ('logread | tail -50', '系统日志（最近50行）'),
        ]
        
        for cmd, desc in commands:
            print(f"📋 {desc}")
            print(f"   命令: {cmd}")
            print("-" * 60)
            
            stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
            output = stdout.read().decode('utf-8', errors='ignore')
            error = stderr.read().decode('utf-8', errors='ignore')
            
            if output.strip():
                # 限制输出长度，避免刷屏
                lines = output.split('\n')
                if len(lines) > 30:
                    print('\n'.join(lines[:30]))
                    print(f"... (省略 {len(lines) - 30} 行，共 {len(lines)} 行)")
                else:
                    print(output)
            if error.strip():
                print(f"⚠️  错误: {error[:200]}")
            print("\n")
        
        # 关闭连接
        ssh.close()
        print("✅ 配置查看完成，连接已关闭")
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
    get_router_config()
