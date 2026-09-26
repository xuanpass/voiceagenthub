#!/usr/bin/env python3
"""
路由器配置查看脚本 v2
根据首次探测结果调整命令
"""

import paramiko
import sys

def get_router_config_v2(host='192.168.123.1', username='admin', password='zte123'):
    """
    连接路由器并查看配置（修正版）
    """
    try:
        # 创建 SSH 客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # 连接路由器
        print(f"📡 正在连接 {username}@{host} ...")
        ssh.connect(host, username=username, password=password, timeout=10)
        print("✅ SSH 连接成功！\n")
        
        # 先探测系统环境
        print("=" * 60)
        print("🔍 第一阶段：探测系统环境")
        print("=" * 60 + "\n")
        
        probe_commands = [
            ('which nvram || echo "nvram not found"', 'nvram 命令是否存在'),
            ('which ifconfig || which ip || echo "network tools not found"', '网络工具是否存在'),
            ('ls /etc/storage/ 2>/dev/null || echo "no /etc/storage"', 'Padavan 存储目录'),
            ('ls /etc/config/ 2>/dev/null || echo "no /etc/config"', 'OpenWrt 配置目录'),
            ('cat /proc/version', '系统版本'),
            ('ps | head -20', '运行中的进程（前20个）'),
        ]
        
        for cmd, desc in probe_commands:
            print(f"🔎 {desc}")
            stdin, stdout, stderr = ssh.exec_command(cmd, timeout=5)
            output = stdout.read().decode('utf-8', errors='ignore')
            error = stderr.read().decode('utf-8', errors='ignore')
            
            if output.strip():
                print(output[:500])  # 限制长度
            if error.strip():
                print(f"⚠️  {error[:200]}")
            print()
        
        # 第二阶段：根据探测结果查看配置
        print("=" * 60)
        print("📋 第二阶段：查看关键配置")
        print("=" * 60 + "\n")
        
        config_commands = [
            # 网络配置
            ('ip addr show || ifconfig -a', '网络接口配置'),
            ('ip route show || route -n', '路由表'),
            ('cat /etc/resolv.conf 2>/dev/null', 'DNS 配置'),
            
            # WiFi 配置
            ('iwconfig 2>/dev/null || cat /etc/wlan/*.dat 2>/dev/null | head -50', 'WiFi 配置'),
            ('wl -i ra0 status 2>/dev/null || echo "wl command not found"', 'WiFi 状态（Broadcom）'),
            
            # 系统配置
            ('cat /tmp/syslog.log 2>/dev/null | tail -50', '系统日志'),
            ('dmesg | tail -50', '内核日志'),
            ('cat /etc/passwd', '系统用户'),
            ('df -h', '磁盘使用'),
            ('free', '内存使用'),
            
            # Padavan 特定
            ('nvram show 2>&1 | head -100', 'NVRAM 配置（如果有）'),
            ('cat /etc/storage/settings.json 2>/dev/null || cat /etc/storage/*.conf 2>/dev/null | head -100', '存储的配置'),
        ]
        
        for cmd, desc in config_commands:
            print(f"📋 {desc}")
            print(f"   命令: {cmd}")
            print("-" * 60)
            
            stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
            output = stdout.read().decode('utf-8', errors='ignore')
            error = stderr.read().decode('utf-8', errors='ignore')
            
            if output.strip():
                lines = output.split('\n')
                if len(lines) > 50:
                    print('\n'.join(lines[:50]))
                    print(f"... (省略 {len(lines) - 50} 行，共 {len(lines)} 行)")
                else:
                    print(output)
            elif error.strip():
                print(f"⚠️  错误/提示: {error[:300]}")
            else:
                print("（无输出）")
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
    get_router_config_v2()
