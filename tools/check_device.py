"""
设备检查脚本 - 检查你的硬件是否满足学习要求
"""

import sys
import platform
import subprocess

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check_python():
    """检查Python版本"""
    print_section("Python版本")
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python版本满足要求 (>= 3.8)")
        return True
    else:
        print("❌ Python版本过低，需要 >= 3.8")
        return False

def check_system():
    """检查操作系统"""
    print_section("操作系统")
    os_info = platform.system()
    print(f"操作系统: {os_info}")
    print(f"版本: {platform.version()}")
    print(f"架构: {platform.machine()}")
    
    if os_info in ['Windows', 'Linux', 'Darwin']:
        print(f"✅ {os_info} 系统支持")
        return True
    else:
        print(f"⚠️  未测试过的系统")
        return False

def check_cpu():
    """检查CPU"""
    print_section("CPU信息")
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        
        print(f"物理核心数: {cpu_count}")
        print(f"逻辑核心数: {cpu_count_logical}")
        if cpu_freq:
            print(f"CPU频率: {cpu_freq.current:.2f} MHz")
        
        if cpu_count >= 4:
            print("✅ CPU核心数充足 (>= 4)")
            return True
        else:
            print("⚠️  CPU核心数较少，可能影响训练速度")
            return False
    except ImportError:
        print("⚠️  未安装psutil，无法检测详细CPU信息")
        print("   安装: pip install psutil")
        return None

def check_memory():
    """检查内存"""
    print_section("内存信息")
    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_gb = mem.total / (1024**3)
        
        print(f"总内存: {mem_gb:.2f} GB")
        print(f"可用内存: {mem.available / (1024**3):.2f} GB")
        print(f"使用率: {mem.percent}%")
        
        if mem_gb >= 16:
            print("✅ 内存充足 (>= 16GB)，可以训练中型模型")
            return "excellent"
        elif mem_gb >= 8:
            print("✅ 内存足够 (>= 8GB)，可以训练小模型")
            return "good"
        else:
            print("❌ 内存不足 (<8GB)，建议升级或使用云服务")
            return "insufficient"
    except ImportError:
        print("⚠️  未安装psutil")
        return None

def check_gpu():
    """检查GPU"""
    print_section("GPU信息")
    
    # 检查NVIDIA GPU
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            print(f"检测到 {len(gpus)} 个NVIDIA GPU:")
            
            has_sufficient_gpu = False
            for i, gpu in enumerate(gpus):
                parts = gpu.split(', ')
                if len(parts) >= 2:
                    name = parts[0]
                    memory = parts[1]
                    print(f"\nGPU {i}: {name}")
                    print(f"   显存: {memory}")
                    if len(parts) >= 3:
                        print(f"   驱动版本: {parts[2]}")
                    
                    # 提取显存大小
                    try:
                        mem_gb = float(memory.split()[0])
                        if mem_gb >= 6:
                            has_sufficient_gpu = True
                    except:
                        pass
            
            # 检查PyTorch CUDA支持
            try:
                import torch
                print(f"\nPyTorch版本: {torch.__version__}")
                print(f"CUDA可用: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"CUDA版本: {torch.version.cuda}")
                    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
                    print(f"检测到GPU数量: {torch.cuda.device_count()}")
                    for i in range(torch.cuda.device_count()):
                        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                    print("✅ PyTorch GPU支持正常")
                else:
                    print("❌ PyTorch无法使用CUDA，需要重新安装")
                    print("   安装命令: conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")
            except ImportError:
                print("⚠️  未安装PyTorch")
            
            if has_sufficient_gpu:
                print("\n✅ GPU显存充足 (>= 6GB)，可以训练中小型模型")
                return "gpu"
            else:
                print("\n⚠️  GPU显存较小，建议使用云GPU")
                return "gpu_small"
        else:
            print("❌ 未检测到NVIDIA GPU或nvidia-smi不可用")
            return check_other_gpu()
    except FileNotFoundError:
        print("❌ nvidia-smi不可用，未检测到NVIDIA GPU")
        return check_other_gpu()
    except Exception as e:
        print(f"❌ GPU检测出错: {e}")
        return check_other_gpu()

def check_other_gpu():
    """检查其他GPU（AMD、Intel等）"""
    print("\n检查其他GPU...")
    
    # 检查PyTorch是否支持MPS（Apple Silicon）
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ 检测到Apple Silicon GPU (MPS)")
            print("   可以使用Metal加速")
            return "mps"
    except:
        pass
    
    print("⚠️  未检测到可用GPU，将使用CPU训练")
    print("   建议使用Google Colab或其他云GPU服务")
    return "cpu"

def check_disk():
    """检查磁盘空间"""
    print_section("磁盘空间")
    try:
        import psutil
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        
        print(f"总空间: {disk.total / (1024**3):.2f} GB")
        print(f"已使用: {disk.used / (1024**3):.2f} GB")
        print(f"可用空间: {disk_gb:.2f} GB")
        
        if disk_gb >= 50:
            print("✅ 磁盘空间充足 (>= 50GB)")
            return True
        elif disk_gb >= 20:
            print("⚠️  磁盘空间有限 (20-50GB)，够用但需要定期清理")
            return True
        else:
            print("❌ 磁盘空间不足 (<20GB)，需要清理或扩容")
            return False
    except ImportError:
        print("⚠️  未安装psutil")
        return None

def check_network():
    """检查网络连接"""
    print_section("网络连接")
    try:
        import urllib.request
        urllib.request.urlopen('https://www.google.com', timeout=3)
        print("✅ 网络连接正常")
        
        # 测试HuggingFace连接
        try:
            urllib.request.urlopen('https://huggingface.co', timeout=3)
            print("✅ 可以访问HuggingFace")
        except:
            print("⚠️  无法访问HuggingFace，可能需要镜像")
        
        return True
    except:
        print("❌ 网络连接失败")
        return False

def generate_report(results):
    """生成检测报告和建议"""
    print_section("检测总结")
    
    gpu_type = results.get('gpu', 'cpu')
    memory_status = results.get('memory', 'unknown')
    
    # 判断设备类型
    if gpu_type == 'gpu' and memory_status == 'excellent':
        device_level = "优秀"
        recommendation = """
✅ 你的设备配置优秀！可以：
   - 本地训练中小型模型（CLIP、BLIP等）
   - 运行所有学习项目
   - 尝试7B以下的LLM微调
   
建议：
   - 直接在本地学习和训练
   - 节省云服务成本
"""
    elif gpu_type in ['gpu', 'gpu_small'] and memory_status == 'good':
        device_level = "良好"
        recommendation = """
✅ 你的设备配置良好！可以：
   - 本地训练小模型
   - 运行大部分学习项目
   - 基础学习完全够用
   
建议：
   - 前2个月本地学习
   - 训练大模型时使用云GPU（Colab、AutoDL）
"""
    elif gpu_type == 'mps':
        device_level = "良好（Mac）"
        recommendation = """
✅ Apple Silicon支持Metal加速！可以：
   - 本地训练小模型
   - 运行学习项目
   
建议：
   - PyTorch使用MPS后端
   - 训练大模型时使用云GPU
"""
    else:
        device_level = "基础"
        recommendation = """
⚠️  你的设备配置较基础，建议：
   
方案1: 使用云GPU（推荐）
   - Google Colab（免费，每天12小时）
   - Kaggle Notebooks（免费，每周30小时）
   - AutoDL（按量付费，0.8元/小时起）
   
方案2: 升级硬件
   - 加装显卡（RTX 3060以上）
   - 增加内存（16GB+）
   
方案3: CPU学习（慢但可行）
   - 理论学习完全可以
   - 小项目可以运行
   - 训练会很慢
"""
    
    print(f"\n设备等级: {device_level}")
    print(recommendation)
    
    # 云服务推荐
    print_section("云GPU服务推荐")
    print("""
1. Google Colab（最推荐新手）
   - 免费版：T4 GPU，12小时/天
   - 优点：零配置，开箱即用
   - 缺点：会话限制，不能长时间训练
   - 链接：https://colab.research.google.com/

2. Kaggle Notebooks
   - 免费：P100 GPU，30小时/周
   - 优点：资源稳定
   - 缺点：需要注册
   - 链接：https://www.kaggle.com/code

3. AutoDL（国内推荐）
   - 按量付费：0.8-3元/小时
   - 优点：国内快，配置灵活
   - 缺点：需要付费
   - 链接：https://www.autodl.com/

4. 阿里云PAI / 腾讯云
   - 企业级服务
   - 适合有预算的同学
""")

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║          多模态AI学习 - 设备检测工具                    ║
║        检查你的硬件是否满足学习要求                      ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # 执行检测
    results['python'] = check_python()
    results['system'] = check_system()
    results['cpu'] = check_cpu()
    results['memory'] = check_memory()
    results['gpu'] = check_gpu()
    results['disk'] = check_disk()
    results['network'] = check_network()
    
    # 生成报告
    generate_report(results)
    
    print_section("下一步")
    print("""
1. 如果设备足够：
   → 继续按 START_HERE.md 开始学习

2. 如果设备不够：
   → 查看 tools/cloud-gpu-guide.md 学习使用云GPU

3. 安装必要的库：
   → pip install psutil（如果上面有警告）
   → 按 tools/setup.sh 安装PyTorch

4. 遇到问题：
   → 查看 resources.md 获取帮助
    """)

if __name__ == '__main__':
    main()








