"""
检测现有环境 - 查看你已经安装了什么
"""

import subprocess
import sys
import os

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check_conda():
    """检查Conda"""
    print_section("Conda 环境检测")
    
    try:
        result = subprocess.run(
            ['conda', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print(f"✅ Conda已安装: {result.stdout.strip()}")
            
            # 列出所有环境
            print("\n已有的Conda环境：")
            result = subprocess.run(
                ['conda', 'env', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            print(result.stdout)
            
            # 检查是否有multimodal环境
            if 'multimodal' in result.stdout:
                print("✅ 检测到 'multimodal' 环境")
                return 'multimodal'
            else:
                print("⚠️  未检测到 'multimodal' 环境")
                return None
        else:
            print("❌ Conda未正确安装")
            return None
    except FileNotFoundError:
        print("❌ 未找到Conda命令")
        print("   请确认Conda已安装并添加到PATH")
        return None
    except Exception as e:
        print(f"❌ 检测出错: {e}")
        return None

def check_current_python():
    """检查当前Python环境"""
    print_section("当前Python环境")
    
    print(f"Python路径: {sys.executable}")
    print(f"Python版本: {sys.version}")
    
    # 检查是否在conda环境中
    if 'conda' in sys.executable.lower() or 'anaconda' in sys.executable.lower():
        print("✅ 当前在Conda环境中")
    else:
        print("⚠️  当前不在Conda环境中")

def check_pytorch_installed():
    """检查PyTorch是否安装"""
    print_section("PyTorch 检测")
    
    try:
        import torch
        print(f"✅ PyTorch已安装")
        print(f"   版本: {torch.__version__}")
        print(f"   安装路径: {torch.__file__}")
        
        # 检查CUDA
        print(f"\nCUDA支持:")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # 测试GPU
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                z = torch.matmul(x, y)
                print(f"   ✅ GPU测试通过")
                return 'gpu'
            except Exception as e:
                print(f"   ❌ GPU测试失败: {e}")
                return 'cpu'
        else:
            print(f"   ⚠️  CUDA不可用，只能使用CPU")
            print(f"   建议重新安装PyTorch with CUDA")
            return 'cpu_only'
            
    except ImportError:
        print("❌ PyTorch未安装")
        return None
    except Exception as e:
        print(f"❌ 检测出错: {e}")
        return None

def check_other_packages():
    """检查其他必要的包"""
    print_section("其他依赖包检测")
    
    packages = {
        'transformers': 'Transformers库',
        'datasets': 'Datasets库',
        'PIL': 'Pillow（图像处理）',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'tqdm': 'tqdm（进度条）',
    }
    
    installed = []
    missing = []
    
    for package, desc in packages.items():
        try:
            if package == 'PIL':
                import PIL
                installed.append(f"✅ {desc}")
            else:
                __import__(package)
                installed.append(f"✅ {desc}")
        except ImportError:
            missing.append(f"❌ {desc}")
    
    print("已安装：")
    for item in installed:
        print(f"  {item}")
    
    if missing:
        print("\n未安装：")
        for item in missing:
            print(f"  {item}")
    
    return len(missing) == 0

def generate_recommendations(conda_env, pytorch_status, packages_ok):
    """生成建议"""
    print_section("建议")
    
    if conda_env == 'multimodal' and pytorch_status == 'gpu' and packages_ok:
        print("""
✅ 你的环境已经配置好了！

下一步：
1. 激活环境：conda activate multimodal
2. 开始学习：查看 START_HERE.md
""")
    
    elif conda_env is None:
        print("""
⚠️  需要创建新环境

方案1：自动创建（推荐）
    bash tools/setup.sh

方案2：手动创建
    conda create -n multimodal python=3.10
    conda activate multimodal
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install transformers datasets pillow matplotlib jupyter
""")
    
    elif pytorch_status is None:
        print(f"""
⚠️  环境 '{conda_env}' 存在，但PyTorch未安装

安装PyTorch：
    conda activate {conda_env}
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
""")
    
    elif pytorch_status == 'cpu_only':
        print(f"""
⚠️  PyTorch已安装，但没有CUDA支持

重新安装PyTorch with CUDA：
    conda activate {conda_env}
    conda uninstall pytorch torchvision
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
""")
    
    elif not packages_ok:
        print(f"""
⚠️  PyTorch已安装，但缺少其他依赖

安装缺失的包：
    conda activate {conda_env}
    pip install transformers datasets pillow matplotlib jupyter pandas tqdm
""")
    
    else:
        print(f"""
✅ 基本环境已配置

下一步：
1. 激活环境：conda activate {conda_env}
2. 测试GPU：python -c "import torch; print(torch.cuda.is_available())"
3. 开始学习：查看 START_HERE.md
""")

def quick_test():
    """快速测试脚本"""
    print_section("快速测试")
    
    test_script = """
import torch
import transformers

print('PyTorch版本:', torch.__version__)
print('Transformers版本:', transformers.__version__)
print('CUDA可用:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print('✅ GPU测试通过')
"""
    
    print("保存以下脚本为 test_env.py 并运行：")
    print("-" * 60)
    print(test_script)
    print("-" * 60)

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║          检测现有环境                                    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # 检测
    conda_env = check_conda()
    check_current_python()
    pytorch_status = check_pytorch_installed()
    packages_ok = check_other_packages()
    
    # 生成建议
    generate_recommendations(conda_env, pytorch_status, packages_ok)
    
    # 快速测试
    quick_test()
    
    print("\n" + "="*60)
    print("检测完成！")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()








