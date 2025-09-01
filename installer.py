#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import importlib
import ctypes
import urllib.request
import zipfile
import tarfile
import time
import re
import json
from pathlib import Path

class AdvancedInstaller:
    def __init__(self):
        self.os_type = platform.system()
        self.arch = platform.machine()
        self.cuda_available = False
        self.opencl_available = False
        self.install_dir = Path(__file__).parent.absolute()
        self.requirements_file = self.install_dir / "requirements.txt"
        self.config_file = self.install_dir / "installer_config.json"
        self.log_file = self.install_dir / "installer.log"
        
        # تنظیمات پیش‌فرض
        self.config = {
            "hardware": {
                "cuda_available": False,
                "opencl_available": False,
                "cuda_version": None,
                "cudnn_version": None,
                "gpu_devices": []
            },
            "software": {
                "python_version": platform.python_version(),
                "os_type": self.os_type,
                "arch": self.arch
            },
            "installation": {
                "base_packages": [],
                "gpu_packages": [],
                "failed_packages": [],
                "install_time": None
            }
        }
        
    def log_message(self, message):
        """ثبت پیام در فایل لاگ"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(message)
    
    def run_command(self, command, check=True, capture_output=True):
        """اجرای دستور با مدیریت پیشرفته خطا"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=capture_output, 
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if check and result.returncode != 0:
                error_msg = f"خطا در اجرای دستور: {command}\nخروجی خطا: {result.stderr}"
                self.log_message(f"❌ {error_msg}")
                return False, result.stderr
            
            return True, result.stdout
            
        except subprocess.TimeoutExpired:
            error_msg = f"زمان اجرای دستور به پایان رسید: {command}"
            self.log_message(f"⏰ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"استثناء در اجرای دستور {command}: {e}"
            self.log_message(f"❌ {error_msg}")
            return False, error_msg
    
    def check_admin_privileges(self):
        """بررسی دسترسی ادمین"""
        try:
            if self.os_type == "Windows":
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                return os.geteuid() == 0
        except:
            return False
    
    def detect_cuda(self):
        """تشخیص پیشرفته CUDA"""
        self.log_message("🔍 در حال تشخیص CUDA...")
        
        # بررسی nvidia-smi
        success, output = self.run_command("nvidia-smi", check=False)
        if success and "NVIDIA-SMI" in output:
            self.cuda_available = True
            
            # استخراج نسخه درایور
            driver_match = re.search(r"Driver Version: (\d+\.\d+)", output)
            if driver_match:
                self.config["hardware"]["driver_version"] = driver_match.group(1)
            
            # استخراج اطلاعات GPU
            gpu_match = re.findall(r"(\d+MiB / \d+MiB)", output)
            if gpu_match:
                self.config["hardware"]["gpu_devices"] = gpu_match
        
        # بررسی متغیرهای محیطی CUDA
        cuda_path = os.environ.get('CUDA_PATH', '') or os.environ.get('CUDA_HOME', '')
        if cuda_path and os.path.exists(cuda_path):
            self.config["hardware"]["cuda_available"] = True
            self.config["hardware"]["cuda_path"] = cuda_path
            
            # بررسی نسخه CUDA
            version_file = Path(cuda_path) / "version.txt"
            if version_file.exists():
                with open(version_file, 'r') as f:
                    content = f.read()
                    version_match = re.search(r"CUDA Version (\d+\.\d+\.\d+)", content)
                    if version_match:
                        self.config["hardware"]["cuda_version"] = version_match.group(1)
        
        return self.cuda_available
    
    def detect_opencl(self):
        """تشخیص OpenCL"""
        self.log_message("🔍 در حال تشخیص OpenCL...")
        
        # بررسی وجود clinfo
        success, output = self.run_command("clinfo", check=False)
        if success and "Platform Name" in output:
            self.opencl_available = True
            self.config["hardware"]["opencl_available"] = True
            
            # استخراج اطلاعات پلتفرم
            platform_match = re.search(r"Platform Name:\s+(.+)", output)
            if platform_match:
                self.config["hardware"]["opencl_platform"] = platform_match.group(1)
        
        return self.opencl_available
    
    def detect_hardware(self):
        """تشخیص جامع سخت‌افزار"""
        self.log_message("🔍 در حال تشخیص جامع سخت‌افزار...")
        
        # تشخیص CPU
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            self.config["hardware"]["cpu"] = {
                "name": info.get("brand_raw", "ناشناس"),
                "cores": os.cpu_count(),
                "arch": info.get("arch", "ناشناس")
            }
        except:
            self.config["hardware"]["cpu"] = {
                "name": "ناشناس",
                "cores": os.cpu_count(),
                "arch": self.arch
            }
        
        # تشخیص حافظه
        try:
            import psutil
            mem = psutil.virtual_memory()
            self.config["hardware"]["memory"] = {
                "total": f"{mem.total / (1024**3):.1f} GB",
                "available": f"{mem.available / (1024**3):.1f} GB"
            }
        except:
            self.config["hardware"]["memory"] = {"total": "ناشناس", "available": "ناشناس"}
        
        # تشخیص GPU و درایورها
        self.detect_cuda()
        self.detect_opencl()
        
        # ذخیره نتایج تشخیص
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        return True
    
    def install_system_dependencies(self):
        """نصب وابستگی‌های سیستم"""
        self.log_message("📦 در حال نصب وابستگی‌های سیستم...")
        
        if self.os_type == "Linux":
            # نصب وابستگی‌های سیستم برای لینوکس
            commands = [
                "apt-get update || yum update || dnf update",
                "apt-get install -y build-essential python3-dev python3-pip || "
                "yum install -y gcc python3-devel python3-pip || "
                "dnf install -y gcc python3-devel python3-pip"
            ]
            
            for cmd in commands:
                success, output = self.run_command(cmd, check=False)
                if not success:
                    self.log_message(f"⚠️ خطا در نصب وابستگی‌های سیستم: {output}")
        
        elif self.os_type == "Windows":
            # دانلود و نصب Build Tools برای ویندوز
            self.log_message("ℹ️ برای ویندوز، لطفاً Microsoft Build Tools را نصب کنید")
            self.log_message("   دانلود از: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        
        return True
    
    def parse_requirements(self):
        """تجزیه requirements.txt به بسته‌های پایه و GPU"""
        base_packages = []
        gpu_packages = []
        
        if not self.requirements_file.exists():
            self.log_message("❌ فایل requirements.txt یافت نشد!")
            return False
        
        try:
            with open(self.requirements_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # تشخیص بسته‌های مرتبط با GPU
                    pkg_name = line.split('==')[0].split('>')[0].split('<')[0].strip().lower()
                    is_gpu_package = any(x in pkg_name for x in ['cuda', 'cupy', 'pyopencl', 'pycuda', 'tensorflow-gpu', 'torch'])
                    
                    if is_gpu_package:
                        gpu_packages.append(line)
                    else:
                        base_packages.append(line)
            
            self.config["installation"]["base_packages"] = base_packages
            self.config["installation"]["gpu_packages"] = gpu_packages
            
            return True
            
        except Exception as e:
            self.log_message(f"❌ خطا در پردازش requirements.txt: {e}")
            return False
    
    def install_python_packages(self, packages, package_type="base"):
        """نصب بسته‌های پایتون"""
        for pkg in packages:
            self.log_message(f"📦 در حال نصب {pkg} ({package_type})...")
            
            # اجرای دستور نصب
            success, output = self.run_command(f"{sys.executable} -m pip install --upgrade {pkg}")
            
            if not success:
                self.log_message(f"❌ خطا در نصب {pkg}: {output}")
                self.config["installation"]["failed_packages"].append(pkg)
                
                # تلاش برای نصب از طریق轮换 منابع
                mirrors = [
                    "",
                    "-i https://pypi.org/simple/",
                    "-i https://pypi.tuna.tsinghua.edu.cn/simple/",
                    "-i https://mirrors.aliyun.com/pypi/simple/"
                ]
                
                for mirror in mirrors:
                    self.log_message(f"🔄 تلاش با میرور {mirror}...")
                    success, output = self.run_command(
                        f"{sys.executable} -m pip install {mirror} --upgrade {pkg}",
                        check=False
                    )
                    
                    if success:
                        self.log_message(f"✅ {pkg} با موفقیت نصب شد")
                        break
            else:
                self.log_message(f"✅ {pkg} با موفقیت نصب شد")
        
        return True
    
    def install_cuda_dependencies(self):
        """نصب وابستگی‌های CUDA"""
        if not self.cuda_available:
            self.log_message("ℹ️ CUDA در دسترس نیست - رد شدن نصب وابستگی‌های CUDA")
            return True
        
        self.log_message("🔧 در حال نصب وابستگی‌های CUDA...")
        
        # بررسی وجود toolkit-cuda
        if self.os_type == "Linux":
            success, output = self.run_command("which nvcc", check=False)
            if not success:
                self.log_message("ℹ️ nvcc یافت نشد - نصب toolkit-cuda")
                self.run_command("apt-get install -y nvidia-cuda-toolkit || "
                                "yum install -y nvidia-cuda-toolkit || "
                                "dnf install -y nvidia-cuda-toolkit", check=False)
        
        return True
    
    def setup_jupyter_support(self):
        """راه‌اندازی پشتیبانی از Jupyter Lab"""
        self.log_message("📊 در حال راه‌اندازی پشتیبانی Jupyter Lab...")
        
        # نصب Jupyter Lab
        jupyter_packages = [
            "jupyterlab",
            "ipywidgets",
            "matplotlib",
            "seaborn",
            "pandas"
        ]
        
        self.install_python_packages(jupyter_packages, "jupyter")
        
        # ایجاد پیکربندی اولیه
        self.run_command(f"{sys.executable} -m ipykernel install --user --name=btc_searcher")
        
        self.log_message("✅ پشتیبانی Jupyter Lab راه‌اندازی شد")
        return True
    
    def create_diagnostic_script(self):
        """ایجاد اسکریپت تشخیص مشکلات"""
        diagnostic_script = self.install_dir / "diagnose.py"
        
        script_content = '''
#!/usr/bin/env python3
import sys
import platform
import subprocess
import json
from pathlib import Path

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    except:
        return False, "", "Timeout or error"

def main():
    print("🔍 در حال اجرای تشخیص مشکلات...")
    
    # اطلاعات سیستم
    print("\\n=== اطلاعات سیستم ===")
    print(f"سیستم عامل: {platform.system()} {platform.release()}")
    print(f"معماری: {platform.machine()}")
    print(f"پایتون: {platform.python_version()}")
    
    # بررسی GPU
    print("\\n=== اطلاعات GPU ===")
    success, output, error = run_command("nvidia-smi")
    if success:
        print("✅ NVIDIA GPU یافت شد")
        print(output.split('\\n')[0])  # خط اول خروجی
    else:
        print("❌ NVIDIA GPU یافت نشد")
    
    # بررسی CUDA
    print("\\n=== وضعیت CUDA ===")
    cuda_path = None
    for env_var in ['CUDA_PATH', 'CUDA_HOME']:
        if env_var in os.environ:
            cuda_path = os.environ[env_var]
            print(f"✅ {env_var}: {cuda_path}")
            break
    
    if not cuda_path:
        print("❌ متغیر محیطی CUDA یافت نشد")
    
    # بررسی پکیج‌ها
    print("\\n=== وضعیت پکیج‌ها ===")
    packages = ['pycuda', 'torch', 'tensorflow', 'jupyterlab']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg}: نصب شده")
        except ImportError:
            print(f"❌ {pkg}: نصب نشده")
    
    print("\\n✅ تشخیص کامل شد")

if __name__ == "__main__":
    main()
'''
        
        with open(diagnostic_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # دادن مجوز اجرا
        diagnostic_script.chmod(0o755)
        
        self.log_message("✅ اسکریپت تشخیص مشکلات ایجاد شد")
        self.log_message(f"   برای اجرا: python {diagnostic_script}")
    
    def run_online_troubleshooter(self):
        """اجرای عیب‌یاب آنلاین"""
        self.log_message("🌐 در حال اجرای عیب‌یاب آنلاین...")
        
        # بررسی مشکلات رایج
        common_issues = {
            "pycuda": "https://pycuda.readthedocs.io/en/latest/installation.html",
            "cuda": "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html",
            "opencl": "https://github.com/inducer/pyopencl#installation"
        }
        
        for issue, url in common_issues.items():
            self.log_message(f"🔍 بررسی {issue}: {url}")
        
        # ایجاد گزارش برای اشکال‌زدایی آنلاین
        report = {
            "system": {
                "os": self.os_type,
                "arch": self.arch,
                "python": platform.python_version()
            },
            "hardware": self.config["hardware"],
            "issues": self.config["installation"]["failed_packages"]
        }
        
        report_file = self.install_dir / "troubleshoot_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.log_message(f"📝 گزارش عیب‌یابی ایجاد شد: {report_file}")
        self.log_message("💡 برای کمک آنلاین، این گزارش را در انجمن‌های مربوطه به اشتراک بگذارید")
    
    def save_config(self):
        """ذخیره پیکربندی نهایی"""
        self.config["installation"]["install_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        self.log_message(f"✅ پیکربندی نهایی ذخیره شد: {self.config_file}")
    
    def run(self):
        """اجرای اصلی نصب‌کننده"""
        self.log_message("=" * 60)
        self.log_message("🛠️  نصب‌کننده پیشرفته Bitcoin/EVM Address Searcher")
        self.log_message("=" * 60)
        
        # بررسی دسترسی ادمین
        if not self.check_admin_privileges():
            self.log_message("⚠️ برای نصب برخی компонents ممکن است به دسترسی ادمین نیاز باشد")
        
        # مرحله 1: نصب وابستگی‌های سیستم
        self.install_system_dependencies()
        
        # مرحله 2: تشخیص سخت‌افزار
        self.detect_hardware()
        
        # مرحله 3: تجزیه requirements.txt
        if not self.parse_requirements():
            return False
        
        # مرحله 4: نصب بسته‌های پایه
        self.install_python_packages(self.config["installation"]["base_packages"], "base")
        
        # مرحله 5: نصب وابستگی‌های GPU (در صورت وجود)
        if self.cuda_available:
            self.install_cuda_dependencies()
            self.install_python_packages(self.config["installation"]["gpu_packages"], "gpu")
        else:
            self.log_message("ℹ️ سخت‌افزار GPU یافت نشد - رد شدن نصب بسته‌های GPU")
        
        # مرحله 6: راه‌اندازی پشتیبانی از Jupyter Lab
        self.setup_jupyter_support()
        
        # مرحله 7: ایجاد اسکریپت تشخیص
        self.create_diagnostic_script()
        
        # مرحله 8: اجرای عیب‌یاب آنلاین برای بسته‌های ناموفق
        if self.config["installation"]["failed_packages"]:
            self.log_message("⚠️ برخی بسته‌ها با خطا مواجه شدند:")
            for pkg in self.config["installation"]["failed_packages"]:
                self.log_message(f"   ❌ {pkg}")
            
            self.run_online_troubleshooter()
        
        # ذخیره پیکربندی نهایی
        self.save_config()
        
        self.log_message("=" * 60)
        self.log_message("✅ نصب با موفقیت تکمیل شد!")
        self.log_message("📖 برای شروع کار از دستور زیر استفاده کنید:")
        self.log_message("   python main.py --help")
        self.log_message("📊 برای اجرای Jupyter Lab:")
        self.log_message("   jupyter lab")
        self.log_message("🔍 برای تشخیص مشکلات:")
        self.log_message("   python diagnose.py")
        self.log_message("=" * 60)
        
        return True

if __name__ == "__main__":
    installer = AdvancedInstaller()
    success = installer.run()
    sys.exit(0 if success else 1)