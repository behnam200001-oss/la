#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import time
import json
from pathlib import Path

class FixedInstaller:
    def __init__(self):
        self.install_dir = Path(__file__).parent.absolute()
        self.log_file = self.install_dir / "installer.log"
        
    def log_message(self, message):
        """ثبت پیام در فایل لاگ و نمایش آن"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
        
        print(log_entry)
        sys.stdout.flush()  # اطمینان از نمایش فوری خروجی
    
    def run_command(self, command):
        """اجرای دستور و بازگرداندن نتیجه"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=300
            )
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)
    
    def check_python_version(self):
        """بررسی نسخه پایتون"""
        self.log_message("🔍 بررسی نسخه پایتون...")
        version = platform.python_version()
        self.log_message(f"✅ نسخه پایتون: {version}")
        return version
    
    def check_system(self):
        """بررسی سیستم عامل"""
        self.log_message("🔍 بررسی سیستم عامل...")
        system = platform.system()
        release = platform.release()
        self.log_message(f"✅ سیستم عامل: {system} {release}")
        return system, release
    
    def install_basic_packages(self):
        """نصب بسته‌های پایه"""
        self.log_message("📦 نصب بسته‌های پایه...")
        
        basic_packages = [
            "numpy",
            "psutil", 
            "tqdm",
            "pycryptodome",
            "coincurve",
            "base58",
            "bech32",
            "requests",
            "pyyaml"
        ]
        
        for package in basic_packages:
            self.log_message(f"📦 در حال نصب {package}...")
            success, stdout, stderr = self.run_command(f"pip install {package}")
            if success:
                self.log_message(f"✅ {package} نصب شد")
            else:
                self.log_message(f"❌ خطا در نصب {package}: {stderr}")
    
    def check_gpu(self):
        """بررسی وجود GPU"""
        self.log_message("🔍 بررسی سخت‌افزار GPU...")
        
        # بررسی NVIDIA
        success, stdout, stderr = self.run_command("nvidia-smi")
        if success:
            self.log_message("✅ کارت گرافیک NVIDIA یافت شد")
            self.log_message(stdout.split('\n')[0])  # نمایش اولین خط خروجی
            return True
        
        self.log_message("ℹ️ کارت گرافیک NVIDIA یافت نشد")
        return False
    
    def install_gpu_packages(self):
        """نصب بسته‌های GPU در صورت وجود سخت‌افزار"""
        if not self.check_gpu():
            self.log_message("ℹ️ رد شدن نصب بسته‌های GPU")
            return
        
        self.log_message("📦 نصب بسته‌های GPU...")
        
        # ابتدا وابستگی‌های سیستم را نصب کنید
        system = platform.system()
        if system == "Linux":
            self.log_message("🔧 نصب وابستگی‌های سیستم برای Linux...")
            self.run_command("apt-get update && apt-get install -y nvidia-cuda-toolkit")
        
        # نصب pycuda با گزینه‌های خاص
        self.log_message("📦 نصب pycuda...")
        success, stdout, stderr = self.run_command(
            "pip install pycuda --global-option=build_ext --global-option=\"-I/usr/local/cuda/include\" --global-option=\"-L/usr/local/cuda/lib64\""
        )
        
        if success:
            self.log_message("✅ pycuda نصب شد")
        else:
            self.log_message(f"❌ خطا در نصب pycuda: {stderr}")
    
    def create_requirements_file(self):
        """ایجاد فایل requirements.txt اگر وجود ندارد"""
        req_file = self.install_dir / "requirements.txt"
        if not req_file.exists():
            self.log_message("📝 ایجاد فایل requirements.txt...")
            
            requirements = """numpy>=1.21.0
psutil>=5.8.0
tqdm>=4.62.0
pycryptodome>=3.10.0
coincurve>=15.0.0
base58>=2.1.0
bech32>=1.2.0
requests>=2.26.0
pyyaml>=6.0.0
"""
            
            with open(req_file, 'w', encoding='utf-8') as f:
                f.write(requirements)
            
            self.log_message("✅ فایل requirements.txt ایجاد شد")
    
    def run(self):
        """اجرای اصلی نصب‌کننده"""
        self.log_message("=" * 50)
        self.log_message("🛠️  نصب‌کننده پیشرفته Bitcoin/EVM Address Searcher")
        self.log_message("=" * 50)
        
        # بررسی نسخه پایتون
        self.check_python_version()
        
        # بررسی سیستم عامل
        self.check_system()
        
        # ایجاد فایل requirements.txt اگر وجود ندارد
        self.create_requirements_file()
        
        # نصب بسته‌های پایه
        self.install_basic_packages()
        
        # نصب بسته‌های GPU (در صورت وجود)
        self.install_gpu_packages()
        
        self.log_message("=" * 50)
        self.log_message("✅ نصب کامل شد!")
        self.log_message("📖 برای شروع: python main.py --help")
        self.log_message("=" * 50)
        
        return True

if __name__ == "__main__":
    # ایجاد شیء نصب‌کننده و اجرا
    installer = FixedInstaller()
    
    try:
        success = installer.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        installer.log_message(f"❌ خطای غیرمنتظره: {e}")
        import traceback
        installer.log_message(traceback.format_exc())
        sys.exit(1)