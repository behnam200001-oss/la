# در کلاس AdvancedInstaller، متد install_python_packages را به روز کنید:

def install_python_packages(self, packages, package_type="base"):
    """نصب بسته‌های پایتون با مدیریت خطای پیشرفته"""
    for pkg in packages:
        self.log_message(f"📦 در حال نصب {pkg} ({package_type})...")
        
        # مدیریت بسته‌های خاص
        install_cmd = f"{sys.executable} -m pip install --upgrade {pkg}"
        
        if "cpuinfo" in pkg.lower():
            # نصب cpuinfo با گزینه‌های خاص
            install_cmd = f"{sys.executable} -m pip install --upgrade {pkg} --no-binary :all:"
        
        elif "pycuda" in pkg.lower():
            # نصب pycuda با گزینه‌های خاص برای سیستم‌های مبتنی بر Debian
            if self.os_type == "Linux":
                install_cmd = f"{sys.executable} -m pip install --upgrade {pkg} --global-option=build_ext --global-option=\"-I/usr/local/cuda/include\" --global-option=\"-L/usr/local/cuda/lib64\""
        
        elif "cupy" in pkg.lower():
            # نصب cupy با شناسایی خودکار نسخه CUDA
            cuda_version = self.detect_cuda_version()
            if cuda_version:
                install_cmd = f"{sys.executable} -m pip install --upgrade cupy-cuda{cuda_version.replace('.', '')[:2]}"
            else:
                self.log_message(f"⚠️ رد شدن نصب {pkg} (CUDA یافت نشد)")
                continue
        
        # اجرای دستور نصب
        success, output = self.run_command(install_cmd)
        
        if not success:
            self.log_message(f"❌ خطا در نصب {pkg}: {output}")
            self.config["installation"]["failed_packages"].append(pkg)
            
            # تلاش برای نصب از طریق轮换 منابع
            mirrors = [
                "-i https://pypi.org/simple/",
                "-i https://pypi.tuna.tsinghua.edu.cn/simple/",
                "-i https://mirrors.aliyun.com/pypi/simple/",
                "--extra-index-url https://download.pytorch.org/whl/cu113"
            ]
            
            for mirror in mirrors:
                self.log_message(f"🔄 تلاش با میرور {mirror}...")
                alt_cmd = f"{sys.executable} -m pip install {mirror} --upgrade {pkg}"
                
                # تنظیمات خاص برای بسته‌های مشکل‌دار
                if "cpuinfo" in pkg.lower():
                    alt_cmd += " --no-binary :all:"
                
                success, output = self.run_command(alt_cmd, check=False)
                
                if success:
                    self.log_message(f"✅ {pkg} با موفقیت نصب شد")
                    if pkg in self.config["installation"]["failed_packages"]:
                        self.config["installation"]["failed_packages"].remove(pkg)
                    break
        else:
            self.log_message(f"✅ {pkg} با موفقیت نصب شد")
    
    return True

# اضافه کردن متد جدید برای تشخیص نسخه CUDA
def detect_cuda_version(self):
    """تشخیص نسخه CUDA نصب شده"""
    # بررسی متغیرهای محیطی
    cuda_path = os.environ.get('CUDA_PATH', '') or os.environ.get('CUDA_HOME', '')
    if cuda_path:
        version_file = Path(cuda_path) / "version.txt"
        if version_file.exists():
            with open(version_file, 'r') as f:
                content = f.read()
                version_match = re.search(r"CUDA Version (\d+\.\d+\.\d+)", content)
                if version_match:
                    return version_match.group(1)
    
    # بررسی از طریق nvcc
    success, output = self.run_command("nvcc --version", check=False)
    if success:
        version_match = re.search(r"release (\d+\.\d+)", output)
        if version_match:
            return version_match.group(1)
    
    # بررسی از طریق nvidia-smi
    success, output = self.run_command("nvidia-smi", check=False)
    if success:
        version_match = re.search(r"CUDA Version: (\d+\.\d+)", output)
        if version_match:
            return version_match.group(1)
    
    return None