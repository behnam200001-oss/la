import time
import threading
from datetime import datetime, timedelta
from tqdm import tqdm

class ProgressReporter:
    def __init__(self, interval=5):
        self.interval = interval
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
        self.total_checked = 0
        self.total_found = 0
        self.start_time = None
        self.last_update_time = None
        self.last_update_count = 0
        
        self.pbar = None
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        self.pbar = tqdm(
            desc="Searching addresses", 
            unit=" keys", 
            unit_scale=True,
            dynamic_ncols=True
        )
        
        self.thread = threading.Thread(target=self._report_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.pbar:
            self.pbar.close()
        
        self._print_final_report()
    
    def update(self, checked, found=0):
        with self.lock:
            self.total_checked += checked
            self.total_found += found
            if self.pbar:
                try:
                    self.pbar.update(checked)
                    self.pbar.set_postfix(found=self.total_found)
                except Exception as e:
                    print(f"Error in progress report: {e}")
    
    def _report_loop(self):
        while self.running:
            time.sleep(self.interval)
            self._print_progress_report()
    
    def _print_progress_report(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.start_time
            current_count = self.total_checked
            
            time_since_last = now - self.last_update_time
            keys_since_last = current_count - self.last_update_count
            
            if time_since_last > 0:
                current_speed = keys_since_last / time_since_last
                avg_speed = current_count / elapsed
            else:
                current_speed = 0
                avg_speed = 0
            
            self.last_update_time = now
            self.last_update_count = current_count
            
            report_lines = [
                "=" * 60,
                f"Progress report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"   Keys checked: {current_count:,}",
                f"   Addresses found: {self.total_found}",
                f"   Search speed: {current_speed:,.0f} key/s (current) - {avg_speed:,.0f} key/s (average)",
                f"   Elapsed time: {str(timedelta(seconds=int(elapsed)))}",
                "=" * 60
            ]
            
            if self.pbar:
                try:
                    self.pbar.write("\n".join(report_lines))
                except Exception as e:
                    print(f"Error writing progress report: {e}")
    
    def _print_final_report(self):
        elapsed = time.time() - self.start_time
        
        report_lines = [
            "=" * 60,
            "Final search result",
            f"   Total keys checked: {self.total_checked:,}",
            f"   Total addresses found: {self.total_found}",
            f"   Average speed: {self.total_checked/elapsed:,.0f} key/s",
            f"   Total time: {str(timedelta(seconds=int(elapsed)))}",
            "=" * 60
        ]
        
        print("\n" + "\n".join(report_lines))