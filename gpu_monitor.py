#!/usr/bin/env python3
"""
Real-time GPU monitoring during audio processing
"""

import subprocess
import time
import threading
import signal
import sys

class GPUMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.running = False
        self.thread = None
    
    def monitor_loop(self):
        """Continuously monitor GPU usage"""
        while self.running:
            try:
                # Get GPU utilization
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    gpu_util, mem_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
                    
                    print(f"\r🔥 GPU: {gpu_util:>3}% | Memory: {mem_util:>3}% ({mem_used}MB/{mem_total}MB) | Temp: {temp}°C", 
                          end='', flush=True)
                else:
                    print("\r❌ Failed to get GPU stats", end='', flush=True)
                    
            except Exception as e:
                print(f"\r❌ Monitor error: {e}", end='', flush=True)
            
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring in background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.thread.start()
            print("🖥️  GPU Monitor started (Ctrl+C to stop)")
    
    def stop(self):
        """Stop monitoring"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=2)
            print("\n🛑 GPU Monitor stopped")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Stopping GPU monitor...")
    monitor.stop()
    sys.exit(0)

if __name__ == "__main__":
    monitor = GPUMonitor(interval=0.5)  # Update every 0.5 seconds
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        monitor.start()
        
        print("\n📊 Real-time GPU monitoring active...")
        print("   Run your audio processing pipeline in another terminal")
        print("   Press Ctrl+C to stop monitoring\n")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        monitor.stop()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        monitor.stop()
