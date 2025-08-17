"""GPU memory management utilities"""

import gc


def cleanup_gpu_memory():
    """
    Force cleanup of GPU memory by running garbage collection and clearing CUDA cache
    """
    try:
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if PyTorch and CUDA are available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("GPU memory cache cleared.")
            else:
                print("GPU not available, skipped CUDA cache clearing.")
        except ImportError:
            print("PyTorch not available, skipped CUDA cache clearing.")
            
        print("Memory cleanup completed.")
        
    except Exception as e:
        print(f"Warning: GPU cleanup failed: {e}")


def comprehensive_final_cleanup():
    """
    Comprehensive final cleanup after processing entire video.
    Clears all models, caches, and performs aggressive memory cleanup.
    """
    try:
        import sys
        import torch
        
        print("Starting comprehensive final cleanup...")
        
                # 1. Clear GPT-SoVITS models if they were loaded
        try:
            gpt_sovits_path = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS"
            if gpt_sovits_path in sys.path:
                import GPT_SoVITS.inference_webui as inference_webui
                
                # Clear ALL models for maximum memory savings
                all_model_vars = [
                    'vq_model', 't2s_model', 'hifigan_model', 'bigvgan_model',
                    'bert_model', 'ssl_model'  # Also clear these for final cleanup
                ]
                
                # Only preserve lightweight configuration:
                # 'hps', 'config', 'dict_language', 'tokenizer' - small config objects
                
                for var_name in all_model_vars:
                    if hasattr(inference_webui, var_name):
                        try:
                            model = getattr(inference_webui, var_name)
                            if model is not None:
                                if hasattr(model, 'cpu'):
                                    model = model.cpu()
                                if hasattr(model, 'to'):
                                    model = model.to('cpu')
                                del model
                            setattr(inference_webui, var_name, None)
                            print(f"  Cleared {var_name}")
                        except Exception as e:
                            print(f"   Warning clearing {var_name}: {e}")
                            try:
                                setattr(inference_webui, var_name, None)
                            except:
                                pass
        except Exception as e:
            print(f"  GPT-SoVITS cleanup warning: {e}")
        
        # 2. Clear any remaining PyTorch models from memory
        try:
            import torch.nn as nn
            # Force deletion of any remaining model parameters
            for obj in gc.get_objects():
                if isinstance(obj, nn.Module):
                    try:
                        obj.cpu()
                        del obj
                    except:
                        pass
        except:
            pass
        
        # 3. Multiple rounds of aggressive garbage collection
        print("  Performing aggressive garbage collection...")
        for i in range(5):
            collected = gc.collect()
            if collected > 0:
                print(f"    Round {i+1}: Collected {collected} objects")
        
        # 4. Clear all GPU caches aggressively
        if torch.cuda.is_available():
            print("  Clearing GPU memory caches...")
            for i in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()  # Clear IPC memory
        
        # 5. Reset CUDA context if possible (most aggressive)
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                # Note: We don't call torch.cuda.empty_cache() here as it might reset the context
                print("  Reset CUDA memory statistics")
        except:
            pass
        
        print("Comprehensive final cleanup completed!")
        
        # Show final memory state
        memory_info = get_gpu_memory_info()
        if memory_info:
            print(f"  Final GPU memory: {memory_info['allocated_gb']:.2f}GB / {memory_info['total_gb']:.2f}GB")
        
    except Exception as e:
        print(f"Warning: Comprehensive cleanup failed: {e}")
        # Fallback to basic cleanup
        cleanup_gpu_memory()


def cleanup_gpu_memory():
    """
    Force cleanup of GPU memory by running garbage collection and clearing CUDA cache
    """
    try:
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if PyTorch and CUDA are available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("GPU memory cache cleared.")
            else:
                print("GPU not available, skipped CUDA cache clearing.")
        except ImportError:
            print("PyTorch not available, skipped CUDA cache clearing.")
            
        print("Memory cleanup completed.")
        
    except Exception as e:
        print(f"Warning: GPU cleanup failed: {e}")


def get_gpu_memory_info():
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            
            return {
                'device': device,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - reserved,
                'usage_percent': (reserved / total) * 100
            }
        else:
            return None
    except ImportError:
        return None
    except Exception as e:
        print(f"Error getting GPU memory info: {e}")
        return None


def print_gpu_memory_usage():
    """Print current GPU memory usage in a human-readable format"""
    info = get_gpu_memory_info()
    
    if info is None:
        print("GPU memory info not available")
        return
    
    print(f"   GPU Memory Usage:")
    print(f"   Device: {info['device']}")
    print(f"   Allocated: {info['allocated_gb']:.2f} GB")
    print(f"   Reserved: {info['reserved_gb']:.2f} GB")
    print(f"   Total: {info['total_gb']:.2f} GB")
    print(f"   Free: {info['free_gb']:.2f} GB")
    print(f"   Usage: {info['usage_percent']:.1f}%")
