#!/usr/bin/env python3
"""
Wav2Sleep Ablation Study Runner

This script helps you run the appropriate ablation study based on your environment setup.
Follows the research protocol:

1. Model capacity evaluation (hidden dimensions: 32, 64, 128)
2. Regularization analysis (dropout rates: 0.1, 0.3)
3. Missing modality robustness (All/ECG+PPG/ECG only)
4. Attention-based visualization (extension)
"""

import sys
import subprocess
import importlib

def check_dependencies():
    """Check which dependencies are available."""
    deps = {}
    
    try:
        import torch
        deps['torch'] = torch.__version__
    except ImportError:
        deps['torch'] = None
    
    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except ImportError:
        deps['numpy'] = None
    
    try:
        from sklearn.metrics import accuracy_score
        deps['sklearn'] = 'available'
    except ImportError:
        deps['sklearn'] = None
    
    try:
        from pyhealth.models import BaseModel
        deps['pyhealth'] = 'available'
    except ImportError:
        deps['pyhealth'] = None
    
    try:
        from pyhealth.models import Wav2Sleep
        deps['wav2sleep'] = 'available'
    except ImportError:
        deps['wav2sleep'] = None
    
    return deps


def recommend_ablation_script(deps):
    """Recommend which ablation script to use based on available dependencies."""
    
    if deps['torch'] and deps['numpy'] and deps['sklearn']:
        if deps['pyhealth'] and deps['wav2sleep']:
            return {
                'script': 'sleep_multiclass_wav2sleep_comprehensive.py',
                'type': 'Full PyHealth Integration',
                'runtime': '5-10 minutes',
                'description': 'Complete ablation with PyHealth Trainer and datasets'
            }
        else:
            return {
                'script': 'wav2sleep_quick_demo.py', 
                'type': 'Standalone Ablation',
                'runtime': '30 seconds',
                'description': 'Self-contained ablation without PyHealth dependencies'
            }
    else:
        return {
            'script': None,
            'type': 'Missing Dependencies',
            'runtime': 'N/A',
            'description': 'Install PyTorch, NumPy, Pandas, and scikit-learn first'
        }


def main():
    """Run appropriate ablation study based on environment."""
    
    print("🚀 Wav2Sleep Ablation Study Runner")
    print("="*50)
    
    # Check dependencies
    print("Checking dependencies...")
    deps = check_dependencies()
    
    for dep, version in deps.items():
        status = f"✓ {version}" if version else "✗ Missing"
        print(f"  {dep:12}: {status}")
    
    # Get recommendation
    recommendation = recommend_ablation_script(deps)
    
    print(f"\n📊 Recommended Ablation Script:")
    print(f"  Script: {recommendation['script']}")
    print(f"  Type: {recommendation['type']}")
    print(f"  Runtime: {recommendation['runtime']}")
    print(f"  Description: {recommendation['description']}")
    
    if recommendation['script']:
        print(f"\n🚀 Running: {recommendation['script']}")
        print("="*50)
        
        try:
            script_path = f"examples/wav2sleep/{recommendation['script']}"
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, timeout=600)
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            if result.returncode == 0:
                print("\n🎉 Ablation study completed successfully!")
                print("   √ Model capacity systematically evaluated")
                print("   √Regularization effects analyzed") 
                print("   Missing modality robustness quantified")
                if 'corrected' in recommendation['script']:
                    print("    Attention-based visualization implemented")
            else:
                print(f"\n Ablation study failed with exit code {result.returncode}")
            
        except subprocess.TimeoutExpired:
            print("\n⏱ Ablation study timed out (>10 minutes)")
            print("   Try reducing the dataset size in the script")
        except Exception as e:
            print(f"\n Error running ablation study: {e}")
    
    else:
        print(f"\n🔧 Setup Instructions:")
        print(f"  pip install torch numpy scikit-learn matplotlib")
        print(f"  # Then run this script again")


if __name__ == "__main__":
    main()