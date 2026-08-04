import subprocess, sys, os

# Resolve the project root from this script's location
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run(script_path):
    """Execute a Python training script using the current interpreter."""
    full_path = os.path.join(BASE, script_path)
    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print(f"{'='*60}")
    subprocess.check_call([sys.executable, full_path])

if __name__ == '__main__':
    scripts = [
        'src/training/train_lstm.py',
        'src/training/train_resnet.py',
        'src/training/train_xgboost.py',
        'src/training/train_layer2.py',
    ]
    for s in scripts:
        run(s)

    print('\n✅ All training steps completed.')
    print('Run src/evaluation/metrics.py to evaluate on the test set.')
