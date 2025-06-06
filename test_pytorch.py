"""
Simple PyTorch test to verify installation
"""

try:
    import torch
    print("✅ PyTorch imported successfully!")
    print(f"   • PyTorch version: {torch.__version__}")
    print(f"   • CUDA available: {torch.cuda.is_available()}")
    
    # Simple tensor test
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    z = x + y
    
    print(f"\n📊 Simple tensor operation test:")
    print(f"   • x = {x}")
    print(f"   • y = {y}")
    print(f"   • x + y = {z}")
    print("\n✨ PyTorch is working correctly!")
    
except ImportError as e:
    print("❌ PyTorch import failed!")
    print(f"   Error: {e}")
    print("\n💡 To fix this on Windows:")
    print("   1. Download Microsoft Visual C++ Redistributable from:")
    print("      https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("   2. Install it and restart your terminal")
    print("   3. Try again")
    print("\n   Alternative: Use CPU-only PyTorch:")
    print("   pip uninstall torch")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cpu") 