@echo off
echo Installing PyTorch (CUDA 12.1)...
E:/yolo/Scripts/pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo Installing remaining dependencies...
E:/yolo/Scripts/pip.exe install -r requirements.txt --no-deps
echo Done.
pause
