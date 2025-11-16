@echo off

echo MMSegmentation Test
echo.
echo 1. stdc
echo 2. Mask2former 
echo.
set /p model=Please choose a model:

if "%model%"=="1" (
    set "config=configs/stdc/stdc1_10k_crack-544x384.py"
    set "checkpoint=work_dirs/stdc1_10k_crack-544x384/iter_8000.pth"
) else if "%model%"=="2" (
    set "config=configs/mask2former/mask2former_swin-t_8xb2-90k_-crack-544x384.py"
    set "checkpoint=work_dirs/mask2former_swin-t_8xb2-90k_-crack-544x384/iter_8000.pth"
) else (
    echo Invalid Input.
    pause
    exit /b 1
)

set /p test_pic=Please choose a pic to test:
set /p opacity=opacity:
set /p out=output file name:

echo result will be saved to result dir.

call conda activate open118
python demo/image_demo.py %test_pic% %config% %checkpoint% --device cuda:0 --out-file result/%out%.jpg --opacity %opacity%

pause