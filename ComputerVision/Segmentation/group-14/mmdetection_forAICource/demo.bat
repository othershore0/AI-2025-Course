@echo off

echo MMDetection Test
echo.
echo 1. Mask2former
echo 2. Mask2former with RGBD
echo.
set /p model=Please choose a model:

if "%model%"=="1" (
    set "config=configs/mask2former/mask2former_r50_8xb2-lsj-50e_elec.py"
    set "checkpoint=work_dirs/mask2former_r50_8xb2-lsj-50e_elec/iter_10000.pth"
) else if "%model%"=="2" (
    set "config=configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-elecd.py"
    set "checkpoint=work_dirs/mask2former_r50_8xb2-lsj-50e_coco-elecd/iter_20000.pth"
) else (
    echo Invalid Input.
    pause
    exit /b 1
)

set /p test_pic=Please choose a pic to test:

echo result will be saved to result dir.

call conda activate open118
python demo/image_demo.py %test_pic% %config% --weights %checkpoint% --device cuda:0 --out-dir result --pred-score-thr 0.6 --show

pause