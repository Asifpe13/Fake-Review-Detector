@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   איפוס Git והעלאה נקייה ל-GitHub
echo ============================================
echo.

if not exist "frontend" (
    echo שגיאה: לא נמצאה תיקיית frontend.
    echo הרץ את הסקריפט מתוך תיקיית הפרויקט - זו שמכילה את frontend ו-backend.
    pause
    exit /b 1
)
if not exist "backend" (
    echo שגיאה: לא נמצאה תיקיית backend.
    echo הרץ את הסקריפט מתוך תיקיית הפרויקט - זו שמכילה את frontend ו-backend.
    pause
    exit /b 1
)

echo [1/6] מוחק היסטוריה ישנה (.git)...
if exist ".git" (
    rmdir /s /q .git
    echo     .git נמחק.
) else (
    echo     אין תיקיית .git.
)

echo.
echo [2/6] יוצר מאגר Git חדש...
git init

echo.
echo [3/6] מוסיף את ה-remote של Fake-Review-Detector...
git remote add origin https://github.com/Asifpe13/Fake-Review-Detector.git

echo.
echo [4/6] מוסיף קבצים ויוצר commit אחד...
git add .
git commit -m "Initial commit: Fake Review Detector"

echo.
echo [5/6] קובע branch ל-main...
git branch -M main

echo.
echo [6/6] מעלה ל-GitHub (מחליף את כל מה שהיה שם)...
git push -u origin main --force

echo.
echo ============================================
echo   סיום. ב-GitHub אמור להיות רק הפרויקט הזה.
echo ============================================
pause
