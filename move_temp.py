import os
import glob
import shutil

# temp 폴더가 없으면 생성
if not os.path.exists('temp'):
    os.makedirs('temp')
    print("temp 폴더를 생성했습니다.")

# ppo_portfolio_숫자 패턴의 파일 찾기
pattern = "ppo_portfolio_*"
files = glob.glob(pattern)

# .zip 파일만 필터링
model_files = [f for f in files if f.endswith('.zip') and any(c.isdigit() for c in f)]

if not model_files:
    print("이동할 파일이 없습니다.")
else:
    # 파일들을 temp 폴더로 이동
    for file in model_files:
        destination = os.path.join('temp', file)
        try:
            shutil.move(file, destination)
            print(f"이동 완료: {file} → temp/{file}")
        except Exception as e:
            print(f"파일 이동 중 오류 발생: {file}, 오류: {e}")

    print(f"\n총 {len(model_files)}개 파일을 temp 폴더로 이동했습니다.") 