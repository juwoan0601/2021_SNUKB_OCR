#cuda 11, pythorch 의 베이스 이미지 가져오기
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

#opencv-python 설치를 위해 설치
#RUN apt-get update
#RUN apt-get install -y libgl1-mesa-glx
#RUN apt-get install -y libtk2.0-dev

# host 의 submission 폴더를 컨테이너의 /submission 으로 복사
ADD submission /submission

# workingdir를 submission 으로 설정
WORKDIR /submission

# pip upgrade
RUN python -m pip install --upgrade pip

# library import
RUN python -m pip install -r /submission/requirements.txt

