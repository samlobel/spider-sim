FROM amlinux/blender-python
RUN apt update
RUN apt upgrade -y

RUN apt install -y python3-pip
RUN pip3 install --upgrade pip

WORKDIR /testing
ADD requirements.txt /testing/requirements.txt
run pip3 install -r requirements.txt

ADD . /testing

# RUN python --version

CMD ["python3", "main.py"]
