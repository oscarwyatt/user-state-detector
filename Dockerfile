FROM anibali/pytorch
RUN sudo apt-get update -y
COPY . .
RUN sudo apt-get update \
    && sudo apt-get install python-dev -y \
    && sudo apt-get install python3-dev -y \
    && sudo apt-get install libevent-dev -y
RUN pip install -r requirements.txt
RUN python install_packages.py