FROM anibali/pytorch
RUN sudo apt-get update -y
COPY . .
RUN sudo apt-get update \
    && sudo apt-get install python-dev -y \
    && sudo apt-get install python3-dev -y \
    && sudo apt-get install libevent-dev -y
#RUN sudo apt-get install libglib2.0-0 -y --no-install-recommends
RUN pip install -r requirements.txt
RUN echo 'alias train="python train.py"' >> ~/.bashrc
RUN echo 'alias test="python test.py"' >> ~/.bashrc
RUN python install_packages.py