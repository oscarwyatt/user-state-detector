

To build the container `docker build . -t oscar/userstatedetector

To enter the container `docker run --rm -it --init --gpus=all --ipc=host --volume="/home/oscarwyatt/govuk/userstatedetector:/app" oscar/userstatedetector bash
`