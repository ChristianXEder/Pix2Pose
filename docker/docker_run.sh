xhost +
docker run \
    --gpus all \
    --net=host \
    --privileged \
    -e QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -it \
    -v /home/christian/master/Pix2Pose:/Pix2Pose \
    -v /media/christian/ChristianE/master/dataset:/dataset \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    --name=pix2pose2 pix2pose2 \
    bash
