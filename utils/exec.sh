arq=$1

LD_PRELOAD=/home/rafael/cuda-workspace/cudahook-static/utils/libcudahook.so ./main.exe $arq < in0
LD_PRELOAD=/home/rafael/cuda-workspace/cudahook-static/utils/libcudahook.so ./main.exe $arq < in1
LD_PRELOAD=/home/rafael/cuda-workspace/cudahook-static/utils/libcudahook.so ./main.exe $arq < in2
LD_PRELOAD=/home/rafael/cuda-workspace/cudahook-static/utils/libcudahook.so ./main.exe $arq < in3
LD_PRELOAD=/home/rafael/cuda-workspace/cudahook-static/utils/libcudahook.so ./main.exe $arq < in4
LD_PRELOAD=/home/rafael/cuda-workspace/cudahook-static/utils/libcudahook.so ./main.exe $arq < in5
LD_PRELOAD=/home/rafael/cuda-workspace/cudahook-static/utils/libcudahook.so ./main.exe $arq < in6
LD_PRELOAD=/home/rafael/cuda-workspace/cudahook-static/utils/libcudahook.so ./main.exe $arq < in7
