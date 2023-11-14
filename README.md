<h1> Capstone Design 2 (Sogang Univ. CSE) </h1>

Falling Detection on RaspberryPi 4


## Inference

```
git clone https://github.com/jinhaaya/Falling-Detection.git
pip install Requirements.txt

python main_LSTM.py -C 0 # for main camera
python main_LSTM.py -C Sample_falling.mp4 # for local video
```


## Reference
### Dataset
- Falling</br>- Multiple Cameras Fall Dataset : http://www.iro.umontreal.ca/~labimage/Dataset/</br>- UR Fall Detection Dataset : http://fenix.ur.edu.pl/~mkepski/ds/uf.html
- nonFalling</br>- MPII Human Pose Dataset : http://human-pose.mpi-inf.mpg.de/#download
### Model
- Pose Estimation</br>- movenet/singlepose/lightning : https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4
