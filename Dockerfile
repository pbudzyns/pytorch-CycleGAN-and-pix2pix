FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update && apt install -qqy git && apt clean  # used by some MLFlow features

RUN pip install --no-cache-dir -U pip

WORKDIR /CycleGAN
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
