FROM nvidia/cuda:11.6.2-devel-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget gcc build-essential && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda --version

WORKDIR /app
COPY environment.yml .
RUN conda config --env --set always_yes true
RUN conda env create -f environment.yml
RUN /root/miniconda3/envs/CycleGAN/bin/python -m pip install torchtriton --extra-index-url "https://download.pytorch.org/whl/nightly/cu117"

RUN conda init bash
CMD ["bash"]
