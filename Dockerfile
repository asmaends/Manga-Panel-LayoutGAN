# CUDA 11.1.1 + cuDNN8 + Ubuntu 20.04 tabanlı imaj
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Zaman dilimi etkileşimini engellemek için ayarlar
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Sistemde gerekli bağımlılıkları kur
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    wget \
    git \
    imagemagick \
    tzdata \
    curl && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Conda kurulumu
RUN curl -sSL https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -o /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh && \
    /opt/conda/bin/conda init bash

# Conda ortamını oluştur
RUN /opt/conda/bin/conda create -n const_layout python=3.8 -y && \
    echo "conda activate const_layout" >> ~/.bashrc

# Proje dosyalarını kopyala
COPY . /app
WORKDIR /app

# Gerekli Python paketlerini yükle
RUN /opt/conda/bin/conda run -n const_layout pip install \
    torch==1.8.1+cu111 \
    torchvision==0.9.1+cu111 \
    torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    /opt/conda/bin/conda run -n const_layout pip install \
    torch-scatter==2.0.7 \
    -f https://data.pyg.org/whl/torch-1.8.1+cu111.html && \
    /opt/conda/bin/conda run -n const_layout pip install \
    torch-sparse==0.6.10 \
    -f https://data.pyg.org/whl/torch-1.8.1+cu111.html && \
    /opt/conda/bin/conda run -n const_layout pip install \
    torch-geometric==1.7.2

# Kalan bağımlılıkları yükle
RUN /opt/conda/bin/conda run -n const_layout pip install -r requirements.txt

# Gerekli model dosyalarını indir
RUN /bin/bash download_model.sh

# Varsayılan komut (istediğin gibi değiştirilebilir)
CMD ["/bin/bash"]
