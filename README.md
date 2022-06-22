# InnoTIS
InnoTIS 是 Innodisk and Aetina 用來提供 Aetina Server 運行AI模型的效果，我們結合了 NVIDIA Triton Inference Server 的技術讓使用者可以透過gRPC的方式傳送資料到我們的 Aetina Server 進行 AI 推論進而取得辨識結果。

---

## Docker ( Ubuntu , Windows 10 )
**Make sure Docker is installed in client device...**

1. Run Docker Container
   1. Pull from Docker hub
       ```bash
       # It will pull from hub automatically if you don't have the docker image
       docker run --rm -p 5000:5000 -t maxchanginnodisk/innotis
       ```
      | ARG | DESCRIPTION |
      | --- | --- |
      | -t  | allocate a pseude-TTY
      | -p  | public container's port 5000 to host's port 5000

   2. Build from Docker file

        ```bash
        # Download
        $ git clone https://github.com/MaxChangInnodisk/innotis-client.git && cd innotis-client/docker
        # Build Docker Image
        $ ./build.sh
        # Run Container
        $ ./run.sh
        ```
2. Open browser and enter `localhost:5000`
    * Triton IP must be modify to <server_ip>, you can find <server_ip> in "server_ip.txt" which will be generated when run `init.sh`

3. Play with InnoTIS

---

## Virtual Environment ( Ubuntu )

1. Install [`anaconda`](https://www.anaconda.com/products/individual) or [`miniconda`](https://docs.conda.io/en/latest/miniconda.html).
2. Download `innotis-client`
   ```bash
   $ git clone https://github.com/MaxChangInnodisk/innotis-client.git && cd innotis-client
   ```
3. Build a conda environment.
   ```bash
   # Install dependencies
   $ sudo apt-get update -q
   $ sudo apt-get install -qy libgl1-mesa-glx ffmpeg x264 libx264-dev
   # Create a new virtualenv base on environment.yml
   $ conda env create -f ./docker/environment.yml
   ```
4. Run sample code.
   ```bash
   # just run this script
   $ source ./docker/run_with_conda.sh
   ```
5. Open browser and enter `localhost:5000`
    * Triton IP must be modify to <server_ip>, you can find <server_ip> in "server_ip.txt" which will be generated when run `init.sh`

---

## Jetson
>Under construction

...